import os
import glob
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm
from torch.nn.utils import spectral_norm
from pytorch_wavelets import DWTForward, DWTInverse
from torch.nn import Identity
import argparse
import wandb
from lpips import LPIPS
from torch_fidelity import calculate_metrics
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import numpy as np

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU cache cleared. Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")


DATASET_PATH = "/home/tanmoyhazra/gan_project/data/"
SAVE_DIR = "/home/tanmoyhazra/gan_project/checkpoints_gw"
SAMPLE_DIR = "/home/tanmoyhazra/gan_project/samples_gw"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)


IMG_SIZE = 512
BATCH_SIZE = 16
LATENT_DIM = 128
EPOCHS = 100
LR = 1e-4
BETA1 = 0.0
BETA2 = 0.999
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAD_CLIP = 0.5
GRAD_ACCUM_STEPS = 4


LAMBDA_REC = 20.0
LAMBDA_KL = 0.1
LAMBDA_ADV = 1.0
LAMBDA_FM = 10.0
LAMBDA_PERCEPT = 5.0

# KL Annealing (Exponential)
def kl_anneal(epoch, batch_idx, total_batches):
    return min(1.0, 1 - 0.99 ** (epoch * total_batches + batch_idx))

# -----------------------------
# Data Preprocessing & Augmentation
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True) + \
                     glob.glob(os.path.join(root, '**', '*.png'), recursive=True)
        if not self.files:
            print(f"!!! WARNING: No image files found in {root}.")
        self.transform = transform
        print(f"Found {len(self.files)} images in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert('L')
            img_rgb = Image.merge('RGB', (img, img, img))
            if self.transform:
                img_rgb = self.transform(img_rgb)
            return img_rgb
        except Exception as e:
            print(f"Error loading image {self.files[idx]}: {e}")
            return torch.zeros((3, IMG_SIZE, IMG_SIZE))

# -----------------------------
# Res Blocks with db1 Wavelet (V7)
# -----------------------------
class ResBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch, use_spectral_norm=False, has_bn=True):
        super().__init__()
        self.dwt = DWTForward(J=1, wave='db1', mode='zero')
        conv_layer = nn.Conv2d(in_ch * 4, out_ch, 3, 1, 1, bias=not has_bn)
        self.conv1 = spectral_norm(conv_layer) if use_spectral_norm else conv_layer
        self.bn1 = nn.BatchNorm2d(out_ch) if has_bn else Identity()
        conv_layer2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=not has_bn)
        self.conv2 = spectral_norm(conv_layer2) if use_spectral_norm else conv_layer2
        self.bn2 = nn.BatchNorm2d(out_ch) if has_bn else Identity()
        skip_layer = nn.Conv2d(in_ch * 4, out_ch, 1, 1, 0, bias=not has_bn)
        self.skip = spectral_norm(skip_layer) if use_spectral_norm else skip_layer
        self.act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        LL, H_bands = self.dwt(x)
        H = H_bands[0]
        LH = H[:, :, 0, :, :]
        HL = H[:, :, 1, :, :]
        HH = H[:, :, 2, :, :]
        x_dwt = torch.cat([LL, LH, HL, HH], dim=1)

        skip_out = self.skip(x_dwt)
        out = self.conv1(x_dwt)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + skip_out
        out = self.act(out)
        return out

class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch, use_spectral_norm=False, has_bn=True):
        super().__init__()
        self.iwt = DWTInverse(wave='db1', mode='zero')
        conv_layer = nn.Conv2d(in_ch, out_ch * 4, 3, 1, 1, bias=not has_bn)
        self.conv1 = spectral_norm(conv_layer) if use_spectral_norm else conv_layer
        self.bn1 = nn.BatchNorm2d(out_ch * 4) if has_bn else Identity()
        conv_layer2 = nn.Conv2d(out_ch * 4, out_ch * 4, 3, 1, 1, bias=not has_bn)
        self.conv2 = spectral_norm(conv_layer2) if use_spectral_norm else conv_layer2
        self.bn2 = nn.BatchNorm2d(out_ch * 4) if has_bn else Identity()
        skip_layer = nn.Conv2d(in_ch, out_ch * 4, 1, 1, 0, bias=not has_bn)
        self.skip = spectral_norm(skip_layer) if use_spectral_norm else skip_layer
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        skip_out = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + skip_out
        out = self.act(out)

        b, c, h, w = out.shape
        assert c % 4 == 0
        c_part = c // 4
        LL = out[:, :c_part, :, :]
        LH = out[:, c_part:2*c_part, :, :]
        HL = out[:, 2*c_part:3*c_part, :, :]
        HH = out[:, 3*c_part:, :, :]

        H_stacked = torch.stack([LH, HL, HH], dim=2)
        H_bands_list = [H_stacked]

        return self.iwt((LL, H_bands_list))

# -----------------------------
# VAE-GAN Models (V7)
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, latent_dim=LATENT_DIM):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1, bias=False)
        self.initial_bn = nn.BatchNorm2d(base_ch)
        self.initial_act = nn.LeakyReLU(0.2, inplace=False)
        self.rb1 = ResBlockDown(base_ch, base_ch * 2, use_spectral_norm=False, has_bn=True)
        self.rb2 = ResBlockDown(base_ch * 2, base_ch * 4, use_spectral_norm=False, has_bn=True)
        self.rb3 = ResBlockDown(base_ch * 4, base_ch * 8, use_spectral_norm=False, has_bn=True)
        self.rb4 = ResBlockDown(base_ch * 8, base_ch * 8, use_spectral_norm=False, has_bn=True)
        self.rb5 = ResBlockDown(base_ch * 8, base_ch * 8, use_spectral_norm=False, has_bn=True)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(base_ch * 8, latent_dim)
        self.fc_logvar = nn.Linear(base_ch * 8, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial_conv(x); x = self.initial_bn(x); x = self.initial_act(x)
        x = self.rb1(x); x = self.rb2(x); x = self.rb3(x); x = self.rb4(x); x = self.rb5(x)
        x = self.avg(x).flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=3, base_ch=64, latent_dim=LATENT_DIM):
        super().__init__()
        s_init = IMG_SIZE // 32; self.s_init = s_init
        self.fc = nn.Linear(latent_dim, base_ch * 8 * s_init * s_init)
        self.post_fc_bn = nn.BatchNorm2d(base_ch * 8)
        self.rb1 = ResBlockUp(base_ch * 8, base_ch * 8, use_spectral_norm=False, has_bn=True)
        self.rb2 = ResBlockUp(base_ch * 8, base_ch * 8, use_spectral_norm=False, has_bn=True)
        self.rb3 = ResBlockUp(base_ch * 8, base_ch * 4, use_spectral_norm=False, has_bn=True)
        self.rb4 = ResBlockUp(base_ch * 4, base_ch * 2, use_spectral_norm=False, has_bn=True)
        self.rb5 = ResBlockUp(base_ch * 2, base_ch, use_spectral_norm=False, has_bn=True)
        self.final_conv = nn.Conv2d(base_ch, out_ch, 3, 1, 1)
        self.tanh = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, z, return_pre_tanh=False):
        x = self.fc(z).view(z.size(0), -1, self.s_init, self.s_init)
        x = self.post_fc_bn(x)
        x = torch.tanh(x) * 2.0
        x = self.rb1(x); x = self.rb2(x); x = self.rb3(x); x = self.rb4(x); x = self.rb5(x)
        pre_tanh_output = self.final_conv(x)
        scale_factor = 5.0
        pre_tanh_output = torch.tanh(pre_tanh_output / scale_factor) * scale_factor
        pre_tanh_output = torch.clamp(pre_tanh_output, -5.0, 5.0)
        output = self.tanh(pre_tanh_output)
        if return_pre_tanh: return output, pre_tanh_output
        else: return output

class Discriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.initial = spectral_norm(nn.Conv2d(in_ch, base_ch, 3, 1, 1))
        self.initial_act = nn.LeakyReLU(0.2, inplace=False)
        self.rb1 = ResBlockDown(base_ch, base_ch * 2, use_spectral_norm=True, has_bn=False)
        self.rb2 = ResBlockDown(base_ch * 2, base_ch * 4, use_spectral_norm=True, has_bn=False)
        self.rb3 = ResBlockDown(base_ch * 4, base_ch * 8, use_spectral_norm=True, has_bn=False)
        self.rb4 = ResBlockDown(base_ch * 8, base_ch * 8, use_spectral_norm=True, has_bn=False)
        self.rb5 = ResBlockDown(base_ch * 8, base_ch * 8, use_spectral_norm=True, has_bn=False)
        self.final_conv = spectral_norm(nn.Conv2d(base_ch * 8, 1, 1, 1, 0))

    def forward(self, x):
        fmaps = []
        x = self.initial(x); x = self.initial_act(x)
        x = self.rb1(x); fmaps.append(x)
        x = self.rb2(x); fmaps.append(x)
        x = self.rb3(x); fmaps.append(x)
        x = self.rb4(x); fmaps.append(x)
        x = self.rb5(x); fmaps.append(x)
        out = self.final_conv(x)
        return out, fmaps

# -----------------------------
# Utilities & Losses
# -----------------------------
def reparameterize(mu, logvar):
    logvar = torch.clamp(logvar, min=-10, max=10)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def hinge_d_loss(real_out, fake_out):
    return torch.mean(F.relu(1.0 - real_out)) + torch.mean(F.relu(1.0 + fake_out))

def hinge_g_loss(fake_out):
    return -torch.mean(fake_out)

def denorm(x):
    return (x + 1) / 2


# -----------------------------
# Eval Function
# -----------------------------
def evaluate_model(E, G, val_loader, device, num_fakes=1024, num_recon=500):
    
    # --- FIX 1: (from user) ADD ROBUST HELPER CLASS ---
    class TensorDatasetWrapper(Dataset):
        def __init__(self, tensor):
            assert isinstance(tensor, torch.Tensor), "TensorDatasetWrapper expects a torch.Tensor"
            # Expect shape: (N, C, H, W) and dtype torch.uint8 (0-255)
            self.tensor = tensor

        def __len__(self):
            return self.tensor.shape[0]

        def __getitem__(self, index):
            img = self.tensor[index]  # shape: (C, H, W)
            
            # --- FINAL FIX: REMOVE THE PERMUTE ---
            # The library expects (C, H, W) after all.
            
            # Return a tensor, not a tuple
            return img.contiguous()
    # --- END OF CLASS ---

    E.eval(); G.eval()
    
    # --- FIX 3: Use the 'device' argument, not global 'DEVICE' ---
    lpips_net = LPIPS(net='vgg').to(device).eval()
    
    # Generate fakes
    fake_imgs = []
    # --- FIX 1: Set a safe batch size for evaluation ---
    batch_size_eval = 16
    with torch.no_grad():
        for i in range(0, num_fakes, batch_size_eval):
            z = torch.randn(min(batch_size_eval, num_fakes - i), LATENT_DIM, device=device)
            fakes = G(z)
            fake_imgs.append(fakes.cpu())
            torch.cuda.empty_cache()
    fake_imgs = torch.cat(fake_imgs)
    
    # --- FIX 2: CONVERT TO UINT8 FOR FID ---
    fake_denorm = (denorm(fake_imgs) * 255).to(torch.uint8)
    # --- FIX 2: FREE MEMORY ---
    del fake_imgs
    torch.cuda.empty_cache()
    
    # Sample reals (Using val_loader to be consistent)
    real_imgs = []
    real_iter = iter(val_loader) # Use val_loader for FID reals
    num_batches_needed = int(np.ceil(num_fakes / BATCH_SIZE))
    
    for _ in range(num_batches_needed):
        try:
            # --- FIX 1: Remove [0] ---
            real_batch = next(real_iter)
            real_imgs.append(real_batch)
        except StopIteration:
            real_iter = iter(val_loader) # Reset iterator if dataset is exhausted
            # --- FIX 1: Remove [0] ---
            real_batch = next(real_iter)
            real_imgs.append(real_batch)
    
    real_imgs = torch.cat(real_imgs)[:num_fakes]
    # --- FIX 2: CONVERT TO UINT8 FOR FID ---
    real_denorm = (denorm(real_imgs) * 255).to(torch.uint8)
    # --- FIX 2: FREE MEMORY ---
    del real_imgs
    torch.cuda.empty_cache()
    
    # --- WRAP TENSORS IN DATASET ---
    fake_dataset = TensorDatasetWrapper(fake_denorm)
    real_dataset = TensorDatasetWrapper(real_denorm)
    
    # FID & IS
    # --- FIX 3: Add batch_size and use .get() ---
    fid_metrics = calculate_metrics(
        input1=fake_dataset, 
        input2=real_dataset, 
        cuda=True, 
        fid=True, 
        isc=True, 
        batch_size=batch_size_eval
    )
    fid_score = fid_metrics.get('frechet_inception_distance')
    is_score = fid_metrics.get('inception_score_mean')
    
    # Recon Metrics
    rec_ssim, rec_psnr, rec_lpips = [], [], []
    recon_iter = iter(val_loader) # Use val_loader for reconstruction metrics
    num_recon_batches = int(np.ceil(num_recon / BATCH_SIZE))

    with torch.no_grad():
        for _ in range(num_recon_batches):
            try:
                # --- FIX 1: Remove [0] ---
                x = next(recon_iter).to(device)
            except StopIteration:
                recon_iter = iter(val_loader) # Reset iterator
                # --- FIX 1: Remove [0] ---
                x = next(recon_iter).to(device)
                
            mu, logvar = E(x)
            z = reparameterize(mu, logvar)
            rec = G(z)
            
            # LPIPS uses [-1, 1] float range
            rec_lpips.append(lpips_net(x, rec).mean().item())

            # Use [0, 1] float range for ssim/psnr
            for j in range(x.size(0)):
                # --- FIX 4: Removed .squeeze() ---
                real_np_img = denorm(x[j].cpu().numpy()).mean(0)
                rec_np_img = denorm(rec[j].cpu().numpy()).mean(0)
                rec_ssim.append(ssim(real_np_img, rec_np_img, data_range=1.0))
                rec_psnr.append(psnr(real_np_img, rec_np_img, data_range=1.0))

    avg_ssim = np.mean(rec_ssim)
    avg_psnr = np.mean(rec_psnr)
    avg_lpips = np.mean(rec_lpips)
    
    E.train(); G.train()
    return fid_score, is_score, avg_ssim, avg_psnr, avg_lpips

# -----------------------------
# Main Training Script
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    
    print("--- HPC V8.4 Wavelet-VAE-GAN (All Fixes) ---")
    print(f"Device: {DEVICE}")
    
    # Dataset & Split (80/20)
    full_dataset = ImageDataset(DATASET_PATH, transform=transform)
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {DATASET_PATH}.")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Models
    E = Encoder().to(DEVICE)
    G = Decoder().to(DEVICE)
    D = Discriminator().to(DEVICE)
    
    # Perceptual Net
    # --- FIX 3: REMOVED pnet_path AND FIXED SYNTAX ---
    lpips_net = LPIPS(net='vgg').to(DEVICE).eval()
    
    # Optimizers
    opt_EG = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), 
                                lr=LR, betas=(BETA1, BETA2), weight_decay=1e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2), weight_decay=1e-4)
    
    # Schedulers (Init BEFORE resume to load states correctly)
    total_epochs_to_run = args.epochs
    scheduler_EG = torch.optim.lr_scheduler.CosineAnnealingLR(opt_EG, T_max=total_epochs_to_run, eta_min=1e-6)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=total_epochs_to_run, eta_min=1e-6)
    
    # WandB
    if args.wandb:
        wandb.init(project="wavelet-vaegan-glioma-hpc", config={"epochs": args.epochs, "batch_size": BATCH_SIZE})
    
    # --- FIX 4: HARD-CODE 16 SAMPLES ---
    z_fixed = torch.randn(16, LATENT_DIM, device=DEVICE)
    start_epoch = 0
    
    # ===================================================================
    # FIXED CHECKPOINT RESUME LOGIC
    # ===================================================================
    if args.resume:
        checkpoint_files = glob.glob(os.path.join(SAVE_DIR, "checkpoint_epoch_*.pth"))
        if checkpoint_files:
            # Sort by epoch number
            latest_checkpoint_path = max(checkpoint_files, key=lambda f: int(re.search(r'_(\d+)\.pth$', f).group(1)))
            
            print(f"Resuming from checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=DEVICE)
            
            E.load_state_dict(checkpoint['E'])
            G.load_state_dict(checkpoint['G'])
            D.load_state_dict(checkpoint['D'])
            opt_EG.load_state_dict(checkpoint['opt_EG'])
            opt_D.load_state_dict(checkpoint['opt_D'])
            scheduler_EG.load_state_dict(checkpoint['scheduler_EG'])
            scheduler_D.load_state_dict(checkpoint['scheduler_D'])
            
            start_epoch = checkpoint['epoch'] + 1 # e.g., saved epoch 5 -> start 6
            
            print(f"Resumed. Starting from Epoch {start_epoch}")
            # Adjust scheduler T_max for remaining epochs (optional but precise)
            remaining_epochs = args.epochs - (start_epoch - 1) # If resumed at 6, remaining=95 for 100 total
            scheduler_EG = torch.optim.lr_scheduler.CosineAnnealingLR(opt_EG, T_max=remaining_epochs, eta_min=1e-6)
            scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=remaining_epochs, eta_min=1e-6)
        else:
            print("`--resume` was specified but no checkpoint was found. Starting fresh.")
    else:
        print("Starting fresh (no resume).")
    # ===================================================================
    
    print(f"Training from Epoch {start_epoch} for {args.epochs} epochs.")
    
    final_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, final_epoch):
        # FIXED GRAD ACCUM: Zero once per epoch
        opt_EG.zero_grad()
        accum_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{final_epoch}")
        epoch_d_loss, epoch_g_loss, epoch_l1, epoch_kl = 0.0, 0.0, 0.0, 0.0
        num_batches = 0
        
        for i, x in enumerate(pbar):
            if x is None or x.nelement() == 0: continue
            x = x.to(DEVICE)
            num_batches += 1
            
            # KL Weight (Exponential)
            kl_weight = kl_anneal(epoch, i, len(train_loader))
            
            # Train D
            opt_D.zero_grad()
            with torch.no_grad():
                mu_no_grad, logvar_no_grad = E(x)
                z_no_grad = reparameterize(mu_no_grad, logvar_no_grad)
                x_rec_no_grad = G(z_no_grad)
            
            real_out, _ = D(x)
            fake_out, _ = D(x_rec_no_grad.detach())
            d_loss = hinge_d_loss(real_out, fake_out)
            
            if torch.isnan(d_loss):
                print(f"NaN in D loss. Skipping batch {i}.")
                continue
            
            d_loss.backward()
            opt_D.step()
            epoch_d_loss += d_loss.item()
            
            # Train EG (with accum)
            mu, logvar = E(x)
            
            if torch.isnan(mu).any() or torch.isinf(mu).any() or torch.isnan(logvar).any() or torch.isinf(logvar).any():
                print(f"NaN/Inf in Encoder. Skipping batch {i}.")
                continue
            
            z = reparameterize(mu, logvar)
            x_rec = G(z)
            
            rec_l1 = F.l1_loss(x_rec, x)
            logvar_clamped = torch.clamp(logvar, min=-10, max=10)
            kl_loss = -0.5 * torch.mean(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                kl_loss = torch.tensor(0.0, device=DEVICE)
            
            fake_out_for_g, fmaps_fake = D(x_rec)
            adv_loss = hinge_g_loss(fake_out_for_g)
            
            with torch.no_grad():
                _, fmaps_real = D(x)
            fm_loss = sum(F.l1_loss(ff, fr.detach()) for ff, fr in zip(fmaps_fake, fmaps_real))
            
            # Perceptual
            perceptual_loss = lpips_net(x, x_rec).mean()
            
            total_g_loss = (LAMBDA_REC * rec_l1) + (LAMBDA_KL * kl_weight * kl_loss) + \
                           (LAMBDA_ADV * adv_loss) + (LAMBDA_FM * fm_loss) + (LAMBDA_PERCEPT * perceptual_loss)
            
            if torch.isnan(total_g_loss):
                print(f"NaN in G loss. Skipping batch {i}.")
                continue
            
            # Grad Accum
            (total_g_loss / GRAD_ACCUM_STEPS).backward()
            accum_count += 1
            if accum_count % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(list(E.parameters()) + list(G.parameters()), GRAD_CLIP)
                opt_EG.step()
                opt_EG.zero_grad()
                accum_count = 0
            
            epoch_g_loss += total_g_loss.item()
            epoch_l1 += rec_l1.item()
            epoch_kl += kl_loss.item()
            
            pbar.set_postfix({
                'D': d_loss.item(), 'G': total_g_loss.item(),
                'L1': rec_l1.item(), 'KL': kl_loss.item(), 'KLW': kl_weight,
                'LR': scheduler_EG.get_last_lr()[0]
            })
        
        # FIXED: Handle last incomplete accum batch
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(list(E.parameters()) + list(G.parameters()), GRAD_CLIP)
            opt_EG.step()
        
        avg_d_loss = epoch_d_loss / num_batches if num_batches > 0 else 0
        avg_g_loss = epoch_g_loss / num_batches if num_batches > 0 else 0
        avg_l1 = epoch_l1 / num_batches if num_batches > 0 else 0
        avg_kl = epoch_kl / num_batches if num_batches > 0 else 0
        print(f"\nEpoch {epoch + 1} Losses: D={avg_d_loss:.4f}, G={avg_g_loss:.4f}, L1={avg_l1:.4f}, KL={avg_kl:.4f}")
        
        if args.wandb:
            wandb.log({"epoch": epoch + 1, "d_loss": avg_d_loss, "g_loss": avg_g_loss,
                       "l1": avg_l1, "kl": avg_kl, "lr": scheduler_EG.get_last_lr()[0]})
        
        scheduler_EG.step()
        scheduler_D.step()
        
        # Debug Print (Always - Lightweight)
        with torch.no_grad():
            G.eval()
            _, pre_tanh_imgs = G(z_fixed, return_pre_tanh=True)
            pre_tanh_imgs = pre_tanh_imgs.cpu()
            print(f"Epoch {epoch + 1} Debug: Pre-Tanh Min/Max/Mean: {pre_tanh_imgs.min().item():.4f}/{pre_tanh_imgs.max().item():.4f}/{pre_tanh_imgs.mean().item():.4f}")
            G.train()
        
        # --- MODIFIED LOGIC: SAVE FIRST, THEN EVALUATE ---
        
        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Saving checkpoint and samples at Epoch {epoch + 1}")
            with torch.no_grad():
                G.eval()
                imgs, _ = G(z_fixed, return_pre_tanh=True)
                imgs = imgs.cpu()
                imgs_to_save = denorm(imgs)
                grid = utils.make_grid(imgs_to_save, nrow=4, padding=2) # 4x4 grid for 16 images
                utils.save_image(grid, os.path.join(SAMPLE_DIR, f"samples_epoch_{epoch + 1:04d}.png"))
                if args.wandb:
                    wandb.log({"epoch": epoch + 1, "samples": wandb.Image(grid)})
                G.train()
            
            torch.save({
                'epoch': epoch, # Save as 0-indexed completed epoch
                'E': E.state_dict(),
                'G': G.state_dict(),
                'D': D.state_dict(),
                'opt_EG': opt_EG.state_dict(),
                'opt_D': opt_D.state_dict(),
                'scheduler_EG': scheduler_EG.state_dict(),
                'scheduler_D': scheduler_D.state_dict()
            }, os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch + 1:04d}.pth"))

        # Evaluate every 25 epochs
        if (epoch + 1) % 25 == 0 or (epoch+1 == 21):
            print(f"--- Starting Evaluation for Epoch {epoch + 1} ---")
            fid, is_score, avg_ssim, avg_psnr, avg_lpips = evaluate_model(E, G, val_loader, DEVICE) # Use val_loader
            print(f"Epoch {epoch + 1} Eval: FID={fid:.2f}, IS={is_score:.2f}, SSIM={avg_ssim:.4f}, PSNR={avg_psnr:.2f}, LPIPS={avg_lpips:.4f}")
            if args.wandb:
                wandb.log({"epoch": epoch + 1, "fid": fid, "is": is_score, "ssim": avg_ssim, "psnr": avg_psnr, "lpips_val": avg_lpips})
            print(f"--- Finished Evaluation for Epoch {epoch + 1} ---")
    
    if args.wandb:
        wandb.finish()
    print("Training completed!")
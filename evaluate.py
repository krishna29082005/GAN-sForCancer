import os, copy, math, random, torch, glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm
from torch.nn.utils import spectral_norm

# ---------------- CONFIG (Same as your trainer) ----------------
DATASET_PATH = r"/home/tanmoyhazra/gan_project/data/brain_glioma"
SAVE_DIR = r"/home/tanmoyhazra/gan_project/checkpoints1"
SAMPLE_DIR = r"/home/tanmoyhazra/gan_project/samples1_eval_skip" 

INITIAL_SIZE = 64
MID_SIZE = 128
HIGH_SIZE = 256
FINAL_SIZE = 512

# Note: These values are placeholders, the script will determine
# the correct size for each checkpoint automatically.
IMG_SIZE = INITIAL_SIZE 
BATCH_SIZE = 8
LATENT_DIM = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EVAL_SAMPLES = 2048 # How many samples to use for evaluation

# ---------------- DATASET (Copied from your script) ----------------
def build_transform(size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, subset=None):
        self.files = []
        for r, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.files.append(os.path.join(r, f))
        self.files.sort()
        if subset: self.files = self.files[:subset]
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('L')
        return self.transform(img) if self.transform else img

# ---------------- MODEL BLOCKS (Copied from your script) ----------------
class ResBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        self.pool = nn.AvgPool2d(2)
        self.bn1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.bn2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out, skip = self.pool(out), self.pool(self.skip(x))
        return self.act(out + skip)

class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        self.bn1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.bn2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.act(self.bn1(self.conv1(x_up)))
        out = self.bn2(self.conv2(out))
        skip = self.skip(x_up)
        return self.act(out + skip)

class Encoder(nn.Module):
    # This class is not used in this eval script but included for completeness
    def __init__(self, in_ch=1, base_ch=64, latent_dim=LATENT_DIM):
        super().__init__()
        self.initial = nn.Conv2d(in_ch, base_ch, 7, 1, 3)
        self.blocks = nn.ModuleList([
            ResBlockDown(base_ch, base_ch),
            ResBlockDown(base_ch, base_ch*2),
            ResBlockDown(base_ch*2, base_ch*4),
            ResBlockDown(base_ch*4, base_ch*8),
            ResBlockDown(base_ch*8, base_ch*8)
        ])
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(base_ch*8, latent_dim)
        self.fc_logvar = nn.Linear(base_ch*8, latent_dim)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.initial(x))
        for blk in self.blocks: x = blk(x)
        x = self.avg(x).flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, base_ch=64, latent_dim=LATENT_DIM, img_size=INITIAL_SIZE):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(latent_dim, base_ch*8*(img_size//32)*(img_size//32))
        self.pshape = (base_ch*8, img_size//32, img_size//32)
        self.blocks = nn.ModuleList([
            ResBlockUp(base_ch*8, base_ch*8),
            ResBlockUp(base_ch*8, base_ch*4),
            ResBlockUp(base_ch*4, base_ch*2),
            ResBlockUp(base_ch*2, base_ch),
            ResBlockUp(base_ch, base_ch)
        ])
        self.final_conv = nn.Conv2d(base_ch, out_ch, 7, 1, 3)
        self.tanh = nn.Tanh()
    def forward(self, z):
        x = self.fc(z).view(z.size(0), *self.pshape)
        for blk in self.blocks: x = blk(x)
        return self.tanh(self.final_conv(x))

# ---------------- EMA (Copied from your script) ----------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        self.decay = decay
    def update(self, model):
        pass 
    def copy_to(self, model):
        model.load_state_dict({k:v.clone() for k,v in self.shadow.items()})

# ---------------- Metrics (Copied and FIXED) ----------------
_have_tm=False
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.functional import peak_signal_noise_ratio as tm_psnr
    from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
    _have_tm=True
    print("[INFO] Torchmetrics found. Ready for evaluation.")
except Exception:
    print("[WARN] torchmetrics not available; eval skipped.")

def compute_eval_metrics(G, dataset, device, n_samples=N_EVAL_SAMPLES):
    if not _have_tm: return {"fid":None,"ssim":None,"psnr":None}
    
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    G.eval()
    
    n = min(n_samples, len(dataset))
    if n < n_samples:
        print(f"[WARN] Requested {n_samples} samples, but dataset only has {len(dataset)}. Using {n} samples.")
        
    gens, reals = [], []
    
    with torch.no_grad():
        # 1. Generate fake images
        print(f"Generating {n} fake images for evaluation...")
        while len(gens) < n:
            z = torch.randn(min(BATCH_SIZE, n - len(gens)), LATENT_DIM, device=device)
            img_generated = G(z).cpu()
            img_normalized = (img_generated + 1) / 2 
            gens.append(img_normalized.repeat(1, 3, 1, 1))
        
        gen = torch.cat(gens, dim=0)[:n].to(device)

        # 2. Get real images
        print(f"Sampling {n} real images for evaluation...")
        idxs = random.sample(range(len(dataset)), n)
        reals_list = []
        for i in idxs:
            img = (dataset[i].unsqueeze(0) + 1) / 2 # to [0, 1]
            reals_list.append(img.repeat(1, 3, 1, 1)) # to 3 channels
            
        real = torch.cat(reals_list, dim=0).to(device)

    # 3. Calculate FID
    print("Calculating FID...")
    bs = 16 # Process in batches to avoid OOM
    for i in tqdm(range(0, n, bs), desc="FID update"):
        fid_metric.update(real[i:i+bs], real=True)
        fid_metric.update(gen[i:i+bs], real=False) 
        
    fidv = fid_metric.compute().item()

    # 4. Calculate SSIM & PSNR
    print("Calculating SSIM & PSNR...")
    svals, pvals = [], []
    
    real_gray_01 = real[:, :1] 
    gen_gray_01 = gen[:, :1] 

    for i in tqdm(range(n), desc="SSIM/PSNR"):
        r = real_gray_01[i:i+1]
        g = gen_gray_01[i:i+1]
        svals.append(tm_ssim(g, r, data_range=1.0).item())
        pvals.append(tm_psnr(g, r, data_range=1.0).item())

    return {"fid": fidv, "ssim": sum(svals) / len(svals), "psnr": sum(pvals) / len(pvals)}

# ---------------- NEW EVALUATION MAIN ----------------
if __name__ == "__main__":
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    ckpt_paths = sorted(
        glob.glob(os.path.join(SAVE_DIR, "checkpoint_epoch_*.pth")),
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    )
    
    if not ckpt_paths:
        print(f"No checkpoints found in {SAVE_DIR}. Exiting.")
        exit()

    print(f"Found {len(ckpt_paths)} checkpoints. Starting evaluation...")
    
    upscale_epochs = {
        'to_128': 60,
        'to_256': 130,
        'to_512': 210
    }
    
    # --- ADDED THIS LIST ---
    skip_epochs = [60, 130, 210]

    log_file_path = os.path.join(SAVE_DIR, "evaluation_results.txt")
    with open(log_file_path, "w") as f:
        f.write("epoch,img_size,fid,ssim,psnr\n")
    print(f"Results will be saved to {log_file_path}")

    for path in ckpt_paths:
        epoch_num = 0 
        try:
            epoch_num = int(path.split("_")[-1].split(".")[0])
            
            # --- THIS IS THE NEW CHECK ---
            if epoch_num in skip_epochs:
                print(f"\n--- Skipping Epoch {epoch_num} (fade-in start epoch) ---")
                continue
            # -----------------------------
            
            # 1. Determine correct image size for this checkpoint
            if epoch_num >= upscale_epochs['to_512']:
                current_size = FINAL_SIZE
            elif epoch_num >= upscale_epochs['to_256']:
                current_size = HIGH_SIZE
            elif epoch_num >= upscale_epochs['to_128']:
                current_size = MID_SIZE
            else:
                current_size = INITIAL_SIZE
            
            print(f"\n--- Evaluating Checkpoint: epoch {epoch_num} ({current_size}px) ---")

            # 2. Build model and dataset for this size
            transform = build_transform(current_size)
            dataset = ImageDataset(DATASET_PATH, transform=transform)
            if len(dataset) == 0:
                print(f"Dataset at {DATASET_PATH} is empty or unreadable. Skipping.")
                continue

            # We only need the Generator for evaluation
            G = Decoder(img_size=current_size, latent_dim=LATENT_DIM).to(DEVICE)
            
            # 3. Load checkpoint and EMA weights
            ck = torch.load(path, map_location=DEVICE)
            
            if 'ema' in ck:
                ema = EMA(G) 
                ema.shadow = ck['ema']
                ema.copy_to(G)
                print(f"[INFO] Loaded EMA weights into Generator for epoch {epoch_num}.")
            else:
                print(f"[WARN] No 'ema' key in checkpoint. Loading standard G weights for epoch {epoch_num}.")
                G.load_state_dict(ck['G'])
            
            G.eval()

            # 4. Run evaluation
            metrics = compute_eval_metrics(G, dataset, DEVICE, n_samples=N_EVAL_SAMPLES)
            
            fid = metrics.get('fid')
            ssim = metrics.get('ssim')
            psnr = metrics.get('psnr')
            
            print(f"[METRICS] Epoch {epoch_num} -> FID: {fid:.4f}, SSIM: {ssim:.4f}, PSNR: {psnr:.4f}")
            
            # 5. Save results to log file
            with open(log_file_path, "a") as f:
                f.write(f"{epoch_num},{current_size},{fid},{ssim},{psnr}\n")

            # 6. (Optional) Save some sample images from this checkpoint
            with torch.no_grad():
                z_fixed = torch.randn(16, LATENT_DIM, device=DEVICE)
                imgs = (G(z_fixed).cpu() + 1.0) / 2.0
                imgs = imgs.clamp(0, 1)
                grid = utils.make_grid(imgs, nrow=4, padding=2)
                utils.save_image(grid, os.path.join(SAMPLE_DIR, f"eval_samples_epoch_{epoch_num}.png"))

        except Exception as e:
            print(f"[ERROR] Failed to evaluate epoch {epoch_num}. Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nEvaluation complete. Results saved to {log_file_path}")
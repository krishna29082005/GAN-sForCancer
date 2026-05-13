import os
import glob
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
DATASET_PATH = "/home/tanmoyhazra/gan_project/data/brain_glioma"
SAVE_DIR = "/home/tanmoyhazra/gan_project/checkpoints"
SAMPLE_DIR = "/home/tanmoyhazra/gan_project/samples"

IMG_SIZE = 512
BATCH_SIZE = 8
LATENT_DIM = 256
EPOCHS = 300
LR = 2e-4
BETA1 = 0.5
BETA2 = 0.999
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDA_REC = 1.0
LAMBDA_PER = 0.3
LAMBDA_KL = 0.01
LAMBDA_ADV = 0.5
LAMBDA_FM = 10.0

# -----------------------------
# Dataset
# -----------------------------
class ImageDataset(Dataset):
    def __init__(self, root, transform=None, subset=5000):
        self.files = [os.path.join(root,f) for f in os.listdir(root) if f.endswith(('.png','.jpg'))][:subset]
        self.transform = transform
    def __len__(self): 
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform: 
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -----------------------------
# Model Blocks
# -----------------------------
class ResBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        self.pool = nn.AvgPool2d(2)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.pool(out)
        skip = self.pool(self.skip(x))
        return self.act(out + skip)

class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.act(self.bn1(self.conv1(x_up)))
        out = self.bn2(self.conv2(out))
        skip = self.skip(x_up)
        return self.act(out + skip)

# -----------------------------
# Encoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, latent_dim=LATENT_DIM):
        super().__init__()
        self.initial = nn.Conv2d(in_ch, base_ch, 7, 1, 3)
        self.rb1 = ResBlockDown(base_ch, base_ch)
        self.rb2 = ResBlockDown(base_ch, base_ch*2)
        self.rb3 = ResBlockDown(base_ch*2, base_ch*4)
        self.rb4 = ResBlockDown(base_ch*4, base_ch*8)
        self.rb5 = ResBlockDown(base_ch*8, base_ch*8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(base_ch*8, latent_dim)
        self.fc_logvar = nn.Linear(base_ch*8, latent_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.initial(x))
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.avg(x).flatten(1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_ch=3, base_ch=64, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_ch*8*(IMG_SIZE//32)*(IMG_SIZE//32))
        self.pshape = (base_ch*8, IMG_SIZE//32, IMG_SIZE//32)
        self.rb1 = ResBlockUp(base_ch*8, base_ch*8)
        self.rb2 = ResBlockUp(base_ch*8, base_ch*4)
        self.rb3 = ResBlockUp(base_ch*4, base_ch*2)
        self.rb4 = ResBlockUp(base_ch*2, base_ch)
        self.rb5 = ResBlockUp(base_ch, base_ch)
        self.final_conv = nn.Conv2d(base_ch, out_ch, 7, 1, 3)
        self.tanh = nn.Tanh()
    def forward(self, z):
        x = self.fc(z).view(z.size(0), *self.pshape)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.final_conv(x)
        return self.tanh(x)

from torch.nn.utils import spectral_norm
class Discriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.init_conv = spectral_norm(nn.Conv2d(in_ch, base_ch, 4, 2, 1))
        self.rb1 = nn.Sequential(spectral_norm(nn.Conv2d(base_ch, base_ch*2, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True))
        self.rb2 = nn.Sequential(spectral_norm(nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True))
        self.rb3 = nn.Sequential(spectral_norm(nn.Conv2d(base_ch*4, base_ch*8, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True))
        self.rb4 = nn.Sequential(spectral_norm(nn.Conv2d(base_ch*8, base_ch*8, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True))
        self.last = spectral_norm(nn.Conv2d(base_ch*8, 1, 1, 1, 0))
    def forward(self, x):
        fmaps = []
        x = self.init_conv(x); fmaps.append(x)
        x = self.rb1(x); fmaps.append(x)
        x = self.rb2(x); fmaps.append(x)
        x = self.rb3(x); fmaps.append(x)
        x = self.rb4(x); fmaps.append(x)
        out = self.last(x)
        return out, fmaps

# -----------------------------
# Utilities
# -----------------------------
def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

def hinge_d_loss(real_out, fake_out):
    return torch.mean(F.relu(1.0 - real_out)) + torch.mean(F.relu(1.0 + fake_out))

def hinge_g_loss(fake_out):
    return -torch.mean(fake_out)

# -----------------------------
# Training
# -----------------------------
if __name__ == "__main__":
    train_dataset = ImageDataset(DATASET_PATH, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    E = Encoder().to(DEVICE)
    G = Decoder().to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_EG = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), lr=LR, betas=(BETA1,BETA2))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1,BETA2))

    z_fixed = torch.randn(16, LATENT_DIM, device=DEVICE)

    # Load latest checkpoint if any
    checkpoint_files = glob.glob(os.path.join(SAVE_DIR, "checkpoint_epoch_*.pth"))
    if checkpoint_files:
        epochs = [int(re.search(r'checkpoint_epoch_(\d+).pth', f).group(1)) for f in checkpoint_files]
        latest_epoch = max(epochs)
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{latest_epoch}.pth")
        checkpoint = torch.load(checkpoint_path, weights_only=True)

        E.load_state_dict(checkpoint['E'])
        G.load_state_dict(checkpoint['G'])
        D.load_state_dict(checkpoint['D'])
        opt_EG.load_state_dict(checkpoint['opt_EG'])
        opt_D.load_state_dict(checkpoint['opt_D'])

        start_epoch = latest_epoch
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        start_epoch = 0
        print("Starting training from scratch.")

    print("CUDA available:", torch.cuda.is_available())
    print("Device:", DEVICE)
    if DEVICE.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x in pbar:
            x = x.to(DEVICE)

            # Discriminator update
            with torch.no_grad():
                mu, logvar = E(x)
                z = reparameterize(mu, logvar)
                x_rec = G(z)
            real_out, _ = D(x)
            fake_out, _ = D(x_rec.detach())
            d_loss = hinge_d_loss(real_out, fake_out)

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # Encoder + Generator update
            mu, logvar = E(x)
            z = reparameterize(mu, logvar)
            x_rec = G(z)

            rec_l1 = F.l1_loss(x_rec, x)
            kl = -0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            fake_out_for_g, fmaps_fake = D(x_rec)
            adv_loss = hinge_g_loss(fake_out_for_g)
            real_out_for_fm, fmaps_real = D(x)
            fm_loss = sum(F.l1_loss(ff, fr.detach()) for ff, fr in zip(fmaps_fake, fmaps_real))
            total_g_loss = (LAMBDA_REC*rec_l1) + (LAMBDA_KL*kl) + (LAMBDA_ADV*adv_loss) + (LAMBDA_FM*fm_loss)

            opt_EG.zero_grad()
            total_g_loss.backward()
            opt_EG.step()

            pbar.set_postfix({'d_loss': d_loss.item(), 'g_loss': total_g_loss.item(), 'rec_l1': rec_l1.item(), 'kl': kl.item()})

        # Save samples and checkpoints every 10 epochs
        if epoch % 5 == 0:
            with torch.no_grad():
                imgs = G(z_fixed).cpu()
                grid = utils.make_grid((imgs + 1) / 2, nrow=4, padding=2)
                utils.save_image(grid, os.path.join(SAMPLE_DIR, f"samples_epoch_{epoch}.png"))

            torch.save({
                'E': E.state_dict(),
                'G': G.state_dict(),
                'D': D.state_dict(),
                'opt_EG': opt_EG.state_dict(),
                'opt_D': opt_D.state_dict()
            }, os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch}.pth"))

    print("Training completed!")

#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from torchmetrics.image.fid import FrechetInceptionDistance
import lpips
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
import pandas as pd 

# ============================= CONFIG =============================
CHANNELS = 1
LATENT_DIM = 256
GEN_IMG_SIZE = 256  
SAVE_IMG_SIZE = 512 
BATCH_SIZE = 32     
NUM_GEN_SAMPLES = 2048

DEFAULT_DATA_DIR = "/home/tanmoyhazra/gan_project/data/brain_glioma"
DEFAULT_CHECKPOINT_DIR = "/home/tanmoyhazra/gan_project/wgan_resnet_pytorch_512_output/checkpoints"
# New filename for this specific range
DEFAULT_OUTPUT_FILE = "wgan_epochs_355_375_metrics.txt" 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =========================== DATA UTILITY ==========================
class ImageDataset(Dataset):
    def __init__(self, root, img_size):
        self.files = []
        for r, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.files.append(os.path.join(r, f))
        self.files.sort()
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('L')
        return self.transform(img)

def get_dataloader(data_dir, img_size, batch_size=BATCH_SIZE, num_workers=2):
    ds = ImageDataset(data_dir, img_size)
    if len(ds) == 0:
        raise FileNotFoundError(f"No images found in {data_dir}")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl, len(ds)

# ============================= MODELS ==============================
def gn(c): return nn.GroupNorm(8, c)

class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.norm1 = gn(out_ch); self.norm2 = gn(out_ch)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        y = self.act(self.norm1(self.conv1(x_up)))
        y = self.norm2(self.conv2(y))
        return self.act(y + self.skip(x_up))

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_dim = 8
        self.base_ch   = 256
        self.fc = nn.Linear(LATENT_DIM, self.base_ch * self.start_dim * self.start_dim)
        self.blocks = nn.Sequential(
            ResBlockUp(self.base_ch, self.base_ch),
            ResBlockUp(self.base_ch, self.base_ch // 2),
            ResBlockUp(self.base_ch // 2, self.base_ch // 4),
            ResBlockUp(self.base_ch // 4, self.base_ch // 8),
            ResBlockUp(self.base_ch // 8, self.base_ch // 16),
        )
        self.final = nn.Sequential(
            nn.Conv2d(self.base_ch // 16, CHANNELS, 3, 1, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        x = self.fc(z).view(z.size(0), self.base_ch, self.start_dim, self.start_dim)
        x = self.blocks(x)
        return self.final(x)  

# =========================== METRIC UTILS ==========================
def to_uint8_for_fid(tensor01):
    """Converts tensor from [0, 1] float to [0, 255] uint8."""
    t = (tensor01 * 255.0).round().clamp(0,255).to(torch.uint8)
    return t

def tensor_to_numpy_gray01(t):
    """Converts [-1, 1] tensor batch to [0, 1] numpy for SSIM/PSNR."""
    if t.ndim == 4:
        t = t[0]
    if t.min() < -0.5:
        t01 = (t.clamp(-1,1) + 1.0) / 2.0
    else:
        t01 = t.clamp(0,1)
    return t01.squeeze(0).cpu().numpy()

def compute_ssim_psnr_batch(real_batch, fake_batch):
    real_batch = real_batch.detach().cpu()
    fake_batch = fake_batch.detach().cpu()
    bs = real_batch.shape[0]
    svals = []
    pvals = []
    for i in range(bs):
        rn = tensor_to_numpy_gray01(real_batch[i])
        fn = tensor_to_numpy_gray01(fake_batch[i])
        try:
            s = ssim_fn(rn, fn, data_range=1.0)
        except Exception:
            s = float('nan')
        try:
            p = psnr_fn(rn, fn, data_range=1.0)
        except Exception:
            p = float('nan')
        svals.append(s)
        pvals.append(p)
    svals = np.array(svals, dtype=np.float32)
    pvals = np.array(pvals, dtype=np.float32)
    s_mean = float(np.nanmean(svals))
    p_mean = float(np.nanmean(pvals))
    return s_mean, p_mean

@torch.no_grad()
def compute_lpips_batch(lpips_model, real_batch, fake_batch, device):
    """LPIPS expects 3 channels, range [-1, 1]."""
    r3 = real_batch.repeat(1,3,1,1).to(device)    
    f3 = fake_batch.repeat(1,3,1,1).to(device)
    r3 = r3.float()
    f3 = f3.float()
    out = lpips_model(f3, r3)  
    out = out.view(out.shape[0], -1).mean(axis=1)
    return float(out.mean().item())

# ========================== EVALUATION LOOP ========================
@torch.no_grad()
def evaluate_checkpoint(checkpoint_path, generator, real_loader_cpu, lpips_model, device, output_file,
                        n_samples=NUM_GEN_SAMPLES, batch_size=BATCH_SIZE):
    print(f"\n=== Evaluating {checkpoint_path} ===")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False) 
    except Exception as e:
        print(f"[ERROR] Could not load checkpoint {checkpoint_path}: {e}")
        return None

    # Load generator weights (with fallbacks)
    gen_key_candidates = ['generator', 'G', 'netG', 'g_state_dict']
    loaded = False
    for k in gen_key_candidates:
        if k in ckpt:
            try:
                generator.load_state_dict(ckpt[k])
                loaded = True
                break
            except Exception as e:
                print(f"[WARN] Found key {k} but load_state_dict failed: {e}")
    if not loaded:
        try:
            generator.load_state_dict(ckpt)
            loaded = True
        except Exception:
            print("[ERROR] No usable generator weights found in checkpoint; skipping.")
            return None

    # Determine epoch number
    epoch = -1
    if isinstance(ckpt, dict):
        for k in ['epoch', 'iter', 'step']:
            if k in ckpt:
                try:
                    epoch = int(ckpt[k])
                    break
                except Exception:
                    pass
    if epoch == -1:
        try:
            epoch = int(Path(checkpoint_path).stem.split('_')[-1])
        except Exception:
            epoch = -1

    generator.eval()
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    print("[INFO] Updating FID with real dataset (batches)...")
    for real_batch in tqdm(real_loader_cpu, desc="FID: adding real"):
        real_01 = (real_batch.clamp(-1,1) + 1.0) / 2.0
        real_uint8 = to_uint8_for_fid(real_01).repeat(1,3,1,1).to(device)
        fid_metric.update(real_uint8, real=True)

    print("[INFO] Generating fakes and updating metrics in batches...")
    generated_count = 0
    total_ssim = 0.0
    total_psnr = 0.0
    total_lpips = 0.0

    pbar = tqdm(total=n_samples, desc=f"Generating (epoch {epoch})")
    real_iter = iter(real_loader_cpu)
    
    while generated_count < n_samples:
        cur_bs = min(batch_size, n_samples - generated_count)
        z = torch.randn(cur_bs, LATENT_DIM, device=device)
        
        # Generator call (no autocast context)
        fake_256 = generator(z) 
            
        # Upscale and move fake to CPU for SSIM/PSNR
        fake_512 = F.interpolate(fake_256, size=(SAVE_IMG_SIZE, SAVE_IMG_SIZE), mode='bilinear', align_corners=False)

        # Get a real batch to pair with (cycle through dataset)
        try:
            real_batch = next(real_iter)
        except StopIteration:
            real_iter = iter(real_loader_cpu)
            real_batch = next(real_iter)

        # Ensure real_batch size matches cur_bs (handling cycling/wrap-around)
        if real_batch.shape[0] < cur_bs:
            parts = [real_batch]
            need = cur_bs - real_batch.shape[0]
            while need > 0:
                try:
                    r = next(real_iter)
                except StopIteration:
                    real_iter = iter(real_loader_cpu)
                    r = next(real_iter)
                take = min(need, r.shape[0])
                parts.append(r[:take])
                need -= take
            real_batch = torch.cat(parts, dim=0)
        real_batch = real_batch[:cur_bs]

        # ---- Update FID with fake images (convert to uint8 & 3-ch) ----
        fake_512_01 = (fake_512.clamp(-1,1) + 1.0) / 2.0
        fake_uint8 = to_uint8_for_fid(fake_512_01.cpu()).repeat(1,3,1,1).to(device)
        fid_metric.update(fake_uint8, real=False)

        # ---- SSIM & PSNR (CPU numpy) ----
        real_512 = real_batch
        s_mean, p_mean = compute_ssim_psnr_batch(real_512, fake_512.detach().cpu())
        total_ssim += s_mean * cur_bs
        total_psnr += p_mean * cur_bs

        # ---- LPIPS (GPU) ----
        fake_lpips = fake_512.detach().float().to(device)
        real_lpips = real_512.detach().float().to(device)
        
        lpips_val = compute_lpips_batch(lpips_model, real_lpips, fake_lpips, device)
        total_lpips += lpips_val * cur_bs

        generated_count += cur_bs
        pbar.update(cur_bs)

    pbar.close()

    # --- Finalize Metrics ---
    final_fid = float(fid_metric.compute().item())
    avg_ssim = total_ssim / float(n_samples)
    avg_psnr = total_psnr / float(n_samples)
    avg_lpips = total_lpips / float(n_samples)

    # --- Write to TXT Output File ---
    header_needed = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
    with open(output_file, "a") as f:
        if header_needed:
            f.write(f"{'Epoch':>6}  {'Samples':>8}  {'FID':>10}  {'LPIPS':>10}  {'SSIM':>8}  {'PSNR':>8}\n")
            f.write("-" * 60 + "\n")
        f.write(f"{epoch:6d}  {n_samples:8d}  {final_fid:10.4f}  {avg_lpips:10.4f}  {avg_ssim:8.4f}  {avg_psnr:8.4f}\n")

    print(f"[DONE] epoch={epoch} FID={final_fid:.4f} LPIPS={avg_lpips:.4f} SSIM={avg_ssim:.4f} PSNR={avg_psnr:.4f}")
    return {
        "epoch": epoch,
        "FID": final_fid,
        "LPIPS": avg_lpips,
        "SSIM": avg_ssim,
        "PSNR": avg_psnr,
    }

# ============================ MAIN EXECUTION =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--n_samples", type=int, default=NUM_GEN_SAMPLES)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    data_dir = args.data_dir
    out_file = args.out
    n_samples = args.n_samples
    batch_size = args.batch_size

    print(f"Checkpoints: {ckpt_dir}")
    print(f"Data: {data_dir}")
    print(f"Output: {out_file}")
    print(f"Samples per checkpoint: {n_samples}, batch_size: {batch_size}")

    try:
        real_loader, real_len = get_dataloader(data_dir, SAVE_IMG_SIZE, batch_size=batch_size, num_workers=2)
        print(f"Found {real_len} real images.")
    except Exception as e:
        print(f"[ERROR] Could not create dataloader: {e}")
        sys.exit(1)

    G = Generator().to(DEVICE).eval()
    lpips_model = lpips.LPIPS(net='vgg').to(DEVICE).eval()

    # --- RESTORED LOGIC: FIND AND EVALUATE ALL CHECKPOINTS ---
    ckpt_paths = sorted(ckpt_dir.glob("wgan_epoch_*.pth"),
                        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0)
    
    if len(ckpt_paths) == 0:
        print(f"[ERROR] No checkpoints found in {ckpt_dir}")
        sys.exit(1)

    # --- NEW FILTERING LOGIC: EVALUATE ONLY EPOCHS 355 to 375 ---
    filtered_paths = []
    MIN_EPOCH = 355
    
    for p in ckpt_paths:
        try:
            epoch = int(p.stem.split("_")[-1])
            # Filter for epochs >= 355
            if epoch >= MIN_EPOCH:
                filtered_paths.append(p)
        except ValueError:
            # Skip files that don't end in a number
            continue

    if not filtered_paths:
        print(f"[ERROR] No checkpoints found in the required range (Epoch {MIN_EPOCH}+).")
        sys.exit(1)

    print(f"Found {len(filtered_paths)} checkpoints in range (Epoch {MIN_EPOCH} to {filtered_paths[-1].name}). Evaluating...")

    # Clear previous output file
    if os.path.exists(out_file):
        print(f"[INFO] Removing existing output file: {out_file}")
        os.remove(out_file)

    results = []
    for ck in filtered_paths:
        try:
            res = evaluate_checkpoint(
                str(ck),
                generator=G,
                real_loader_cpu=real_loader,
                lpips_model=lpips_model,
                device=DEVICE,
                output_file=out_file,
                n_samples=n_samples,
                batch_size=batch_size
            )
            if res is not None:
                results.append(res)
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {ck}: {e}")
            # If a checkpoint fails, the script will skip it and continue to the next one.

    # Final summary CSV (requires pandas)
    if results:
        try:
            df = pd.DataFrame(results)
            csv_path = Path(out_file).with_suffix(".csv")
            df.sort_values("epoch", inplace=True)
            df.to_csv(csv_path, index=False)
            print(f"[INFO] Summary CSV saved to {csv_path}")
        except ImportError:
            print("[WARN] Pandas not found. Skipping final CSV summary creation.")


if __name__ == "__main__":
    main()
import os, copy, math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm
from torch.nn.utils import spectral_norm

# ---------------- CONFIG ----------------
DATASET_PATH = r"/home/tanmoyhazra/gan_project/data/brain_glioma"
SAVE_DIR = r"/home/tanmoyhazra/gan_project/checkpoints1"
SAMPLE_DIR = r"/home/tanmoyhazra/gan_project/samples1"

INITIAL_SIZE = 64
MID_SIZE = 128
HIGH_SIZE = 256
FINAL_SIZE = 512

IMG_SIZE = INITIAL_SIZE
BATCH_SIZE = 8
LATENT_DIM = 256
EPOCHS = 300

LR_G = 2e-4
LR_D = 4e-4
BETA1, BETA2 = 0.5, 0.999
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDA_REC, LAMBDA_KL, LAMBDA_ADV, LAMBDA_FM = 1.0, 0.05, 1.0, 0.5

N_PRETRAIN = 25
KL_WARMUP_EPOCHS = 60
ADV_RAMP_EPOCHS = 40
FADE_EPOCHS = 10

EVAL_EVERY = 10
N_EVAL_SAMPLES = 100

# ---------------- DATASET ----------------
def build_transform(size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

transform = build_transform(IMG_SIZE)

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

# ---------------- MODEL BLOCKS ----------------
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

class Discriminator(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.init_conv = spectral_norm(nn.Conv2d(in_ch, base_ch, 4, 2, 1))
        self.blocks = nn.ModuleList([
            nn.Sequential(spectral_norm(nn.Conv2d(base_ch, base_ch*2, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(spectral_norm(nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(spectral_norm(nn.Conv2d(base_ch*4, base_ch*8, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(spectral_norm(nn.Conv2d(base_ch*8, base_ch*8, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True))
        ])
        self.last = spectral_norm(nn.Conv2d(base_ch*8, 1, 1))
    def forward(self, x):
        fmaps = []
        x = self.init_conv(x)
        for blk in self.blocks:
            x = blk(x)
            fmaps.append(x)
        return self.last(x), fmaps

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar); eps = torch.randn_like(std)
    return mu + eps*std
def hinge_d_loss(r,f): return torch.mean(F.relu(1-r))+torch.mean(F.relu(1+f))
def hinge_g_loss(f): return -torch.mean(f)

# ---------------- EMA ----------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        self.decay = decay
    def update(self, model):
        for k,v in model.state_dict().items():
            dev = v.device
            old = self.shadow[k].to(dev)
            new = self.decay * old + (1-self.decay) * v.detach()
            self.shadow[k] = new.cpu()
    def copy_to(self, model):
        model.load_state_dict({k:v.clone() for k,v in self.shadow.items()})

# ---------------- Metrics ----------------
_have_tm=False
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.functional import peak_signal_noise_ratio as tm_psnr
    from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
    _have_tm=True
except Exception:
    print("[WARN] torchmetrics not available; eval skipped.")

def compute_eval_metrics(G, dataset, device, n_samples=N_EVAL_SAMPLES):
    if not _have_tm: return {"fid":None,"ssim":None,"psnr":None}
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    G.eval(); n=min(n_samples,len(dataset))
    gens,reals=[],[]
    with torch.no_grad():
        while len(gens)<n:
            z=torch.randn(min(16,n-len(gens)),LATENT_DIM,device=device)
            img=(G(z).cpu()+1)/2
            gens.append(img.repeat(1,3,1,1))
        gen=torch.cat(gens,dim=0)[:n].to(device)
        idxs=random.sample(range(len(dataset)),n)
        reals=[dataset[i].unsqueeze(0).repeat(1,3,1,1) for i in idxs]
        real=torch.cat(reals,dim=0).to(device)
    for i in range(0,n,16):
        fid.update(real[i:i+16],real=True)
        fid.update(gen[i:i:16],real=False)
    fidv=fid.compute().item()
    svals,pvals=[],[]
    for i in range(n):
        r=real[i:i+1,:1]; g=gen[i:i+1,:1]
        svals.append(tm_ssim(g,r,data_range=1.).item())
        pvals.append(tm_psnr(g,r,data_range=1.).item())
    return {"fid":fidv,"ssim":sum(svals)/len(svals),"psnr":sum(pvals)/len(pvals)}

# ---------------- Progressive Update ----------------
def update_decoder_resolution(G,new_size):
    base=64
    G.img_size=new_size
    G.fc=nn.Linear(LATENT_DIM,base*8*(new_size//32)*(new_size//32)).to(DEVICE)
    G.pshape=(base*8,new_size//32,new_size//32)
    G.blocks=nn.ModuleList([
        ResBlockUp(base*8,base*8).to(DEVICE),
        ResBlockUp(base*8,base*4).to(DEVICE),
        ResBlockUp(base*4,base*2).to(DEVICE),
        ResBlockUp(base*2,base).to(DEVICE),
        ResBlockUp(base,base).to(DEVICE)
    ])
    G.final_conv=nn.Conv2d(base,1,7,1,3).to(DEVICE)

# ---------------- TRAIN ----------------
if __name__=="__main__":
    os.makedirs(SAVE_DIR,exist_ok=True); os.makedirs(SAMPLE_DIR,exist_ok=True)
    dataset=ImageDataset(DATASET_PATH,transform=transform)
    loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,pin_memory=True)

    E,G,D=Encoder().to(DEVICE),Decoder(img_size=IMG_SIZE).to(DEVICE),Discriminator().to(DEVICE)
    opt_EG=torch.optim.Adam(list(E.parameters())+list(G.parameters()),lr=LR_G,betas=(BETA1,BETA2))
    opt_D=torch.optim.Adam(D.parameters(),lr=LR_D,betas=(BETA1,BETA2))
    ema=EMA(G)

    # ----- Resume -----
    start_epoch=1
    ckpts=[f for f in os.listdir(SAVE_DIR) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    if ckpts:
        latest=max(ckpts,key=lambda f:int(f.split("_")[-1].split(".")[0]))
        path=os.path.join(SAVE_DIR,latest)
        ck=torch.load(path,map_location=DEVICE)
        E.load_state_dict(ck['E']); G.load_state_dict(ck['G']); D.load_state_dict(ck['D'])
        opt_EG.load_state_dict(ck['opt_EG']); opt_D.load_state_dict(ck['opt_D'])
        start_epoch=int(latest.split("_")[-1].split(".")[0])+1
        if 'ema' in ck:
            ema.shadow=ck['ema']; print(f"[INFO] Resumed from {path} (epoch {start_epoch}), EMA restored.")
        else:
            print(f"[INFO] Resumed from {path} (epoch {start_epoch}), [WARN] no EMA found ? new EMA started.")
    else:
        print("[INFO] No checkpoint found. Starting fresh.")

    z_fixed=torch.randn(16,LATENT_DIM,device=DEVICE)
    current_size=IMG_SIZE
    next_up_epoch={'to_128':60,'to_256':130,'to_512':210}
    fade_state={'active':False,'start':None,'from':None,'to':None,'G_prev':None}

    # --- Adjust for resume ---
    # Need to rebuild models, data, and fade state if resuming *after* an upscale
    if start_epoch > 1:
        resumed_size = INITIAL_SIZE
        if start_epoch > next_up_epoch['to_128']:
            resumed_size = MID_SIZE
        if start_epoch > next_up_epoch['to_256']:
            resumed_size = HIGH_SIZE
        if start_epoch > next_up_epoch['to_512']:
            resumed_size = FINAL_SIZE
        
        if resumed_size != current_size:
            print(f"[INFO] Resuming at {resumed_size}px, updating models and data...")
            # Update G (already loaded from ckpt, but need to re-init submodules if not saved properly)
            update_decoder_resolution(G, resumed_size) 
            G.load_state_dict(ck['G']) # Re-load state dict *after* rebuilding architecture
            G = G.to(DEVICE)
            # Re-init optimizer just in case
            opt_EG = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), lr=LR_G, betas=(BETA1, BETA2))
            opt_EG.load_state_dict(ck['opt_EG'])
            # Update EMA
            if 'ema' in ck:
                ema.shadow = ck['ema']
            else:
                ema = EMA(G) # Re-init EMA if not in checkpoint
            # Update dataset
            transform = build_transform(resumed_size)
            dataset = ImageDataset(DATASET_PATH, transform=transform)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
            current_size = resumed_size
        
        # Check if resuming *during* a fade-in
        for key, size in [('to_128', MID_SIZE), ('to_256', HIGH_SIZE), ('to_512', FINAL_SIZE)]:
            if (start_epoch > next_up_epoch[key]) and (start_epoch <= next_up_epoch[key] + FADE_EPOCHS):
                print(f"[WARN] Resuming during a fade-in ({size}px) is not supported. Starting fade from scratch.")
                # This is complex to restore. Simplest is to just restart the fade.
                # Or, you'd need to save G_prev in the checkpoint.
                # For this code, we'll just let it proceed without fade_state.
                pass 


    for epoch in range(start_epoch,EPOCHS+1):
        # --- progressive updates ---
        for key,size in [('to_128',MID_SIZE),('to_256',HIGH_SIZE),('to_512',FINAL_SIZE)]:
            if epoch==next_up_epoch[key] and current_size<size:
                print(f"\n[INFO] Upscaling {current_size}->{size} at epoch {epoch}")
                G_prev=copy.deepcopy(G).to(DEVICE)
                update_decoder_resolution(G,size)
                G=G.to(DEVICE); opt_EG=torch.optim.Adam(list(E.parameters())+list(G.parameters()),lr=LR_G,betas=(BETA1,BETA2))
                ema=EMA(G) # Re-init EMA
                transform=build_transform(size)
                dataset=ImageDataset(DATASET_PATH,transform=transform)
                loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,pin_memory=True)
                current_size=size
                fade_state={'active':True,'start':epoch,'from':G_prev.img_size if hasattr(G_prev,'img_size') else current_size,'to':size,'G_prev':G_prev}
                print(f"[INFO] Fade-in for {FADE_EPOCHS} epochs.")

        kl_w=LAMBDA_KL*min(1.0,epoch/KL_WARMUP_EPOCHS)
        adv_w=0.0 if epoch<=N_PRETRAIN else LAMBDA_ADV*min(1.0,(epoch-N_PRETRAIN)/ADV_RAMP_EPOCHS)
        pbar=tqdm(loader,desc=f"Epoch {epoch} ({current_size}px)")

        for x in pbar:
            x=x.to(DEVICE)
            # ---- Train D ----
            with torch.no_grad():
                mu,lv=E(x); z=reparameterize(mu,lv); xr_new=G(z)
                if fade_state['active']:
                    alpha=min(1.0,(epoch-fade_state['start']+1)/FADE_EPOCHS)
                    Gp=fade_state['G_prev']; xr_prev=Gp(z)
                    xr_prev_up=F.interpolate(xr_prev,size=xr_new.shape[-2:],mode='bilinear',align_corners=False)
                    xr=(1-alpha)*xr_prev_up+alpha*xr_new
                else: xr=xr_new
            r_out,_=D(x); f_out,_=D(xr.detach())
            d_loss=hinge_d_loss(r_out,f_out)
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # ---- Train E+G ----
            mu,lv=E(x); z=reparameterize(mu,lv); xr_new=G(z)
            if fade_state['active']:
                alpha=min(1.0,(epoch-fade_state['start']+1)/FADE_EPOCHS)
                Gp=fade_state['G_prev']; xr_prev=Gp(z)
                xr_prev_up=F.interpolate(xr_prev,size=xr_new.shape[-2:],mode='bilinear',align_corners=False)
                xr=(1-alpha)*xr_prev_up+alpha*xr_new
            else: xr=xr_new
            rec=F.l1_loss(xr,x)
            kl=-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
            f_out_g,f_f=D(xr); adv=hinge_g_loss(f_out_g)
            r_out_f,f_r=D(x)
            fm=sum(F.l1_loss(ff,fr.detach()) for ff,fr in zip(f_f,f_r))
            g_loss=LAMBDA_REC*rec+kl_w*kl+adv_w*adv+LAMBDA_FM*fm
            opt_EG.zero_grad(); g_loss.backward(); opt_EG.step(); ema.update(G)
            pbar.set_postfix({'d':f"{d_loss.item():.3f}",'g':f"{g_loss.item():.3f}",'rec':f"{rec.item():.3f}"})

        if fade_state['active'] and (epoch-fade_state['start']+1)>=FADE_EPOCHS:
            fade_state={'active':False,'start':None,'from':None,'to':None,'G_prev':None}
            print(f"[INFO] Fade-in complete for {current_size}px.")

        if epoch % EVAL_EVERY == 0:
            G_eval = copy.deepcopy(G).to(DEVICE)
            try:
                ema.copy_to(G_eval)
            except Exception:
                pass

            # ---- Save generated samples ----
            with torch.no_grad():
                G_eval.eval() # Set to eval mode for sampling
                imgs = (G_eval(z_fixed).cpu() + 1.0) / 2.0
                imgs = imgs.clamp(0, 1)
                grid = utils.make_grid(imgs, nrow=4, padding=2)
                utils.save_image(grid, os.path.join(SAMPLE_DIR, f"samples_epoch_{epoch}.png"))

            # ---- Save checkpoint ----
            ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'E': E.state_dict(),
                'G': G.state_dict(),
                'D': D.state_dict(),
                'opt_EG': opt_EG.state_dict(),
                'opt_D': opt_D.state_dict(),
                'ema': ema.shadow    # ?? save EMA weights
            }, ckpt_path)
            print(f"[INFO] Checkpoint saved: {ckpt_path}")

            # ---- Evaluate metrics ----
            try:
                metrics = compute_eval_metrics(G_eval, dataset, DEVICE,
                                                 n_samples=min(N_EVAL_SAMPLES, len(dataset)))
                print(f"[METRICS] Epoch {epoch} -> "
                      f"FID: {metrics.get('fid')}, "
                      f"SSIM: {metrics.get('ssim')}, "
                      f"PSNR: {metrics.get('psnr')}")
                with open(os.path.join(SAVE_DIR, "eval_log.txt"), "a") as f:
                    f.write(f"{epoch},{metrics.get('fid')},{metrics.get('ssim')},{metrics.get('psnr')}\n")
            except Exception as e:
                print("[WARN] Evaluation failed:", e)

    print(" Training completed successfully.")
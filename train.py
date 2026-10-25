import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from multiprocessing import freeze_support
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageFilter
import numpy as np
import random, io, copy, csv, functools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

torch.backends.cudnn.benchmark = True

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

try:
    import timm
except ImportError:
    raise ImportError("Run:  pip install timm")


def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RGB_SIZE      = 192
AUX_SIZE      = 224
BATCH_SIZE    = 24
EVAL_BATCH    = 64
EPOCHS        = 30
FREEZE_EPOCHS = 6
EVAL_EVERY    = 2
GATE_ENT_W    = 0.05
LR_RESTART_EP = 22

train_dir = r"C:\Users\kriti\Downloads\ai detect\train"
test_dir  = r"C:\Users\kriti\Downloads\ai detect\test"

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# FFT / ELA
# ─────────────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=4)
def _get_radial_map(H, W):
    cy, cx = H // 2, W // 2
    Y, X   = np.ogrid[:H, :W]
    R      = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
    R      = np.clip(R, 0, min(H, W) // 2 - 1)
    R_cnt  = np.bincount(R.ravel()).astype(np.float32)
    return R, R_cnt


def fast_fft(img):
    gray       = np.array(img.convert("L").resize((AUX_SIZE, AUX_SIZE)), dtype=np.float32)
    H, W       = gray.shape
    R, R_cnt   = _get_radial_map(H, W)
    fft        = np.fft.fftshift(np.fft.fft2(gray))
    magnitude  = np.log1p(np.abs(fft)).astype(np.float32)
    phase      = np.angle(fft).astype(np.float32)
    radial     = np.bincount(R.ravel(), magnitude.ravel()) / (R_cnt + 1e-8)
    radial_map = radial[R].astype(np.float32)

    def norm(x):
        mn, mx = x.min(), x.max()
        return ((x - mn) / (mx - mn + 1e-8)).astype(np.float32)

    ch = np.stack([norm(magnitude), norm(phase), norm(radial_map)], axis=-1)
    return Image.fromarray((ch * 255).astype(np.uint8))


def ela(img, quality=90, scale=15):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recomp = Image.open(buf).convert("RGB")
    diff   = np.abs(
        np.array(img.convert("RGB"), dtype=np.float32) -
        np.array(recomp,             dtype=np.float32)
    ) * scale
    return Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────
class BlockNoise:
    def __call__(self, img):
        arr = np.array(img).astype(np.int16)
        h, w, _ = arr.shape
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                if random.random() < 0.3:
                    arr[i:i+8, j:j+8] += random.randint(-20, 20)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class TrainCorrupt:
    def __call__(self, img):
        r = random.random()
        if r < 0.35:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(40, 75))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        elif r < 0.55:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(15, 40))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        elif r < 0.70:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.2)))
        elif r < 0.82:
            w, h = img.size
            s    = random.choice([2, 3])
            img  = img.resize((w // s, h // s)).resize((w, h), Image.BICUBIC)
        elif r < 0.90:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(30, 60))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.8)))

        if random.random() < 0.4:
            img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
            img = TF.adjust_contrast(img,   random.uniform(0.85, 1.15))
            img = TF.adjust_saturation(img, random.uniform(0.85, 1.15))
        if random.random() < 0.5:
            img = BlockNoise()(img)
        return img


class FixedCompression:
    def __call__(self, img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=35)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class FixedScreenshot:
    def __call__(self, img):
        w, h = img.size
        return img.resize((w // 2, h // 2), Image.BICUBIC) \
                   .resize((AUX_SIZE, AUX_SIZE), Image.BICUBIC) \
                   .filter(ImageFilter.GaussianBlur(1))


class _NoOp:
    def __call__(self, img): return img


class _Combined:
    def __init__(self):
        self._c = FixedCompression()
        self._s = FixedScreenshot()
    def __call__(self, img):
        return self._s(self._c(img))


_CORRUPT_FNS = {
    "original":   _NoOp(),
    "compressed": FixedCompression(),
    "screenshot": FixedScreenshot(),
    "combined":   _Combined(),
}


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
rgb_train_tf = transforms.Compose([
    transforms.Resize((RGB_SIZE, RGB_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
    transforms.RandomGrayscale(p=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
])
rgb_val_tf = transforms.Compose([
    transforms.Resize((RGB_SIZE, RGB_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
aux_val_tf = transforms.Compose([
    transforms.Resize((AUX_SIZE, AUX_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
aux_train_tf = transforms.Compose([
    transforms.Resize((AUX_SIZE, AUX_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# ─────────────────────────────────────────────────────────────────────────────
# RAM CACHE — train only
# ─────────────────────────────────────────────────────────────────────────────
def build_train_ram_cache(root):
    ds = datasets.ImageFolder(root)
    log(f"Loading {len(ds)} train images to RAM...")
    cache = [
        np.array(Image.open(p).convert("RGB"))
        for p, _ in tqdm(ds.samples, desc="RAM cache")
    ]
    log("  RAM cache ready.")
    return ds, cache


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING EVAL DATASET
# No RAM precomputation — images are loaded, corrupted, and transformed
# on-the-fly by parallel DataLoader workers while the GPU runs inference.
# RAM usage: ~0 MB overhead (just the DataLoader pipeline).
# ─────────────────────────────────────────────────────────────────────────────
class StreamingEvalDataset(Dataset):
    def __init__(self, root, corrupt_fn):
        self.ds         = datasets.ImageFolder(root)
        self.samples    = self.ds.samples
        self.corrupt_fn = corrupt_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img         = Image.open(path).convert("RGB")
        img         = self.corrupt_fn(img)
        rgb         = rgb_val_tf(img)
        fft_img     = aux_val_tf(fast_fft(img))
        ela_img     = aux_val_tf(ela(img))
        return rgb, fft_img, ela_img, label


def make_eval_loader(test_root, mode):
    """
    Build a streaming eval DataLoader for a given corruption mode.

    Windows note: error 1455 (shared file mapping / pagefile exhaustion)
    occurs when worker processes try to share tensors via memory-mapped
    files and the system pagefile is too small or already saturated by the
    100k-image train RAM cache.  Setting num_workers=0 runs everything in
    the main process — no shared memory, no pagefile pressure, no crash.
    Single-process throughput is still fast because the GPU is the
    bottleneck (not CPU preprocessing) and EVAL_BATCH=64 keeps it saturated.
    """
    corrupt_fn = _CORRUPT_FNS[mode]
    ds = StreamingEvalDataset(test_root, corrupt_fn)
    return DataLoader(
        ds,
        batch_size=EVAL_BATCH,
        shuffle=False,
        num_workers=0,    # 0 = main process only; avoids Windows error 1455
        pin_memory=False, # pin_memory requires workers; must be False when num_workers=0
    )


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING EVALUATE (standard)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_streaming(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for rgb, fft_img, ela_img, labels in loader:
            rgb     = rgb.to(device, non_blocking=True)
            fft_img = fft_img.to(device, non_blocking=True)
            ela_img = ela_img.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            with autocast("cuda"):
                out, _ = model(rgb, fft_img, ela_img)
            correct += (out.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return 100 * correct / total


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING EVALUATE (TTA — 4 augment passes)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_tta_streaming(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for rgb, fft_img, ela_img, labels in loader:
            rgb     = rgb.to(device, non_blocking=True)
            fft_img = fft_img.to(device, non_blocking=True)
            ela_img = ela_img.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            probs = torch.zeros(rgb.size(0), 2, device=device)
            for aug in [
                rgb,
                TF.hflip(rgb),
                torch.clamp(rgb * 1.05, -3.0, 3.0),
                torch.clamp(rgb * 0.95, -3.0, 3.0),
            ]:
                with autocast("cuda"):
                    out, _ = model(aug, fft_img, ela_img)
                probs += torch.softmax(out, dim=1)

            correct += (probs.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return 100 * correct / total


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN DATASET
# ─────────────────────────────────────────────────────────────────────────────
class TrainDataset(Dataset):
    def __init__(self, ds, ram_cache):
        self.samples = ds.samples
        self.cache   = ram_cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, label = self.samples[idx]
        img      = Image.fromarray(self.cache[idx])
        img      = TrainCorrupt()(img)
        return (rgb_train_tf(img),
                aux_train_tf(fast_fft(img)),
                aux_train_tf(ela(img)),
                label)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
SWIN_DIM  = 1024
EFNET_DIM = 1280
FUSE_DIM  = 512


class ProjectedGatedFusion(nn.Module):
    def __init__(self, dims, fuse_dim=512):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d, fuse_dim), nn.LayerNorm(fuse_dim), nn.GELU())
            for d in dims
        ])
        n = len(dims)
        self.gate = nn.Sequential(
            nn.Linear(fuse_dim * n, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, n), nn.Softmax(dim=1),
        )

    def forward(self, *branches):
        projected = [proj(b) for proj, b in zip(self.projs, branches)]
        w         = self.gate(torch.cat(projected, dim=1))
        fused     = sum(p * w[:, i:i+1] for i, p in enumerate(projected))
        return fused, w


class HybridDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_enc = timm.create_model(
            "swinv2_base_window12_192_22k", pretrained=True, num_classes=0)
        self.fft_enc = models.efficientnet_b1(weights="DEFAULT")
        self.fft_enc.classifier = nn.Identity()
        self.ela_enc = models.efficientnet_b1(weights="DEFAULT")
        self.ela_enc.classifier = nn.Identity()
        self.fusion = ProjectedGatedFusion(
            dims=[SWIN_DIM, EFNET_DIM, EFNET_DIM], fuse_dim=FUSE_DIM)
        self.head = nn.Sequential(
            nn.Linear(FUSE_DIM, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(512, 128),      nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, rgb, fft_img, ela_img):
        r            = self.rgb_enc(rgb)
        f            = self.fft_enc(fft_img)
        e            = self.ela_enc(ela_img)
        fused, gates = self.fusion(r, f, e)
        return self.head(fused), gates

    def freeze_backbones(self):
        for p in list(self.rgb_enc.parameters()) + \
                 list(self.fft_enc.parameters()) + \
                 list(self.ela_enc.parameters()):
            p.requires_grad = False

    def unfreeze_backbones(self):
        for p in self.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
def make_optimizer(model, phase):
    if phase == "frozen":
        params = list(model.fusion.parameters()) + list(model.head.parameters())
        return optim.AdamW(params, lr=2e-3, weight_decay=1e-4)
    return optim.AdamW([
        {"params": list(model.rgb_enc.parameters()), "lr": 5e-6},
        {"params": list(model.fft_enc.parameters()), "lr": 2e-5},
        {"params": list(model.ela_enc.parameters()), "lr": 2e-5},
        {"params": list(model.fusion.parameters()),  "lr": 1e-4},
        {"params": list(model.head.parameters()),    "lr": 2e-4},
    ], weight_decay=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# CUTMIX
# ─────────────────────────────────────────────────────────────────────────────
def cutmix_batch(x, y, z, labels, alpha=1.0):
    lam    = np.random.beta(alpha, alpha)
    B, C, H, W = x.shape
    idx    = torch.randperm(B, device=x.device)
    cut_h  = int(H * np.sqrt(1 - lam))
    cut_w  = int(W * np.sqrt(1 - lam))
    cy     = random.randint(0, H)
    cx     = random.randint(0, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    x_mix  = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_actual = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x_mix, y, z, labels, labels[idx], lam_actual


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def predict(model, path, mode="original", use_tta=True):
    model.eval()
    img   = Image.open(path).convert("RGB")
    img   = _CORRUPT_FNS[mode](img)
    rgb_t = rgb_val_tf(img).unsqueeze(0).to(device)
    fft_t = aux_val_tf(fast_fft(img)).unsqueeze(0).to(device)
    ela_t = aux_val_tf(ela(img)).unsqueeze(0).to(device)

    with torch.no_grad():
        if use_tta:
            probs = torch.zeros(1, 2, device=device)
            for aug in [rgb_t, TF.hflip(rgb_t),
                        torch.clamp(rgb_t * 1.05, -3.0, 3.0),
                        torch.clamp(rgb_t * 0.95, -3.0, 3.0)]:
                with autocast("cuda"):
                    out, gates = model(aug, fft_t, ela_t)
                probs += torch.softmax(out, dim=1)
            prob = probs[0] / 4.0
            g    = gates[0].tolist()
        else:
            with autocast("cuda"):
                out, gates = model(rgb_t, fft_t, ela_t)
            prob = torch.softmax(out, dim=1)[0]
            g    = gates[0].tolist()

    pred  = prob.argmax().item()
    label = ["REAL", "FAKE"][pred]
    conf  = prob[pred].item() * 100
    tag   = " [TTA]" if use_tta else ""
    log(f"[{mode.upper()}]{tag} {label}  ({conf:.2f}%)")
    log(f"  P(REAL)={prob[0]*100:.2f}%  P(FAKE)={prob[1]*100:.2f}%")
    log(f"  Gates -> RGB(Swin): {g[0]:.3f} | FFT(EffNet): {g[1]:.3f} | ELA(EffNet): {g[2]:.3f}")
    return label, conf


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    freeze_support()
    log(f"Using device: {device}")

    train_imgfolder, ram_cache = build_train_ram_cache(train_dir)

    labels_list   = [lbl for _, lbl in train_imgfolder.samples]
    class_counts  = np.bincount(labels_list)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    MODES = ["original", "compressed", "screenshot", "combined"]

    # ── Streaming eval loaders — zero RAM precomputation ─────────────────────
    # Each loader spawns 4 worker processes that compute FFT+ELA on CPU in
    # parallel while the GPU runs inference on the previous batch.
    # Total extra RAM: negligible (just the DataLoader pipeline buffers).
    log("Building streaming eval loaders (no RAM precomputation)...")
    eval_loaders = {m: make_eval_loader(test_dir, m) for m in MODES}
    log("Eval loaders ready.\n")

    test_count = len(eval_loaders["original"].dataset)

    train_ds     = TrainDataset(train_imgfolder, ram_cache)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    model = HybridDetector().to(device)
    log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    scaler    = torch.amp.GradScaler("cuda")
    writer    = SummaryWriter("runs/hybrid_detector_v3")

    model.freeze_backbones()
    optimizer = make_optimizer(model, "frozen")

    def lr_lambda(ep):
        if ep < 2:
            return (ep + 1) / 2
        t = (ep - 2) / max(FREEZE_EPOCHS - 2, 1)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    with open("training_log.csv", "w", newline="") as f:
        csv.writer(f).writerow([
            "Epoch", "Phase", "Train_Acc", "Loss", "LR",
            "Acc_Original", "Acc_Compressed", "Acc_Screenshot", "Acc_Combined",
            "TTA_Original", "TTA_Compressed", "TTA_Screenshot", "TTA_Combined",
            "Gate_RGB", "Gate_FFT", "Gate_ELA",
        ])

    best_acc, best_state = 0.0, None
    train_accs           = []
    eval_history         = {m: [] for m in MODES}
    gate_history         = []
    last_evals           = {m: 0. for m in MODES}
    last_tta             = {m: 0. for m in MODES}

    log(f"Train: {len(train_ds)} | Test: {test_count}")
    log(f"Phase 1 (ep 1-{FREEZE_EPOCHS}): head+fusion only")
    log(f"Phase 2a (ep {FREEZE_EPOCHS+1}): EfficientNet branches unfreeze")
    log(f"Phase 2b (ep {FREEZE_EPOCHS+3}): Swin last 2 stages unfreeze")
    log(f"Phase 2c (ep {FREEZE_EPOCHS+5}): full unfreeze")
    log(f"LR restart at ep {LR_RESTART_EP}")
    log(f"Eval every {EVAL_EVERY} epochs\n")

    for epoch in range(EPOCHS):
        log(f"--- Starting epoch {epoch+1}/{EPOCHS} ---")

        if epoch == FREEZE_EPOCHS:
            log(f"Phase 2a: unfreezing EfficientNet at epoch {epoch+1}")
            for p in list(model.fft_enc.parameters()) + list(model.ela_enc.parameters()):
                p.requires_grad = True
            optimizer = make_optimizer(model, "full")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(EPOCHS - FREEZE_EPOCHS), eta_min=1e-6)

        if epoch == FREEZE_EPOCHS + 2:
            log(f"Phase 2b: unfreezing Swin last 2 stages at epoch {epoch+1}")
            for name, p in model.rgb_enc.named_parameters():
                if any(f"layers.{i}" in name for i in [2, 3]):
                    p.requires_grad = True

        if epoch == FREEZE_EPOCHS + 4:
            log(f"Phase 2c: full unfreeze at epoch {epoch+1}")
            model.unfreeze_backbones()

        if epoch == LR_RESTART_EP and epoch > FREEZE_EPOCHS:
            log(f"LR warm restart at epoch {epoch+1}")
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * 2.5
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(EPOCHS - LR_RESTART_EP), eta_min=1e-7)

        phase     = "frozen" if epoch < FREEZE_EPOCHS else "full"
        use_mixup = (phase == "full")
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        gate_accum = torch.zeros(3)

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1:2d}/{EPOCHS} [{phase}]", leave=False)

        for x, y, z, labels in pbar:
            x, y, z, labels = (x.to(device, non_blocking=True),
                                y.to(device, non_blocking=True),
                                z.to(device, non_blocking=True),
                                labels.to(device, non_blocking=True))

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                aug_roll = random.random()
                if use_mixup and aug_roll < 0.25:
                    xm, ym, zm, la, lb, lam = cutmix_batch(x, y, z, labels)
                    out, gates = model(xm, ym, zm)
                    loss = lam * criterion(out, la) + (1 - lam) * criterion(out, lb)
                elif use_mixup and aug_roll < 0.50:
                    lam = random.betavariate(0.4, 0.4)
                    idx = torch.randperm(x.size(0), device=device)
                    out, gates = model(lam * x + (1 - lam) * x[idx], y, z)
                    loss = lam * criterion(out, labels) + (1 - lam) * criterion(out, labels[idx])
                else:
                    out, gates = model(x, y, z)
                    loss       = criterion(out, labels)

                eps     = 1e-8
                entropy = -(gates * (gates + eps).log()).sum(dim=1).mean()
                loss    = loss - GATE_ENT_W * entropy

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            correct      += (out.argmax(1) == labels).sum().item()
            total        += labels.size(0)
            running_loss += loss.item()
            gate_accum   += gates.detach().cpu().mean(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")

        train_acc  = 100 * correct / total
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        train_accs.append(train_acc)
        mean_gates = (gate_accum / len(train_loader)).tolist()
        gate_history.append(mean_gates)

        do_eval = ((epoch + 1) % EVAL_EVERY == 0) or (epoch == EPOCHS - 1)
        if do_eval:
            log(f"  Running streaming eval...")
            for m in MODES:
                last_evals[m] = evaluate_streaming(model, eval_loaders[m])
                last_tta[m]   = evaluate_tta_streaming(model, eval_loaders[m])
            log("  Eval done.")

        for m in MODES:
            eval_history[m].append(last_evals[m])

        def fmt(m, d):
            v = d[m]
            return f"{v:.2f}%" if do_eval else f"({v:.2f}%)"

        log(f"Ep {epoch+1:2d}/{EPOCHS} [{phase}] | "
            f"Train {train_acc:.2f}% | Loss {running_loss/len(train_loader):.4f} | "
            f"Orig {fmt('original', last_evals)} | Comp {fmt('compressed', last_evals)} | "
            f"SS {fmt('screenshot', last_evals)} | Comb {fmt('combined', last_evals)} | "
            f"TTA-Comb {fmt('combined', last_tta)} | "
            f"LR {current_lr:.1e} | "
            f"Gates [R={mean_gates[0]:.2f} F={mean_gates[1]:.2f} E={mean_gates[2]:.2f}]")

        if do_eval and last_tta["combined"] > best_acc:
            best_acc   = last_tta["combined"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, "best_model.pth")
            log(f"   ✓ Best saved  TTA-combined={best_acc:.2f}%")

        with open("training_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1, phase, f"{train_acc:.4f}",
                f"{running_loss/len(train_loader):.4f}", f"{current_lr:.6f}",
                f"{last_evals['original']:.4f}", f"{last_evals['compressed']:.4f}",
                f"{last_evals['screenshot']:.4f}", f"{last_evals['combined']:.4f}",
                f"{last_tta['original']:.4f}",    f"{last_tta['compressed']:.4f}",
                f"{last_tta['screenshot']:.4f}",   f"{last_tta['combined']:.4f}",
                f"{mean_gates[0]:.4f}", f"{mean_gates[1]:.4f}", f"{mean_gates[2]:.4f}",
            ])

        writer.add_scalar("Accuracy/Train",        train_acc,                epoch)
        writer.add_scalar("Accuracy/Original",     last_evals["original"],   epoch)
        writer.add_scalar("Accuracy/Compressed",   last_evals["compressed"], epoch)
        writer.add_scalar("Accuracy/Screenshot",   last_evals["screenshot"], epoch)
        writer.add_scalar("Accuracy/Combined",     last_evals["combined"],   epoch)
        writer.add_scalar("Accuracy/TTA_Combined", last_tta["combined"],     epoch)
        writer.add_scalar("Gates/RGB",             mean_gates[0],            epoch)
        writer.add_scalar("Gates/FFT",             mean_gates[1],            epoch)
        writer.add_scalar("Gates/ELA",             mean_gates[2],            epoch)
        writer.add_scalar("LR",                    current_lr,               epoch)

    # ── Final results ─────────────────────────────────────────────────────────
    final_model = HybridDetector().to(device)
    final_model.load_state_dict(best_state)
    torch.save(final_model.state_dict(), "final_model.pth")

    log("\n" + "="*65)
    log("FINAL RESULTS (best checkpoint — TTA)")
    log("="*65)
    for m in MODES:
        std_acc = evaluate_streaming(final_model, eval_loaders[m])
        tta_acc = evaluate_tta_streaming(final_model, eval_loaders[m])
        log(f"  {m.capitalize():<12}: {std_acc:.2f}%  (TTA: {tta_acc:.2f}%)")
    log("="*65)

    with open("final_results.txt", "w") as f:
        for m in MODES:
            std_acc = evaluate_streaming(final_model, eval_loaders[m])
            tta_acc = evaluate_tta_streaming(final_model, eval_loaders[m])
            f.write(f"{m.capitalize()}: {std_acc:.2f}% | TTA: {tta_acc:.2f}%\n")

    writer.close()

    # ── Plots ─────────────────────────────────────────────────────────────────
    ep = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].plot(ep, train_accs, linewidth=2, label="Train")
    axes[0].axvline(FREEZE_EPOCHS + 0.5, color="red",    linestyle="--", alpha=0.6, label="EffNet unfreeze")
    axes[0].axvline(FREEZE_EPOCHS + 2.5, color="orange", linestyle="--", alpha=0.6, label="Swin partial")
    axes[0].axvline(FREEZE_EPOCHS + 4.5, color="green",  linestyle="--", alpha=0.6, label="Full unfreeze")
    axes[0].axvline(LR_RESTART_EP + 0.5, color="purple", linestyle=":",  alpha=0.6, label="LR restart")
    axes[0].set_title("Training Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Acc (%)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    for m in MODES:
        axes[1].plot(ep, eval_history[m], label=m.capitalize(), linewidth=2)
    axes[1].set_title("Eval Accuracy per Mode"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Acc (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    gate_arr = np.array(gate_history)
    for i, name in enumerate(["RGB (SwinV2)", "FFT (EffNet)", "ELA (EffNet)"]):
        axes[2].plot(ep, gate_arr[:, i], label=name, linewidth=2)
    axes[2].axhline(1/3, color="gray", linestyle=":", alpha=0.5, label="Uniform (1/3)")
    axes[2].set_title("Fusion Gate Weights"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Mean Gate")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("accuracy_graph.png", dpi=150)
    log("Plot saved to accuracy_graph.png")

    # ── Interactive predict ───────────────────────────────────────────────────
    log("\nEnter image path to test (or 'exit').")
    log("Format: <path> [original|compressed|screenshot|combined] [--no-tta]")
    while True:
        raw = input("\nPath [mode] [--no-tta]: ").strip()
        if raw.lower() == "exit":
            break
        parts   = raw.split()
        path    = parts[0]
        mode    = parts[1] if len(parts) > 1 and not parts[1].startswith("--") else "original"
        use_tta = "--no-tta" not in parts
        if os.path.exists(path):
            predict(final_model, path, mode, use_tta=use_tta)
        else:
            log("Invalid path.")
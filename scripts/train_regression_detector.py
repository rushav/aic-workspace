"""
Train a ResNet-18 regression port detector — ONE port per model (2 outputs: u, v).

Each model is trained on samples containing that specific port key.
No missing-port masking needed since every sample has the target port.

  --port-key sfp_port_0 → trained only on samples with sfp_port_0
  --port-key sfp_port_1 → trained only on samples with sfp_port_1
  --port-key sc_port_0  → trained only on samples with sc_port_0
  --port-key sc_port_1  → trained only on samples with sc_port_1

Input: 256x256 camera image, ImageNet normalized
Output: (u_norm, v_norm) — normalized pixel coordinates for ONE port

Run: source ~/aic-workspace/train-env/bin/activate
     python scripts/train_regression_detector.py --port-key sfp_port_0
     python scripts/train_regression_detector.py --port-key sfp_port_1
     python scripts/train_regression_detector.py --port-key sc_port_0
     python scripts/train_regression_detector.py --port-key sc_port_1
"""

import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json, glob, os, numpy as np, random

IMG_SIZE = 256
ORIG_W, ORIG_H = 1152, 1024

VALID_PORT_KEYS = ["sfp_port_0", "sfp_port_1", "sc_port_0", "sc_port_1"]


# ── Model ────────────────────────────────────────────────────────────────────

class PortDetector(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),  # just u_norm, v_norm
        )

    def forward(self, x):
        feat = self.features(x)
        return self.head(feat)


# ── Dataset ──────────────────────────────────────────────────────────────────

class PortDataset(Dataset):
    def __init__(self, sample_paths, port_key, img_size=IMG_SIZE, augment=False):
        self.samples = sample_paths
        self.port_key = port_key
        self.img_size = img_size
        self.augment = augment
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx]) as f:
            label = json.load(f)

        img_path = os.path.join(os.path.dirname(self.samples[idx]), label["image"])
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Extract normalized (u, v) for the single port
        p = label["ports"][self.port_key]
        u_norm = p["u"] / orig_w
        v_norm = p["v"] / orig_h

        # Augmentation
        if self.augment:
            img = self.color_jitter(img)

            # Random horizontal flip — mirror u
            if random.random() < 0.5:
                img = TF.hflip(img)
                u_norm = 1.0 - u_norm

            # Small random translate (±5%)
            tx = random.uniform(-0.05, 0.05)
            ty = random.uniform(-0.05, 0.05)
            img = TF.affine(img, angle=0, translate=[int(tx * orig_w), int(ty * orig_h)],
                            scale=1.0, shear=0, interpolation=T.InterpolationMode.BILINEAR)
            u_norm += tx
            v_norm += ty

        # Resize and normalize
        img = TF.resize(img, [self.img_size, self.img_size])
        img_tensor = TF.to_tensor(img)
        img_tensor = self.normalize(img_tensor)

        coords = torch.tensor([u_norm, v_norm], dtype=torch.float32)
        return img_tensor, coords


# ── Training ─────────────────────────────────────────────────────────────────

def train(port_key):
    print(f"Training single-port detector for: {port_key}")

    data_dir = os.path.expanduser("~/aic-workspace/datasets/port_detection")
    save_dir = os.path.expanduser("~/aic-workspace/checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Collect samples that contain this port key
    all_json = sorted(glob.glob(f"{data_dir}/sample_*.json"))
    valid_samples = []
    for jp in all_json:
        with open(jp) as f:
            d = json.load(f)
        if port_key in d["ports"]:
            valid_samples.append(jp)

    print(f"Found {len(valid_samples)} samples with {port_key} (out of {len(all_json)} total)")

    if len(valid_samples) < 5:
        print(f"ERROR: Not enough samples for {port_key}")
        return

    # 85/15 split
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(valid_samples))
    n_val = max(1, len(valid_samples) // 6)  # ~15%
    val_idx = set(indices[:n_val].tolist())
    train_paths = [valid_samples[i] for i in range(len(valid_samples)) if i not in val_idx]
    val_paths = [valid_samples[i] for i in range(len(valid_samples)) if i in val_idx]
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    train_set = PortDataset(train_paths, port_key, augment=True)
    val_set = PortDataset(val_paths, port_key, augment=False)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PortDetector().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=15, factor=0.5)

    best_val_px = float("inf")
    patience_counter = 0
    num_epochs = 500

    for epoch in range(num_epochs):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0
        for imgs, coords in train_loader:
            imgs = imgs.to(device)
            coords = coords.to(device)  # (B, 2)

            pred = model(imgs)  # (B, 2) → u_norm, v_norm
            loss = nn.functional.mse_loss(pred, coords)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0
        pixel_errors = []

        with torch.no_grad():
            for imgs, coords in val_loader:
                imgs = imgs.to(device)
                coords = coords.to(device)

                pred = model(imgs)
                loss = nn.functional.mse_loss(pred, coords)
                val_loss += loss.item()

                # Compute pixel errors
                for b in range(imgs.shape[0]):
                    pu = pred[b, 0].item() * ORIG_W
                    pv = pred[b, 1].item() * ORIG_H
                    gu = coords[b, 0].item() * ORIG_W
                    gv = coords[b, 1].item() * ORIG_H
                    err = np.sqrt((pu - gu) ** 2 + (pv - gv) ** 2)
                    pixel_errors.append(err)

        val_loss /= len(val_loader)

        mean_px = np.mean(pixel_errors) if pixel_errors else float("inf")
        median_px = np.median(pixel_errors) if pixel_errors else float("inf")

        scheduler.step(mean_px)

        if epoch % 10 == 0 or mean_px < best_val_px:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | loss: {train_loss:.6f}/{val_loss:.6f} | "
                  f"px: mean={mean_px:.1f} med={median_px:.1f} | lr: {lr:.2e}")

        if mean_px < best_val_px:
            best_val_px = mean_px
            torch.save(model.state_dict(), f"{save_dir}/{port_key}_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 60:
            print(f"Early stopping at epoch {epoch}")
            break

    torch.save(model.state_dict(), f"{save_dir}/{port_key}_final.pth")

    # ── Final evaluation on best model ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training complete ({port_key}). Loading best model...")
    model.load_state_dict(torch.load(f"{save_dir}/{port_key}_best.pth", weights_only=True))
    model.eval()

    pixel_errors = []
    with torch.no_grad():
        for imgs, coords in val_loader:
            imgs = imgs.to(device)
            coords = coords.to(device)
            pred = model(imgs)

            for b in range(imgs.shape[0]):
                pu = pred[b, 0].item() * ORIG_W
                pv = pred[b, 1].item() * ORIG_H
                gu = coords[b, 0].item() * ORIG_W
                gv = coords[b, 1].item() * ORIG_H
                err = np.sqrt((pu - gu) ** 2 + (pv - gv) ** 2)
                pixel_errors.append(err)

    print(f"\n┌──────────────┬──────────────┬──────────────┬────────┐")
    print(f"│ Port         │ Mean px err  │ Median px err│ Count  │")
    print(f"├──────────────┼──────────────┼──────────────┼────────┤")
    if pixel_errors:
        print(f"│ {port_key:12s} │   {np.mean(pixel_errors):6.1f} px   │   {np.median(pixel_errors):6.1f} px   │ {len(pixel_errors):5d}  │")
        print(f"└──────────────┴──────────────┴──────────────┴────────┘")
        print(f"  p90={np.percentile(pixel_errors, 90):.0f}px  max={np.max(pixel_errors):.0f}px")
    else:
        print(f"│ {port_key:12s} │      N/A      │      N/A      │     0  │")
        print(f"└──────────────┴──────────────┴──────────────┴────────┘")

    print(f"\nBest mean pixel error: {best_val_px:.1f}px")
    print(f"Model saved to {save_dir}/{port_key}_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-key", required=True, choices=VALID_PORT_KEYS)
    args = parser.parse_args()
    train(args.port_key)

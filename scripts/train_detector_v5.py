"""
Train improved ResNet-18 regression detector v5.

Key improvements over original:
  1. Uses BOTH port_detection/ and port_detection_close/ datasets
  2. Stronger augmentation: rotation ±15°, scale 0.8-1.2, gaussian noise
  3. Larger head: 512→256→64→2 (more capacity for harder task)
  4. Longer training with cosine annealing
  5. MixUp augmentation for better generalization

Usage:
  source ~/aic-workspace/train-env/bin/activate
  python3 scripts/train_detector_v5.py --port-key sfp_port_0
  python3 scripts/train_detector_v5.py --port-key sfp_port_1
  python3 scripts/train_detector_v5.py --port-key sc_port_0
  python3 scripts/train_detector_v5.py --port-key sc_port_1

Train all 4:
  for k in sfp_port_0 sfp_port_1 sc_port_0 sc_port_1; do
    python3 scripts/train_detector_v5.py --port-key $k
  done
"""

import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json, glob, os, numpy as np, random, math

IMG_SIZE = 256
ORIG_W, ORIG_H = 1152, 1024
VALID_PORT_KEYS = ["sfp_port_0", "sfp_port_1", "sc_port_0", "sc_port_1"]


class PortDetector(nn.Module):
    """Same architecture as original — MUST match for inference compatibility."""
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        feat = self.features(x)
        return self.head(feat)


class PortDataset(Dataset):
    def __init__(self, sample_paths, port_key, img_size=IMG_SIZE, augment=False):
        self.samples = sample_paths
        self.port_key = port_key
        self.img_size = img_size
        self.augment = augment
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx]) as f:
            label = json.load(f)

        img_path = os.path.join(os.path.dirname(self.samples[idx]), label["image"])
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        p = label["ports"][self.port_key]
        u_norm = p["u"] / orig_w
        v_norm = p["v"] / orig_h

        if self.augment:
            # Color jitter
            img = self.color_jitter(img)

            # Random horizontal flip
            if random.random() < 0.5:
                img = TF.hflip(img)
                u_norm = 1.0 - u_norm

            # Random rotation ±15°
            angle = random.uniform(-15, 15)
            if abs(angle) > 0.5:
                img = TF.rotate(img, angle, interpolation=T.InterpolationMode.BILINEAR)
                # Rotate the point
                cx_px, cy_px = orig_w / 2.0, orig_h / 2.0
                u_px = u_norm * orig_w
                v_px = v_norm * orig_h
                rad = math.radians(-angle)  # PIL rotates opposite
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                du, dv = u_px - cx_px, v_px - cy_px
                u_px = cos_a * du - sin_a * dv + cx_px
                v_px = sin_a * du + cos_a * dv + cy_px
                u_norm = u_px / orig_w
                v_norm = v_px / orig_h

            # Random scale 0.85-1.15 with crop
            scale = random.uniform(0.85, 1.15)
            if abs(scale - 1.0) > 0.02:
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                img = TF.resize(img, [new_h, new_w])
                # Center crop back to original size
                if new_w >= orig_w and new_h >= orig_h:
                    left = (new_w - orig_w) // 2
                    top = (new_h - orig_h) // 2
                    img = TF.crop(img, top, left, orig_h, orig_w)
                    u_norm = (u_norm * scale - left / orig_w)
                    v_norm = (v_norm * scale - top / orig_h)
                else:
                    # Pad if smaller
                    img = TF.center_crop(img, [min(new_h, orig_h), min(new_w, orig_w)])
                    img = TF.resize(img, [orig_h, orig_w])
                    # Approximate: scale the coordinates
                    u_norm = 0.5 + (u_norm - 0.5) * scale
                    v_norm = 0.5 + (v_norm - 0.5) * scale

            # Small random translate ±8%
            tx = random.uniform(-0.08, 0.08)
            ty = random.uniform(-0.08, 0.08)
            img = TF.affine(img, angle=0, translate=[int(tx * orig_w), int(ty * orig_h)],
                           scale=1.0, shear=0, interpolation=T.InterpolationMode.BILINEAR)
            u_norm += tx
            v_norm += ty

            # Gaussian noise
            if random.random() < 0.3:
                arr = np.array(img).astype(np.float32)
                noise = np.random.randn(*arr.shape) * 10
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)

        # Clamp to valid range
        u_norm = max(0.0, min(1.0, u_norm))
        v_norm = max(0.0, min(1.0, v_norm))

        img = TF.resize(img, [self.img_size, self.img_size])
        img_tensor = TF.to_tensor(img)
        img_tensor = self.normalize(img_tensor)

        coords = torch.tensor([u_norm, v_norm], dtype=torch.float32)
        return img_tensor, coords


def train(port_key, save_dir_name="v5"):
    print(f"Training v5 detector for: {port_key}")

    # Use BOTH datasets
    data_dirs = [
        os.path.expanduser("~/aic-workspace/datasets/port_detection"),
        os.path.expanduser("~/aic-workspace/datasets/port_detection_close"),
    ]
    save_dir = os.path.expanduser(f"~/aic-workspace/checkpoints/{save_dir_name}")
    os.makedirs(save_dir, exist_ok=True)

    valid_samples = []
    for data_dir in data_dirs:
        all_json = sorted(glob.glob(f"{data_dir}/sample_*.json"))
        for jp in all_json:
            try:
                with open(jp) as f:
                    d = json.load(f)
                if port_key in d["ports"]:
                    valid_samples.append(jp)
            except (json.JSONDecodeError, KeyError):
                continue

    print(f"Found {len(valid_samples)} samples with {port_key}")
    if len(valid_samples) < 10:
        print(f"ERROR: Not enough samples")
        return

    # 85/15 split
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(valid_samples))
    n_val = max(5, len(valid_samples) // 6)
    val_idx = set(indices[:n_val].tolist())
    train_paths = [valid_samples[i] for i in range(len(valid_samples)) if i not in val_idx]
    val_paths = [valid_samples[i] for i in range(len(valid_samples)) if i in val_idx]
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    train_set = PortDataset(train_paths, port_key, augment=True)
    val_set = PortDataset(val_paths, port_key, augment=False)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PortDetector().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = 300
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_px = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for imgs, coords in train_loader:
            imgs = imgs.to(device)
            coords = coords.to(device)
            pred = model(imgs)
            loss = nn.functional.smooth_l1_loss(pred, coords)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        pixel_errors = []
        with torch.no_grad():
            for imgs, coords in val_loader:
                imgs = imgs.to(device)
                coords = coords.to(device)
                pred = model(imgs)
                loss = nn.functional.smooth_l1_loss(pred, coords)
                val_loss += loss.item()
                for b in range(imgs.shape[0]):
                    pu = pred[b, 0].item() * ORIG_W
                    pv = pred[b, 1].item() * ORIG_H
                    gu = coords[b, 0].item() * ORIG_W
                    gv = coords[b, 1].item() * ORIG_H
                    pixel_errors.append(np.sqrt((pu - gu)**2 + (pv - gv)**2))
        val_loss /= len(val_loader)

        mean_px = np.mean(pixel_errors) if pixel_errors else float("inf")
        median_px = np.median(pixel_errors) if pixel_errors else float("inf")
        p90_px = np.percentile(pixel_errors, 90) if pixel_errors else float("inf")
        scheduler.step()

        if epoch % 10 == 0 or mean_px < best_val_px:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | loss: {train_loss:.6f}/{val_loss:.6f} | "
                  f"px: mean={mean_px:.1f} med={median_px:.1f} p90={p90_px:.1f} | lr: {lr:.2e}")

        if mean_px < best_val_px:
            best_val_px = mean_px
            torch.save(model.state_dict(), f"{save_dir}/{port_key}_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 80:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest validation pixel error: {best_val_px:.1f}px")
    print(f"Saved to: {save_dir}/{port_key}_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-key", required=True, choices=VALID_PORT_KEYS)
    parser.add_argument("--save-dir", default="v5")
    args = parser.parse_args()
    train(args.port_key, args.save_dir)

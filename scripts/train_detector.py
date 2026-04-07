"""
Train a heatmap-based port keypoint detector (v8 — masked loss, all layers unfrozen).

Each model is a 2-channel detector:
  --port-type sfp → channels: sfp_port_0, sfp_port_1
  --port-type sc  → channels: sc_port_0, sc_port_1

Samples are included if they have ANY of the two port keys (not both required).
Missing channels get zero heatmaps and are masked out of the loss.
All ResNet layers are unfrozen for full fine-tuning.

Input: 384x384 camera image
Output: 2-channel heatmap (96x96) → soft-argmax → (u,v) per port

Run in train-env: source ~/aic-workspace/train-env/bin/activate
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json, glob, os, numpy as np, random

IMG_SIZE = 384
HEATMAP_SIZE = 96
ORIG_W, ORIG_H = 1152, 1024
SIGMA = 5.0

PORT_KEYS = {
    "sfp": ["sfp_port_0", "sfp_port_1"],
    "sc": ["sc_port_0", "sc_port_1"],
}


# ── Heatmap utilities ────────────────────────────────────────────────────────
def make_gaussian_heatmap(cx, cy, h, w, sigma):
    y = torch.arange(h, dtype=torch.float32)
    x = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))


def soft_argmax_2d(heatmap, temperature=10.0):
    """Differentiable soft-argmax: extract (x, y) from heatmap."""
    B, C, H, W = heatmap.shape
    flat = heatmap.view(B, C, -1)
    weights = F.softmax(flat * temperature, dim=-1).view(B, C, H, W)

    device = heatmap.device
    y_coords = torch.arange(H, dtype=torch.float32, device=device).view(1, 1, H, 1)
    x_coords = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, 1, W)

    x = (weights * x_coords).sum(dim=(2, 3))
    y = (weights * y_coords).sum(dim=(2, 3))
    return torch.stack([x, y], dim=-1)  # (B, C, 2)


# ── Dataset ──────────────────────────────────────────────────────────────────
class PortDataset(Dataset):
    def __init__(self, sample_paths, port_keys, img_size=IMG_SIZE,
                 hm_size=HEATMAP_SIZE, sigma=SIGMA, augment=False):
        self.samples = sample_paths
        self.port_keys = port_keys  # e.g. ["sfp_port_0", "sfp_port_1"]
        self.img_size = img_size
        self.hm_size = hm_size
        self.sigma = sigma
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

        # Extract normalized (u, v) for each port key; missing ports get None
        raw_uvs = []
        for key in self.port_keys:
            if key in label["ports"]:
                p = label["ports"][key]
                raw_uvs.append((p["u"] / orig_w, p["v"] / orig_h))
            else:
                raw_uvs.append(None)

        # Augmentation — only transform coordinates that exist
        if self.augment:
            img = self.color_jitter(img)

            if random.random() < 0.5:
                img = TF.hflip(img)
                # Swap port 0 ↔ 1 and mirror u
                swapped = [None, None]
                if raw_uvs[1] is not None:
                    swapped[0] = (1.0 - raw_uvs[1][0], raw_uvs[1][1])
                if raw_uvs[0] is not None:
                    swapped[1] = (1.0 - raw_uvs[0][0], raw_uvs[0][1])
                raw_uvs = swapped

            tx = random.uniform(-0.06, 0.06)
            ty = random.uniform(-0.06, 0.06)
            scale = random.uniform(0.88, 1.12)
            img = TF.affine(img, angle=0, translate=[int(tx * orig_w), int(ty * orig_h)],
                            scale=scale, shear=0, interpolation=T.InterpolationMode.BILINEAR)
            for i in range(2):
                if raw_uvs[i] is not None:
                    u, v = raw_uvs[i]
                    u = (u - 0.5) * scale + 0.5 + tx
                    v = (v - 0.5) * scale + 0.5 + ty
                    raw_uvs[i] = (u, v)

        img = TF.resize(img, [self.img_size, self.img_size])
        img_tensor = TF.to_tensor(img)

        if self.augment and random.random() < 0.3:
            img_tensor = (img_tensor + torch.randn_like(img_tensor) * 0.03).clamp(0, 1)

        img_tensor = self.normalize(img_tensor)

        if self.augment and random.random() < 0.4:
            img_tensor = T.RandomErasing(p=1.0, scale=(0.02, 0.1))(img_tensor)

        # Build heatmaps, coords, and validity mask
        hm_list = []
        coords_list = []
        valid_list = []

        for i in range(2):
            if raw_uvs[i] is not None:
                u, v = raw_uvs[i]
                hm_list.append(make_gaussian_heatmap(
                    u * self.hm_size, v * self.hm_size,
                    self.hm_size, self.hm_size, self.sigma))
                coords_list.extend([u, v])
                valid_list.append(1.0)
            else:
                hm_list.append(torch.zeros(self.hm_size, self.hm_size))
                coords_list.extend([-1.0, -1.0])
                valid_list.append(0.0)

        heatmaps = torch.stack(hm_list, dim=0)
        coords = torch.tensor(coords_list, dtype=torch.float32)
        valid = torch.tensor(valid_list, dtype=torch.float32)

        return img_tensor, heatmaps, coords, valid


# ── Model ────────────────────────────────────────────────────────────────────
class HeatmapDetector(nn.Module):
    def __init__(self, num_keypoints=2):
        super().__init__()
        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.conv1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # All layers unfrozen — full fine-tuning from pretrained weights

        # Decoder
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Dropout2d(0.15))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout2d(0.1))
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.head = nn.Conv2d(64, num_keypoints, 1)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d3 = self.up4(x4)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))
        d2 = self.up3(d3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))
        d1 = self.up2(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))
        return self.head(d1)


# ── Training ─────────────────────────────────────────────────────────────────
def train(port_type):
    port_keys = PORT_KEYS[port_type]
    ckpt_name = f"{port_type}_detector"
    print(f"Training {port_type.upper()} detector: channels = {port_keys}")

    data_dir = os.path.expanduser("~/aic-workspace/datasets/port_detection")
    save_dir = os.path.expanduser("~/aic-workspace/checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Survey all samples and print detailed counts
    all_json = sorted(glob.glob(f"{data_dir}/sample_*.json"))
    valid_samples = []
    has_0 = 0
    has_1 = 0
    has_both = 0
    for jp in all_json:
        with open(jp) as f:
            d = json.load(f)
        p0 = port_keys[0] in d["ports"]
        p1 = port_keys[1] in d["ports"]
        if p0:
            has_0 += 1
        if p1:
            has_1 += 1
        if p0 and p1:
            has_both += 1
        # Include if ANY port key present
        if p0 or p1:
            valid_samples.append(jp)

    print(f"{port_type.upper()} samples: {len(valid_samples)} "
          f"(with {port_keys[0]}: {has_0}, with {port_keys[1]}: {has_1}, with both: {has_both})")

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(valid_samples))
    n_val = max(1, len(valid_samples) // 5)
    val_idx = set(indices[:n_val].tolist())
    train_paths = [valid_samples[i] for i in range(len(valid_samples)) if i not in val_idx]
    val_paths = [valid_samples[i] for i in range(len(valid_samples)) if i in val_idx]
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    train_set = PortDataset(train_paths, port_keys, augment=True)
    val_set = PortDataset(val_paths, port_keys, augment=False)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HeatmapDetector(num_keypoints=2).to(device)

    trainable = list(model.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"Parameters: {n_total:,} total, {n_trainable:,} trainable ({100*n_trainable/n_total:.0f}%)")

    optimizer = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=1e-4)
    num_epochs = 300
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_px = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for imgs, heatmaps, coords, valid in train_loader:
            imgs = imgs.to(device)
            heatmaps = heatmaps.to(device)
            coords = coords.to(device)
            valid = valid.to(device)  # (B, 2)

            pred_hm = model(imgs)

            # Per-channel masked heatmap MSE
            hm_diff = (pred_hm - heatmaps) ** 2  # (B, 2, H, W)
            valid_hw = valid.unsqueeze(-1).unsqueeze(-1)  # (B, 2, 1, 1)
            n_valid_pixels = valid_hw.sum() * HEATMAP_SIZE * HEATMAP_SIZE
            hm_loss = (hm_diff * valid_hw).sum() / (n_valid_pixels + 1e-8)

            # Per-channel masked coordinate loss via soft-argmax
            pred_coords = soft_argmax_2d(pred_hm)  # (B, 2, 2)
            gt_hm_coords = coords.view(-1, 2, 2) * HEATMAP_SIZE  # (B, 2, 2)
            # Zero out invalid gt coords so they don't produce NaN gradients
            valid_c = valid.unsqueeze(-1)  # (B, 2, 1)
            gt_hm_coords = gt_hm_coords * valid_c
            pred_masked = pred_coords * valid_c
            coord_diff = F.smooth_l1_loss(pred_masked, gt_hm_coords, reduction='none')  # (B, 2, 2)
            coord_loss = (coord_diff * valid_c).sum() / (valid_c.sum() * 2 + 1e-8)

            loss = hm_loss + 0.1 * coord_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        pixel_errors = []
        with torch.no_grad():
            for imgs, heatmaps, coords, valid in val_loader:
                imgs = imgs.to(device)
                heatmaps = heatmaps.to(device)
                coords = coords.to(device)
                valid = valid.to(device)

                pred_hm = model(imgs)

                valid_hw = valid.unsqueeze(-1).unsqueeze(-1)
                hm_diff = (pred_hm - heatmaps) ** 2
                n_valid_pixels = valid_hw.sum() * HEATMAP_SIZE * HEATMAP_SIZE
                hm_loss = (hm_diff * valid_hw).sum() / (n_valid_pixels + 1e-8)

                pred_c = soft_argmax_2d(pred_hm)
                gt_c = coords.view(-1, 2, 2) * HEATMAP_SIZE
                valid_c = valid.unsqueeze(-1)
                gt_c_masked = gt_c * valid_c
                pred_c_masked = pred_c * valid_c
                coord_diff = F.smooth_l1_loss(pred_c_masked, gt_c_masked, reduction='none')
                coord_loss = (coord_diff * valid_c).sum() / (valid_c.sum() * 2 + 1e-8)
                val_loss += (hm_loss + 0.1 * coord_loss).item()

                for b in range(imgs.shape[0]):
                    for p in range(2):
                        if valid[b, p].item() < 0.5:
                            continue
                        pu = pred_c[b, p, 0].item() / HEATMAP_SIZE * ORIG_W
                        pv = pred_c[b, p, 1].item() / HEATMAP_SIZE * ORIG_H
                        gu = coords[b, p*2].item() * ORIG_W
                        gv = coords[b, p*2+1].item() * ORIG_H
                        pixel_errors.append(np.sqrt((pu - gu)**2 + (pv - gv)**2))

        val_loss /= len(val_loader)
        mean_px = np.mean(pixel_errors)
        median_px = np.median(pixel_errors)
        p90 = np.percentile(pixel_errors, 90)
        max_px = np.max(pixel_errors)
        scheduler.step()

        if epoch % 10 == 0 or mean_px < best_val_px:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | loss: {train_loss:.5f}/{val_loss:.5f} | "
                  f"px: mean={mean_px:.1f} med={median_px:.1f} p90={p90:.0f} max={max_px:.0f} | lr: {lr:.2e}")

        if mean_px < best_val_px:
            best_val_px = mean_px
            torch.save(model.state_dict(), f"{save_dir}/{ckpt_name}_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 80:
            print(f"Early stopping at epoch {epoch}")
            break

    torch.save(model.state_dict(), f"{save_dir}/{ckpt_name}_final.pth")
    print(f"\nTraining complete ({port_type.upper()}).")
    print(f"Best mean pixel error: {best_val_px:.1f}px")
    print(f"Models saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-type", required=True, choices=["sfp", "sc"],
                        help="Which port type to train: sfp or sc")
    args = parser.parse_args()
    train(args.port_type)

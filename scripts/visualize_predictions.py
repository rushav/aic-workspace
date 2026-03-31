"""
Visualize heatmap detector predictions (v5 model).

Shows predictions on all validation samples with error stats.
Green circle = ground truth, Red circle = prediction, Yellow line = error.

Run in train-env: source ~/aic-workspace/train-env/bin/activate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import json, glob, os, numpy as np

IMG_SIZE = 384
HEATMAP_SIZE = 96
ORIG_W, ORIG_H = 1152, 1024


def soft_argmax_2d(heatmap, temperature=10.0):
    B, C, H, W = heatmap.shape
    flat = heatmap.view(B, C, -1)
    weights = F.softmax(flat * temperature, dim=-1).view(B, C, H, W)
    device = heatmap.device
    y_coords = torch.arange(H, dtype=torch.float32, device=device).view(1, 1, H, 1)
    x_coords = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, 1, W)
    x = (weights * x_coords).sum(dim=(2, 3))
    y = (weights * y_coords).sum(dim=(2, 3))
    return torch.stack([x, y], dim=-1)


class HeatmapDetector(nn.Module):
    def __init__(self, num_keypoints=2):
        super().__init__()
        backbone = models.resnet34(weights=None)
        self.conv1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
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


def main():
    ckpt = os.path.expanduser("~/aic-workspace/checkpoints/port_detector_best.pth")
    data_dir = os.path.expanduser("~/aic-workspace/datasets/port_detection")
    save_dir = os.path.expanduser("~/aic-workspace/data/predictions")
    os.makedirs(save_dir, exist_ok=True)

    model = HeatmapDetector(num_keypoints=2)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    samples = sorted(glob.glob(f"{data_dir}/sample_*.json"))

    all_errors = []
    worst_samples = []

    for i, json_path in enumerate(samples):
        with open(json_path) as f:
            label = json.load(f)

        if "sfp_port_0" not in label["ports"] or "sfp_port_1" not in label["ports"]:
            continue

        img_path = os.path.join(data_dir, label["image"])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Prepare input
        inp = TF.resize(img, [IMG_SIZE, IMG_SIZE])
        inp = TF.to_tensor(inp)
        inp = normalize(inp).unsqueeze(0)

        with torch.no_grad():
            pred_hm = model(inp)
            pred_coords = soft_argmax_2d(pred_hm)  # (1, 2, 2)

        errors_this = []
        points = []
        for p_idx, port_name in enumerate(["sfp_port_0", "sfp_port_1"]):
            gt = label["ports"][port_name]
            gt_u, gt_v = gt["u"], gt["v"]

            pred_u = pred_coords[0, p_idx, 0].item() / HEATMAP_SIZE * w
            pred_v = pred_coords[0, p_idx, 1].item() / HEATMAP_SIZE * h

            error = np.sqrt((pred_u - gt_u)**2 + (pred_v - gt_v)**2)
            errors_this.append(error)
            all_errors.append(error)
            points.append((gt_u, gt_v, pred_u, pred_v, error, port_name))

        avg_err = np.mean(errors_this)
        worst_samples.append((avg_err, i, label["image"]))

        # Only visualize worst 20 and best 10
        # (we'll sort and pick later)

    # Sort by error
    worst_samples.sort(reverse=True)
    all_errors = np.array(all_errors)

    print(f"Total predictions: {len(all_errors)}")
    print(f"Mean pixel error: {all_errors.mean():.1f}px")
    print(f"Median pixel error: {np.median(all_errors):.1f}px")
    print(f"P90 pixel error: {np.percentile(all_errors, 90):.1f}px")
    print(f"P95 pixel error: {np.percentile(all_errors, 95):.1f}px")
    print(f"Max pixel error: {all_errors.max():.1f}px")
    print(f"Under 10px: {(all_errors < 10).sum()}/{len(all_errors)} ({100*(all_errors < 10).mean():.0f}%)")
    print(f"Under 20px: {(all_errors < 20).sum()}/{len(all_errors)} ({100*(all_errors < 20).mean():.0f}%)")
    print(f"Under 50px: {(all_errors < 50).sum()}/{len(all_errors)} ({100*(all_errors < 50).mean():.0f}%)")

    print(f"\nWorst 10 samples:")
    for err, idx, name in worst_samples[:10]:
        print(f"  {name}: avg error = {err:.1f}px")

    # Visualize worst 15 and best 5
    to_visualize = [s for s in worst_samples[:15]] + [s for s in worst_samples[-5:]]

    for rank, (avg_err, sample_idx, img_name) in enumerate(to_visualize):
        json_path = os.path.join(data_dir, img_name.replace('.png', '.json'))
        with open(json_path) as f:
            label = json.load(f)

        img = Image.open(os.path.join(data_dir, img_name)).convert("RGB")
        w, h = img.size

        inp = TF.resize(img, [IMG_SIZE, IMG_SIZE])
        inp = TF.to_tensor(inp)
        inp = normalize(inp).unsqueeze(0)

        with torch.no_grad():
            pred_hm = model(inp)
            pred_coords = soft_argmax_2d(pred_hm)

        draw = ImageDraw.Draw(img)

        for p_idx, port_name in enumerate(["sfp_port_0", "sfp_port_1"]):
            gt = label["ports"][port_name]
            gt_u, gt_v = gt["u"], gt["v"]
            pred_u = pred_coords[0, p_idx, 0].item() / HEATMAP_SIZE * w
            pred_v = pred_coords[0, p_idx, 1].item() / HEATMAP_SIZE * h
            error = np.sqrt((pred_u - gt_u)**2 + (pred_v - gt_v)**2)

            r = 10
            draw.ellipse([gt_u-r, gt_v-r, gt_u+r, gt_v+r], outline="green", width=3)
            draw.ellipse([pred_u-r, pred_v-r, pred_u+r, pred_v+r], outline="red", width=3)
            draw.line([gt_u, gt_v, pred_u, pred_v], fill="yellow", width=2)
            draw.text((pred_u+12, pred_v-8), f"{error:.0f}px", fill="red")

        tag = "worst" if rank < 15 else "best"
        out_name = f"{tag}_{rank:02d}_{avg_err:.0f}px_{img_name}"
        img.save(os.path.join(save_dir, out_name))

    print(f"\nSaved visualizations to {save_dir}/")


if __name__ == "__main__":
    main()

"""
Train a simple port keypoint detector.

Input: 256x256 center-cropped camera image
Output: (u, v) pixel coordinates of SFP port 0 entrance

Uses ResNet-18 backbone with a regression head.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json, glob, os, numpy as np

# ── Dataset ──────────────────────────────────────────────────────────────────
class PortDataset(Dataset):
    def __init__(self, data_dir, img_size=256):
        self.samples = sorted(glob.glob(f"{data_dir}/sample_*.json"))
        self.img_size = img_size
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx]) as f:
            label = json.load(f)

        img_path = os.path.join(os.path.dirname(self.samples[idx]), label["image"])
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Normalize pixel coords to [0, 1]
        port = label["ports"]["sfp_port_0"]
        u_norm = port["u"] / orig_w
        v_norm = port["v"] / orig_h

        # Also store the 3D position in camera frame (for later PnP)
        target = torch.tensor([
            u_norm, v_norm,
            port["x"], port["y"], port["z"],
        ], dtype=torch.float32)

        img_tensor = self.transform(img)
        return img_tensor, target


# ── Model ────────────────────────────────────────────────────────────────────
class PortDetector(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the classification head
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # Regression head: predict (u, v, x, y, z)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        feat = self.features(x)
        return self.head(feat)


# ── Training ─────────────────────────────────────────────────────────────────
def train():
    data_dir = os.path.expanduser("~/aic-workspace/datasets/port_detection")
    save_dir = os.path.expanduser("~/aic-workspace/checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    dataset = PortDataset(data_dir)
    print(f"Dataset: {len(dataset)} samples")

    # Split 80/20 train/val
    n_val = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PortDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_val_loss = float("inf")

    for epoch in range(100):
        # Train
        model.train()
        train_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = nn.functional.mse_loss(preds, targets)
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
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                loss = nn.functional.mse_loss(preds, targets)
                val_loss += loss.item()

                # Compute pixel error in original image coords
                pred_u = preds[:, 0].cpu().numpy() * 1152
                pred_v = preds[:, 1].cpu().numpy() * 1024
                gt_u = targets[:, 0].cpu().numpy() * 1152
                gt_v = targets[:, 1].cpu().numpy() * 1024
                errors = np.sqrt((pred_u - gt_u)**2 + (pred_v - gt_v)**2)
                pixel_errors.extend(errors.tolist())

        val_loss /= len(val_loader)
        mean_px_error = np.mean(pixel_errors)
        scheduler.step()

        if epoch % 10 == 0 or val_loss < best_val_loss:
            print(f"Epoch {epoch:3d} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | pixel_error: {mean_px_error:.1f}px")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_dir}/port_detector_best.pth")

    # Save final model
    torch.save(model.state_dict(), f"{save_dir}/port_detector_final.pth")
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Mean pixel error: {mean_px_error:.1f}px")
    print(f"Models saved to {save_dir}/")


if __name__ == "__main__":
    train()

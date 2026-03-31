import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
from PIL import Image, ImageDraw
import json, glob, os, numpy as np

class PortDetector(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5),
        )
    def forward(self, x):
        return self.head(self.features(x))

# Load model
model = PortDetector()
ckpt = os.path.expanduser("~/aic-workspace/checkpoints/port_detector_best.pth")
model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
model.eval()

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = os.path.expanduser("~/aic-workspace/datasets/port_detection")
save_dir = os.path.expanduser("~/aic-workspace/data/predictions")
os.makedirs(save_dir, exist_ok=True)

samples = sorted(glob.glob(f"{data_dir}/sample_*.json"))

# Visualize first 10
for i, json_path in enumerate(samples[:10]):
    with open(json_path) as f:
        label = json.load(f)

    img_path = os.path.join(data_dir, label["image"])
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # Ground truth
    gt = label["ports"]["sfp_port_0"]
    gt_u, gt_v = gt["u"], gt["v"]

    # Prediction
    inp = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(inp)[0].numpy()
    pred_u = pred[0] * w
    pred_v = pred[1] * h

    error = np.sqrt((pred_u - gt_u)**2 + (pred_v - gt_v)**2)

    # Draw on image
    draw = ImageDraw.Draw(img)
    # Green = ground truth
    draw.ellipse([gt_u-8, gt_v-8, gt_u+8, gt_v+8], outline="green", width=3)
    # Red = prediction
    draw.ellipse([pred_u-8, pred_v-8, pred_u+8, pred_v+8], outline="red", width=3)
    # Line between them
    draw.line([gt_u, gt_v, pred_u, pred_v], fill="yellow", width=2)

    img.save(f"{save_dir}/pred_{i:03d}_{error:.0f}px.png")
    print(f"Sample {i}: error={error:.1f}px  gt=({gt_u:.0f},{gt_v:.0f}) pred=({pred_u:.0f},{pred_v:.0f})")

print(f"\nSaved to {save_dir}/")

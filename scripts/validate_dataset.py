"""
Dataset validation script — visual diagnostics for port detection data quality.

Step 1: Basic stats (sample counts, coordinate bounds, per-port distributions)
Step 2: Visual spot-check (draw labeled crosshairs on random samples)
Step 3: Early vs late sample comparison (original 308 vs newer)

Run: cd ~/aic-workspace && source train-env/bin/activate && python scripts/validate_dataset.py
"""

import json, glob, os, random
import numpy as np
from PIL import Image, ImageDraw

DATA_DIR = os.path.expanduser("~/aic-workspace/datasets/port_detection")
CHECK_DIR = os.path.expanduser("~/aic-workspace/datasets/validation_check")

# ── Step 1: Basic stats ─────────────────────────────────────────────────────

files = sorted(glob.glob(f"{DATA_DIR}/sample_*.json"))
print(f"Total samples: {len(files)}")

all_coords = {"sfp_port_0": [], "sfp_port_1": [], "sc_port_0": [], "sc_port_1": []}
bad_samples = []

for f in files:
    d = json.load(open(f))
    img_path = os.path.join(os.path.dirname(f), d["image"])

    # Check image exists
    if not os.path.exists(img_path):
        bad_samples.append((f, "missing image"))
        continue

    img = Image.open(img_path)
    w, h = img.size

    for port_name, port_data in d["ports"].items():
        u, v = port_data["u"], port_data["v"]
        # Check bounds
        if u < 0 or u >= w or v < 0 or v >= h:
            bad_samples.append((f, f"{port_name} out of bounds: ({u:.1f}, {v:.1f}) in {w}x{h}"))
        if port_name in all_coords:
            all_coords[port_name].append((u, v))

print(f"\nBad samples: {len(bad_samples)}")
for path, reason in bad_samples[:20]:
    print(f"  {os.path.basename(path)}: {reason}")

# Print coordinate statistics per port
for port_name, coords in all_coords.items():
    if coords:
        coords_arr = np.array(coords)
        print(f"\n{port_name}: {len(coords_arr)} samples")
        print(f"  u: mean={coords_arr[:,0].mean():.1f}, std={coords_arr[:,0].std():.1f}, "
              f"min={coords_arr[:,0].min():.1f}, max={coords_arr[:,0].max():.1f}")
        print(f"  v: mean={coords_arr[:,1].mean():.1f}, std={coords_arr[:,1].std():.1f}, "
              f"min={coords_arr[:,1].min():.1f}, max={coords_arr[:,1].max():.1f}")
    else:
        print(f"\n{port_name}: 0 samples")

# ── Step 2: Visual spot-check ────────────────────────────────────────────────

os.makedirs(CHECK_DIR, exist_ok=True)
random.seed(42)
check_samples = random.sample(files, min(10, len(files)))

for f in check_samples:
    d = json.load(open(f))
    img_path = os.path.join(os.path.dirname(f), d["image"])
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for port_name, port_data in d["ports"].items():
        u, v = port_data["u"], port_data["v"]
        color = "red" if "sfp" in port_name else "blue"
        # Draw crosshair
        draw.ellipse([u - 10, v - 10, u + 10, v + 10], outline=color, width=3)
        draw.line([u - 15, v, u + 15, v], fill=color, width=2)
        draw.line([u, v - 15, u, v + 15], fill=color, width=2)
        draw.text((u + 12, v - 10), port_name, fill=color)

    basename = os.path.basename(f).replace(".json", "_check.png")
    img.save(os.path.join(CHECK_DIR, basename))

print(f"\nSaved {len(check_samples)} validation images to {CHECK_DIR}/")

# ── Step 3: Early vs late sample comparison ──────────────────────────────────

early = files[:308]
late = files[308:]

for label, subset in [("Original 308", early), ("Newer samples", late)]:
    for port_key in ["sfp_port_0", "sfp_port_1", "sc_port_0", "sc_port_1"]:
        port_coords = []
        for f in subset:
            d = json.load(open(f))
            if port_key in d["ports"]:
                p = d["ports"][port_key]
                entry = [p["u"], p["v"]]
                if "z" in p:
                    entry.append(p["z"])
                port_coords.append(entry)
        if port_coords:
            arr = np.array(port_coords)
            line = (f"\n{label} — {port_key} ({len(port_coords)} samples):\n"
                    f"  u: mean={arr[:,0].mean():.1f} std={arr[:,0].std():.1f}\n"
                    f"  v: mean={arr[:,1].mean():.1f} std={arr[:,1].std():.1f}")
            if arr.shape[1] > 2:
                line += f"\n  z (depth): mean={arr[:,2].mean():.3f} std={arr[:,2].std():.3f}"
            print(line)

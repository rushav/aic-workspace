"""
Convert port_detection JSON dataset to YOLO format.

Class IDs: 0=sfp_port_0, 1=sfp_port_1, 2=sc_port_0, 3=sc_port_1
Bounding boxes use fixed sizes based on approximate port dimensions:
  SFP: ~40x30 px, SC: ~60x35 px (normalized by image dimensions)
"""

import json, glob, os, shutil, random

DATA_DIR = os.path.expanduser("~/aic-workspace/datasets/port_detection")
OUT_DIR = os.path.expanduser("~/aic-workspace/datasets/yolo_ports")

CLASS_MAP = {
    "sfp_port_0": 0,
    "sfp_port_1": 1,
    "sc_port_0": 2,
    "sc_port_1": 3,
}

# Fixed bbox sizes in pixels (will be normalized per-image)
BBOX_SIZES = {
    "sfp_port_0": (40, 30),
    "sfp_port_1": (40, 30),
    "sc_port_0": (60, 35),
    "sc_port_1": (60, 35),
}

NAMES = ["sfp_port_0", "sfp_port_1", "sc_port_0", "sc_port_1"]


def main():
    all_json = sorted(glob.glob(f"{DATA_DIR}/sample_*.json"))
    print(f"Found {len(all_json)} samples")

    # Shuffle with fixed seed, 85/15 split
    random.seed(42)
    indices = list(range(len(all_json)))
    random.shuffle(indices)
    n_val = max(1, int(len(all_json) * 0.15))
    val_set = set(indices[:n_val])

    # Create directory structure
    for split in ("train", "val"):
        os.makedirs(f"{OUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/labels/{split}", exist_ok=True)

    counts = {"train": 0, "val": 0}
    port_counts = {name: 0 for name in NAMES}

    for i, jp in enumerate(all_json):
        with open(jp) as f:
            label = json.load(f)

        img_w = label["width"]
        img_h = label["height"]
        ports = label["ports"]

        if not ports:
            continue

        split = "val" if i in val_set else "train"
        stem = os.path.splitext(label["image"])[0]

        # Write YOLO label file
        lines = []
        for port_name, cls_id in CLASS_MAP.items():
            if port_name not in ports:
                continue
            p = ports[port_name]
            cx = p["u"] / img_w
            cy = p["v"] / img_h
            bw, bh = BBOX_SIZES[port_name]
            w_norm = bw / img_w
            h_norm = bh / img_h
            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}")
            port_counts[port_name] += 1

        if not lines:
            continue

        label_path = f"{OUT_DIR}/labels/{split}/{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        # Symlink image
        src_img = os.path.join(DATA_DIR, label["image"])
        dst_img = f"{OUT_DIR}/images/{split}/{label['image']}"
        if not os.path.exists(dst_img):
            os.symlink(os.path.abspath(src_img), dst_img)

        counts[split] += 1

    # Write data.yaml
    yaml_content = f"""path: {OUT_DIR}
train: images/train
val: images/val

nc: 4
names: {NAMES}
"""
    with open(f"{OUT_DIR}/data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Train: {counts['train']}, Val: {counts['val']}")
    print("Port annotation counts:")
    for name in NAMES:
        print(f"  {name}: {port_counts[name]}")
    print(f"YOLO dataset written to {OUT_DIR}/")
    print(f"Config: {OUT_DIR}/data.yaml")


if __name__ == "__main__":
    main()

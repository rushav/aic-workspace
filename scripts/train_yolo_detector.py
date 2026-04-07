"""
Train YOLOv8n port detector and evaluate centroid accuracy.

Standalone experiment to compare against the heatmap-based detector.
"""

import argparse, json, glob, os, numpy as np
from collections import defaultdict


ORIG_W, ORIG_H = 1152, 1024
DATA_YAML = os.path.expanduser("~/aic-workspace/datasets/yolo_ports/data.yaml")
DATA_DIR = os.path.expanduser("~/aic-workspace/datasets/port_detection")
PROJECT_DIR = os.path.expanduser("~/aic-workspace/checkpoints/yolo")

CLASS_NAMES = ["sfp_port_0", "sfp_port_1", "sc_port_0", "sc_port_1"]


def train():
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=DATA_YAML,
        epochs=200,
        imgsz=640,
        batch=16,
        patience=50,
        device=0,
        project=PROJECT_DIR,
        name="port_detector",
        exist_ok=True,
    )
    return results


def evaluate():
    from ultralytics import YOLO

    weights = f"{PROJECT_DIR}/port_detector/weights/best.pt"
    if not os.path.exists(weights):
        print(f"ERROR: {weights} not found")
        return
    model = YOLO(weights)

    # Load val images and their ground truth from the original JSON labels
    val_img_dir = os.path.expanduser("~/aic-workspace/datasets/yolo_ports/images/val")
    val_images = sorted(glob.glob(f"{val_img_dir}/*.png"))
    print(f"\nEvaluating on {len(val_images)} val images...")

    # Build GT lookup: stem -> {class_name: (u_px, v_px)}
    gt_lookup = {}
    for jp in sorted(glob.glob(f"{DATA_DIR}/sample_*.json")):
        with open(jp) as f:
            label = json.load(f)
        stem = os.path.splitext(label["image"])[0]
        gt_lookup[stem] = {}
        for port_name in CLASS_NAMES:
            if port_name in label["ports"]:
                p = label["ports"][port_name]
                gt_lookup[stem][port_name] = (p["u"], p["v"])

    # Run predictions
    errors = defaultdict(list)  # class_name -> [pixel_errors]
    gt_counts = defaultdict(int)
    det_counts = defaultdict(int)

    for img_path in val_images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt = gt_lookup.get(stem, {})

        results = model.predict(img_path, conf=0.25, verbose=False)
        if len(results) == 0:
            for port_name in gt:
                gt_counts[port_name] += 1
            continue

        r = results[0]
        boxes = r.boxes

        # Count GT ports in this image
        for port_name in gt:
            gt_counts[port_name] += 1

        # Match detections to GT by class
        for box in boxes:
            cls_id = int(box.cls.item())
            class_name = CLASS_NAMES[cls_id]

            if class_name not in gt:
                continue

            # Box center in the predicted image coords -> scale to original
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            # Predictions are in the input image space (resized to imgsz)
            # but ultralytics returns coords in original image space
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            gt_u, gt_v = gt[class_name]
            err = np.sqrt((cx - gt_u) ** 2 + (cy - gt_v) ** 2)
            errors[class_name].append(err)
            det_counts[class_name] += 1

    # Print comparison table
    print()
    print("┌────────────┬──────────────┬──────────────┬───────────┐")
    print("│ Port Class │ Mean px err  │ Median px err│ Detections│")
    print("├────────────┼──────────────┼──────────────┼───────────┤")
    for name in CLASS_NAMES:
        errs = errors.get(name, [])
        n_gt = gt_counts.get(name, 0)
        n_det = det_counts.get(name, 0)
        if errs:
            mean_e = np.mean(errs)
            med_e = np.median(errs)
            print(f"│ {name:10s} │   {mean_e:6.1f} px   │   {med_e:6.1f} px   │  {n_det:3d}/{n_gt:<3d}  │")
        else:
            print(f"│ {name:10s} │      N/A      │      N/A      │  {n_det:3d}/{n_gt:<3d}  │")
    print("└────────────┴──────────────┴──────────────┴───────────┘")

    # Overall summary
    all_errs = []
    for e in errors.values():
        all_errs.extend(e)
    if all_errs:
        print(f"\nOverall: mean={np.mean(all_errs):.1f}px, median={np.median(all_errs):.1f}px, "
              f"p90={np.percentile(all_errs, 90):.0f}px, max={np.max(all_errs):.0f}px")
        total_det = sum(det_counts.values())
        total_gt = sum(gt_counts.values())
        print(f"Detection rate: {total_det}/{total_gt} ({100*total_det/total_gt:.1f}%)")


def main():
    print("=== YOLOv8n Port Detector Training ===\n")
    train()
    print("\n=== Centroid Accuracy Evaluation ===")
    evaluate()


if __name__ == "__main__":
    main()

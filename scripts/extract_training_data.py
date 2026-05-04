#!/usr/bin/env python3
"""Extract detector training data from CheatCodeRecorder demos.

For each trial, extracts step 0 (survey-pose frame) with ground truth
pixel coordinates from the npz file.

Usage:
    source train-env/bin/activate
    python3 scripts/extract_training_data.py
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np


def extract_from_episode(episode_dir, output_dir, sample_idx):
    npz_path = os.path.join(episode_dir, "episode.npz")
    meta_path = os.path.join(episode_dir, "metadata.json")

    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        return sample_idx, False

    try:
        data = np.load(npz_path, allow_pickle=True)
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        print(f"  Error: {episode_dir}: {e}")
        return sample_idx, False

    # Get port info from metadata
    port_type = meta.get("port_type", "sfp")
    port_name = meta.get("port_name", "sfp_port_0")
    target_module = meta.get("target_module_name", "")

    # Determine port key for training
    if port_type == "sfp":
        if "port_0" in port_name:
            port_key = "sfp_port_0"
        else:
            port_key = "sfp_port_1"
    else:
        if "sc_port_0" in target_module:
            port_key = "sc_port_0"
        else:
            port_key = "sc_port_1"

    # Get ground truth pixel coordinates from npz
    gt_center = data.get("gt_port_pixels_center", None)
    if gt_center is None or len(gt_center) == 0:
        return sample_idx, False

    # Use step 0 (survey pose)
    for step_idx in [0, 1, 2]:
        if step_idx >= len(gt_center):
            continue

        u = float(gt_center[step_idx, 0])
        v = float(gt_center[step_idx, 1])

        if u < 0 or u > 1200 or v < 0 or v > 1100:
            continue

        img_name = f"step_{step_idx:04d}_center.png"
        img_path = os.path.join(episode_dir, img_name)
        if not os.path.exists(img_path):
            continue

        # Get 3D position if available
        gt_pos = data.get("gt_port_positions", None)
        port_3d = {}
        if gt_pos is not None and step_idx < len(gt_pos):
            p = gt_pos[step_idx]
            if len(p) >= 3:
                port_3d = {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}

        sample_name = f"sample_{sample_idx:05d}"
        out_img = os.path.join(output_dir, f"{sample_name}.png")
        out_json = os.path.join(output_dir, f"{sample_name}.json")

        shutil.copy2(img_path, out_img)

        label = {
            "image": f"{sample_name}.png",
            "width": 1152,
            "height": 1024,
            "fx": 1236.63,
            "fy": 1236.63,
            "cx": 576.0,
            "cy": 512.0,
            "ports": {
                port_key: {"u": u, "v": v, **port_3d}
            },
        }

        with open(out_json, "w") as f:
            json.dump(label, f, indent=2)

        return sample_idx + 1, True

    return sample_idx, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos-dir", default=os.path.expanduser("~/aic-workspace/datasets/demos_v2"))
    parser.add_argument("--output-dir", default=os.path.expanduser("~/aic-workspace/datasets/port_detection_v2"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    episodes = sorted(glob.glob(os.path.join(args.demos_dir, "config_*/trial_*")))
    print(f"Found {len(episodes)} episodes")

    existing = len(glob.glob(os.path.join(args.output_dir, "sample_*.json")))
    sample_idx = existing
    success = fail = 0

    for ep_dir in episodes:
        new_idx, ok = extract_from_episode(ep_dir, args.output_dir, sample_idx)
        if ok:
            sample_idx = new_idx
            success += 1
        else:
            fail += 1

    print(f"Extracted {success}, failed {fail}, total {sample_idx}")

    # Count per port
    all_json = glob.glob(os.path.join(args.output_dir, "sample_*.json"))
    counts = {}
    for jp in all_json:
        d = json.load(open(jp))
        for k in d.get("ports", {}):
            counts[k] = counts.get(k, 0) + 1
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

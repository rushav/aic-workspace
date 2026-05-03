#!/bin/bash
# Start overnight data collection for detector retraining.
# Run this with: bash scripts/start_data_collection.sh
# It will prompt for sudo password once, then run unattended.

set -e
cd ~/aic-workspace

echo "=== Overnight Data Collection ==="
echo "Configs: configs_v3 (200 configs, full 360° yaw)"
echo "Output: datasets/demos/"
echo ""
echo "This will take ~8-12 hours for 200 configs."
echo "Press Ctrl+C to cancel, or enter sudo password to start."
echo ""

python3 scripts/run_data_collection.py \
    --config-dir configs_v3 \
    --start-idx 1 \
    --end-idx 200 \
    2>&1 | tee /tmp/data_collection_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=== Data collection complete ==="
echo "Run: python3 scripts/extract_training_data.py"
echo "Then: python3 scripts/train_detector_v5.py --port-key sfp_port_0"

#!/usr/bin/env bash
# Apply local fixes to the AIC submodule
set -e

AIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../aic" && pwd)"

if grep -q "pypi-options.dependency-overrides" "$AIC_DIR/pixi.toml" 2>/dev/null; then
  echo "RTX 5090 PyTorch fix already applied."
else
  cat >> "$AIC_DIR/pixi.toml" << 'EOF'

# RTX 5090 (Blackwell / sm_120) PyTorch compatibility fix
[pypi-options.dependency-overrides]
torch = ">=2.7.1"
torchvision = ">=0.22.1"
EOF
  echo "RTX 5090 PyTorch fix applied to aic/pixi.toml"
fi

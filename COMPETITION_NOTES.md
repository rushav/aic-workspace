# AIC 2026 — Quick Reference Notes

## Score History

### Portal Submissions

| Version | Date | Portal Score | Strategy |
|---------|------|-------------|----------|
| v17 | Apr 22 | 41.3 | First ACT submission |
| v18 | Apr 23 | 50.1 | ACT + spiral descent |
| v19 | Apr 23 | 47.7 | ACT + force-aware backoff |
| v20 | Apr 24 | 47.7 | ACT + stricter stall detection |
| v27 | Apr 25 | **98.3** | Clean V2 baseline + XY search + force-drop |
| v28 | Apr 26 | 94.5 | Iterative re-detection (regressed) |
| v38 | May 1 | 98.0 | V3 original |
| v44 | May 3 | **123.0** | V3 + v6 detectors (360° yaw data) |

### Local Benchmarks

| Tag | Date | Configs | Mean | Max |
|-----|------|---------|------|-----|
| v32_wide | Apr 27 | 5 | 93.8 | 114.2 |
| v33b | Apr 27 | 5 | 92.4 | 107.0 |
| v34_wideyaw | Apr 27 | 4 | 61.5 | 82.9 |
| v2_final | May 5 | 3 | 101.7 | 112.4 |
| v9_test | May 6 | 3 | 92.5 | 103.4 |

**Best single config:** 160.9 pts (v44, yaw=4.71) — 2 partial SFP insertions

## Key Technical Findings

### Detection Accuracy (The Bottleneck)

| Metric | Best Case | Worst Case |
|--------|-----------|------------|
| Pixel error | 21 px | 115 px |
| XY world error | 6 mm | 40 mm |
| SFP port opening | 13.75 mm wide | 4.225 mm tall |
| SC port opening | ~2.5 mm diameter | — |

**Conclusion:** Even best-case 6mm error barely fits in a 14mm opening. Worst-case is hopeless.

### Port Z Planes (Base Frame)

| Port Type | Measured Z | Originally Used | Error |
|-----------|-----------|----------------|-------|
| SFP entrance | 0.179 m | 0.195 m | 16 mm |
| SC base | 0.015 m | 0.072 m | 57 mm |

Correcting Z improves XY for good detections but Y offset then overcorrects. Original params (Z=0.195/0.072) most robust on average.

### Grasp Geometry

```
GRASP_ROLL  = 0.4432 rad    GRASP_POS_SFP = [0, 0.015, 0.042] m
GRASP_PITCH = -0.4838 rad   GRASP_POS_SC  = [0, 0.015, 0.040] m
GRASP_YAW   = 1.3303 rad    Plug hangs ~25° from vertical
```

### SC Descent Problem

- TCP stalls at Z≈0.14 with survey orientation (gripper pointing down)
- CheatCode uses ~123° rotation (from ground truth TF) to unlock IK
- Never solved without ground truth — SC capped at ~30 pts proximity

### Impedance Controller Notes

- Position mode: set target pose, controller drives toward it. Stalls at kinematic limits.
- Velocity mode: set desired velocity, controller integrates. Better for compliant contact.
- Default stiffness: `[150,150,150,50,50,50]` (N/m, Nm/rad)
- Insertion stiffness: `[60,60,60,30,30,30]` — more compliant
- Force safety: Emergency stop at >22N, slow down at >5N, halt at >12N
- `SFP_DESCENT_Z_END = +0.005` (NOT -0.015) — prevents finger collision

### What Broke Scores

| Idea | Result | Why |
|------|--------|-----|
| Higher stiffness | -24 contact penalty | Pushes too hard into card |
| Deeper descent past port | No improvement | Hits surrounding components |
| SC tilt around base Y | Regression (-14 pts) | Lateral drift > IK benefit |
| Stereo triangulation | Wildly wrong 3D points | Pixel noise compounds |
| Iterative re-detection | -4 pts portal | Second detection adds noise |
| Over-engineered tilt-search | 57 → 93 mean | Too complex, too many edge cases |

### What Helped Most

1. **Diverse training data** (360° yaw): +60 pts on hard configs
2. **Benchmark on random configs**: Exposed overfitting immediately
3. **SFP_DESCENT_Z_END fix** (+0.005): Eliminated -24 penalty
4. **ALIGN phase** (50mm hover, 1.2s dwell): Reduced XY oscillation
5. **Ray-plane Z projection**: Consistent port Z regardless of board pose
6. **Force-drop detection**: Identifies chamfer entry for commit

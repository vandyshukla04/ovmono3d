"""The 294x518 <-> 1920x1080 mapping — MEASURED, adopted 2026-08-26 (M1 Gate A).

Verdict: **PER-AXIS anamorphic scaling, no letterbox padding.**
  u_full = u_518 * (1920/518)      v_full = v_518 * (1080/294)
Evidence (tools/tadetr/verify_resolution_mapping.py, full 345-segment run):
  - T-A3: depth/conf rows 292-294 carry normal data on every sampled segment (0.0% zero depth,
    confidence equal to interior rows) => all 294 rows are real image; the earlier "bottom rows are
    letterbox padding" hypothesis is REFUTED. VGGT squeezed 1080 -> 294 directly (0.9% anamorphic).
  - Full-res json K has cx = 960 = 259 * (1920/518) and cy = 540 = 147 * (1080/294) EXACTLY.
  - T-A1 internal consistency (project 3D centers with 518-space K vs bbox_2d): p50 0.77 px,
    p90 1.95 px over 345 segments; residual is dominated by the legitimate centroid-vs-box-center
    offset (project-wide ~2.6 px 518-space), not the mapping.

Consequences:
  - SAM3 masks (1920x1080) map onto the depth grid by dividing coordinates by (SX, SY) -- nothing
    is cropped, no rows are discarded.
  - Depth-grid intrinsics (cameras.json) and full-res intrinsics relate by diag(SX, SY, 1).
"""
FULL_W, FULL_H = 1920, 1080
GRID_W, GRID_H = 518, 294
SX = FULL_W / GRID_W          # 3.70656...
SY = FULL_H / GRID_H          # 3.67347...


def full_to_grid_xy(x, y):
    return x / SX, y / SY


def grid_to_full_xy(x, y):
    return x * SX, y * SY


def K_grid_to_full(K):
    """(3,3) 518-space intrinsics -> full-res intrinsics."""
    K = K.copy()
    K[0, :] *= SX
    K[1, :] *= SY
    return K

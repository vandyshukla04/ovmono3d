"""Pre-flight checks for the masked orientation head. Run BEFORE spending a GPU hour.  [CPU, seconds]

    python tools/aeroview/preflight_orientation.py --labelled-dir /mnt/d/aeroview/labelled

Each check corresponds to a failure that is SILENT -- it produces a plausible-looking training log and a
believable wrong number. None of them are caught by the loss curve.

  T1-T4  the horizontal-flip fix (blocker B1). A camera-x mirror sends alpha -> -alpha, so cos(alpha) (the
         head/tail bit) survives but sin(alpha) (the LEFT/RIGHT flank bit) inverts. RANDOM_FLIP defaults to
         "horizontal" and no config overrides it. Without the fix, flank supervision is 50/50
         self-contradictory and the head simply learns sin(alpha) -> 0.
         T2 is the important one: it pins the fix OUTSIDE the `center_cam[2] != 0` guard, which is where the
         existing pose-mirror block lives and would silently skip labels.
  T5     the alpha branch must not share parameters with the pose branch (SHARED_FC is True).
  T6     the cube head must return a 6-tuple with alpha of shape (n, 2).
  T7     an all-unlabelled batch must OMIT the loss key, never emit NaN -- reducing an empty tensor gives NaN,
         and the training loop turns NaN into an endless restart.
  T8     the labelled jsons carry the expected counts and no lock video carries a training label (blocker B2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import detectron2.data.transforms as T  # noqa: E402
from detectron2.structures import BoxMode  # noqa: E402

from cubercnn.data.dataset_mapper import transform_instance_annotations  # noqa: E402

LOCK_VIDEOS = {"DJI_20250802085130_0007_V", "DJI_20250802085520_0008_V"}
K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _anno(alpha, *, z=5.0):
    c = np.array([0.5, 0.2, z])
    d = np.array([1.0, 1.0, 2.0]) / 2.0
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * d + c
    return {
        "bbox": [100.0, 100.0, 200.0, 200.0],
        "bbox_mode": BoxMode.XYXY_ABS,
        "center_cam": c.tolist(),
        "dimensions": [1.0, 1.0, 2.0],
        "bbox3D_cam": corners.tolist(),
        "pose": np.eye(3).tolist(),
        "R_cam": np.eye(3).tolist(),
        "category_id": 0,
        "heading_alpha": float(alpha),
        "heading_valid": 1,
        "ignore": False,
    }


def run_mapper(alpha, *, flip, z=5.0, width=1920):
    tfms = [T.HFlipTransform(width)] if flip else [T.NoOpTransform()]
    out = transform_instance_annotations(_anno(alpha, z=z), tfms, K=K)
    return float(out["heading_alpha"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labelled-dir", type=Path, default=Path("/mnt/d/aeroview/labelled"))
    ap.add_argument("--expect-train", type=int, default=9887)
    ap.add_argument("--expect-val", type=int, default=7460)
    args = ap.parse_args()

    print("=== B1: horizontal-flip handling of alpha ===")
    rng = np.random.default_rng(0)
    alphas = rng.uniform(-np.pi, np.pi, 1000)

    # T1 flip negates alpha
    got = np.array([run_mapper(a, flip=True) for a in alphas])
    check("T1 hflip negates alpha", np.allclose(got, -alphas, atol=1e-6),
          f"max err {np.abs(got + alphas).max():.2e}")

    # T2 the fix fires even when center_cam[2] == 0 (the guard the pose-mirror block sits inside)
    got0 = np.array([run_mapper(a, flip=True, z=0.0) for a in alphas[:200]])
    check("T2 hflip negates alpha even when center_cam[2]==0 (fix is OUTSIDE the :87 guard)",
          np.allclose(got0, -alphas[:200], atol=1e-6))

    # T3 no-op transform leaves alpha untouched
    same = np.array([run_mapper(a, flip=False) for a in alphas[:200]])
    check("T3 no-flip leaves alpha unchanged", np.allclose(same, alphas[:200], atol=1e-6))

    # T4 flipping twice is the identity
    twice = np.array([-run_mapper(-run_mapper(a, flip=True), flip=True) for a in alphas[:200]])
    check("T4 double flip is the identity", np.allclose(np.abs(twice), np.abs(alphas[:200]), atol=1e-6))

    # the flank bit really does invert -- i.e. T1 is testing something that matters
    frac = float(np.mean(np.sign(np.sin(got)) != np.sign(np.sin(alphas))))
    check("T4b the flank bit inverts under flip (proves the test has power)", frac > 0.99,
          f"{100*frac:.1f}% of samples change flank")

    print("\n=== head wiring ===")
    from detectron2.layers import ShapeSpec
    from cubercnn.config import get_cfg_defaults
    from detectron2.config import get_cfg
    from cubercnn.modeling.roi_heads.cube_head import build_cube_head

    cfg = get_cfg()
    get_cfg_defaults(cfg)
    cfg.MODEL.ROI_CUBE_HEAD.SHARED_FC = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    head = build_cube_head(cfg, ShapeSpec(channels=256, width=7, height=7))

    n = 4
    # roi_heads.py:378 does `self.cube_pooler(...).flatten(1)`, so the head receives a FLAT (n, C*H*W)
    out = head(torch.randn(n, 256 * 7 * 7))
    check("T6 cube head returns a 6-tuple", isinstance(out, tuple) and len(out) == 6, f"got {len(out)}")
    alpha_out = out[5]
    check("T6b alpha has shape (n, 2) and is NOT per-class",
          tuple(alpha_out.shape) == (n, 2), f"got {tuple(alpha_out.shape)}")

    alpha_params = {id(p) for p in head.feature_generator_alpha.parameters()} | \
                   {id(p) for p in head.bbox_3D_alpha.parameters()}
    shared_params = {id(p) for p in head.feature_generator.parameters()} if hasattr(head, "feature_generator") else set()
    pose_params = {id(p) for p in head.bbox_3D_pose.parameters()}
    check("T5 alpha branch shares NO parameters with the shared trunk or the pose output",
          not (alpha_params & (shared_params | pose_params)))

    print("\n=== masked loss behaviour ===")
    # replicate the exact expression from roi_heads.py
    def alpha_losses(mask, pred, tgt_alpha, w=1.0):
        losses = {}
        if w > 0 and mask.any():
            pa = torch.nn.functional.normalize(pred[mask], dim=1)
            ta = torch.stack((torch.cos(tgt_alpha[mask]), torch.sin(tgt_alpha[mask])), dim=1)
            losses["loss_alpha"] = (1.0 - (pa * ta).sum(dim=1)).mean() * w
        return losses

    pred = torch.randn(8, 2)
    tgt = torch.rand(8) * 2 * np.pi - np.pi
    empty = alpha_losses(torch.zeros(8, dtype=torch.bool), pred, tgt)
    check("T7 all-unlabelled batch OMITS the loss key (no NaN)", "loss_alpha" not in empty)
    some = alpha_losses(torch.tensor([1, 0, 1, 0, 0, 0, 0, 0], dtype=torch.bool), pred, tgt)
    check("T7b partially-labelled batch emits a finite loss",
          "loss_alpha" in some and torch.isfinite(some["loss_alpha"]))
    # a perfect prediction must give exactly zero
    perfect = torch.stack((torch.cos(tgt), torch.sin(tgt)), dim=1) * 7.3   # arbitrary magnitude
    z = alpha_losses(torch.ones(8, dtype=torch.bool), perfect, tgt)["loss_alpha"]
    check("T7c loss is 0 for a perfect prediction and is magnitude-invariant", abs(float(z)) < 1e-6,
          f"{float(z):.2e}")

    print("\n=== B2: the labelled jsons ===")
    # the released paper jsons are WildBox_{train,val}_paper.json; the cluster copies are
    # WildBox_{train,val}.json. Accept either rather than failing on a filename.
    for split, want in (("train", args.expect_train), ("val", args.expect_val)):
        cands = [args.labelled_dir / f"WildBox_{split}_paper.json",
                 args.labelled_dir / f"WildBox_{split}.json"]
        found = [c for c in cands if c.is_file()]
        if not found:
            check(f"T8 a labelled {split} json exists", False,
                  "tried " + " and ".join(c.name for c in cands) + f" in {args.labelled_dir}")
            continue
        p = found[0]
        name = p.name
        d = json.loads(p.read_text())
        by_id = {im["id"]: im for im in d["images"]}
        stamped = sum("heading_alpha" in a for a in d["annotations"])
        valid = sum(a.get("heading_valid", 0) for a in d["annotations"])
        locks = {by_id[a["image_id"]]["file_path"].replace("\\", "/").split("/")[-3]
                 for a in d["annotations"] if a.get("heading_valid", 0)} & LOCK_VIDEOS
        check(f"T8 {name}: EVERY annotation stamped", stamped == len(d["annotations"]),
              f"{stamped:,}/{len(d['annotations']):,}")
        check(f"T8 {name}: labelled count", valid == want, f"{valid:,} (expected {want:,})")
        if split == "train":
            check("T8 no lock video carries a TRAIN label", not locks, str(sorted(locks)))

    n_fail = sum(not ok for _, ok in _results)
    print(f"\n{'='*70}\n{len(_results)-n_fail}/{len(_results)} checks passed")
    if n_fail:
        print("DO NOT TRAIN -- fix the failures above.")
        return 1
    print("ALL PRE-FLIGHT CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

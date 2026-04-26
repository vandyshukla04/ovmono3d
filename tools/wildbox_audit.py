"""
Methodology audit for the OVMono3D × WildBox benchmarking pipeline.

Battery of pass/fail integrity checks against on-disk artifacts: the
two category_meta symlinks, Omni3D stats.json, train/val Omni3D JSONs,
the GDino oracle JSON, and any prediction directories you point at.

Catches the *kinds* of bugs we've actually hit during this benchmark:

  - category_meta symlink divergence (per-class metrics all 0 except one)
  - stats.json missing wildlife species (ValueError: 'plains_zebra' is not in list)
  - thing_classes ordering rule (must be sorted by ascending dataset-id)
  - video-level train/val leakage
  - oracle JSON image_id coverage / out-of-taxonomy category_id
  - iteration-skip-training bug (model_final.pth byte-identical to pretrained)
  - per-class AP table missing in eval log (symlink bug recurrence)
  - prediction category_id space (must be {0..5} contiguous OR {1000..1005} dataset)
  - score / bbox / bbox3D / pose schema validity
  - Rel-AP3D best-scale at grid boundary
  - multi-seed consistency (all seeds have same artifacts)

Usage::

    # Full audit on the canonical 6-species 3-seed setup.
    python tools/wildbox_audit.py \\
        --gt-train      datasets/Omni3D/WildBox_train.json \\
        --gt-val        datasets/Omni3D/WildBox_val.json \\
        --stats         datasets/Omni3D/stats.json \\
        --oracle        datasets/Omni3D/gdino_WildBox_val_oracle_2d.json \\
        --category-meta-top configs/category_meta.json \\
        --category-meta-wbx configs/wildbox/category_meta.json \\
        --pretrained    checkpoints/ovmono3d_lift.pth \\
        --pred-dir      output/wl6_init5sp_multiseed/seed0/eval \\
        --pred-dir      output/wl6_init5sp_multiseed/seed1/eval \\
        --pred-dir      output/wl6_init5sp_multiseed/seed2/eval \\
        --pred-dir      output/wl6_zeroshot_rpn \\
        --pred-dir      output/wl6_zeroshot_oracle2d \\
        --multiseed-dir output/wl6_init5sp_multiseed

If a check FAILs, the eval results may be polluted; investigate before
trusting numbers in the paper.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------- printing helpers ----------

def _section(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _ok(msg: str): print(f"  OK    {msg}")
def _fail(msg: str): print(f"  FAIL  {msg}")
def _warn(msg: str): print(f"  WARN  {msg}")
def _info(msg: str): print(f"        {msg}")


def _try_load_torch():
    """Lazy-import torch (only needed for prediction audits)."""
    return __import__("torch")


# ---------- audit functions ----------

def audit_category_meta(top_path: Path, wbx_path: Path) -> Optional[dict]:
    """Both symlinks must point at the same content; ordering must follow §2.8."""
    _section("[1] Category metadata (symlink consistency + sort order)")

    if not top_path.exists():
        _fail(f"top-level meta not found: {top_path}")
        return None
    if not wbx_path.exists():
        _fail(f"wildbox-dir meta not found: {wbx_path}")
        return None

    try:
        top = json.loads(top_path.read_text())
        wbx = json.loads(wbx_path.read_text())
    except Exception as e:
        _fail(f"could not parse JSON: {e}")
        return None

    # Both must agree exactly
    if top == wbx:
        _ok("both symlinks resolve to identical content")
    else:
        _fail("symlinks DIVERGE — this is the per-class-AP-all-zero bug source")
        _info(f"top-level thing_classes = {top.get('thing_classes')}")
        _info(f"wildbox thing_classes   = {wbx.get('thing_classes')}")
        return None

    classes = top["thing_classes"]
    mp = {int(k): int(v) for k, v in top["thing_dataset_id_to_contiguous_id"].items()}

    # Must be 6 species for current paper
    if len(classes) == 6:
        _ok(f"6-species mapping: {classes}")
    else:
        _warn(f"expected 6 species (current paper); found {len(classes)}: {classes}")

    # Contiguous-id injectivity
    if len(set(mp.values())) == len(mp):
        _ok(f"contiguous-id map injective ({len(mp)} entries)")
    else:
        _fail(f"contiguous-id map has duplicate values: {mp}")

    # Critical sort rule: contiguous ids must be 0..N-1 sorted by ascending dataset-id
    sorted_ds_ids = sorted(mp.keys())
    sorted_contig = [mp[d] for d in sorted_ds_ids]
    if sorted_contig == list(range(len(mp))):
        _ok("contiguous ids 0..N-1 sorted by ascending dataset_id "
            "(matches register_and_store_model_metadata convention)")
    else:
        _fail(f"contiguous ids NOT sorted-by-dataset-id; "
              f"sorted_ds={sorted_ds_ids} → contig={sorted_contig}")

    # Cross-check: thing_classes[contig_id] must align with mapping
    name_at_contig = {mp[ds_id]: classes[mp[ds_id]] for ds_id in sorted_ds_ids
                      if mp[ds_id] < len(classes)}
    _info(f"contig_id → species: {name_at_contig}")

    # Specifically catch the wildlife6 (giraffe=1000, grevys=1001, ..., gazelle=1005) layout
    expected_wl6 = {1000: "giraffe", 1001: "grevys_zebra", 1002: "elephant",
                    1003: "plains_zebra", 1004: "rhino", 1005: "gazelle"}
    if all(classes[mp[k]] == v for k, v in expected_wl6.items() if k in mp):
        _ok("layout matches the canonical wildlife6 (giraffe=1000, grevys_zebra=1001, "
            "elephant=1002, plains_zebra=1003, rhino=1004, gazelle=1005)")
    else:
        _warn("layout differs from canonical wildlife6 — verify intentional")

    return top


def audit_stats_json(stats_path: Path, meta: dict) -> bool:
    """Every species in category_meta must be in stats.json or training crashes."""
    _section("[2] Omni3D stats.json registration")

    if not stats_path.exists():
        _fail(f"stats.json missing: {stats_path}")
        return False

    try:
        stats = json.loads(stats_path.read_text())
    except Exception as e:
        _fail(f"stats.json unparseable: {e}")
        return False

    species = meta.get("thing_classes", [])
    cat_names = stats.get("category_names", [])

    # Top-level category_names list
    missing_in_names = [s for s in species if s not in cat_names]
    if not missing_in_names:
        _ok(f"category_names contains all {len(species)} species: {species}")
    else:
        _fail(f"category_names missing {missing_in_names}; "
              f"training will crash with 'ValueError: ... is not in list'")

    # Categories list (id→name)
    cats_by_id = {c.get("id"): c.get("name") for c in stats.get("categories", [])}
    expected_ids = set(int(k) for k in meta["thing_dataset_id_to_contiguous_id"].keys())
    missing_in_categories = expected_ids - set(cats_by_id.keys())
    if not missing_in_categories:
        _ok(f"categories list has entries for all dataset ids {sorted(expected_ids)}")
    else:
        _fail(f"categories list missing dataset ids {sorted(missing_in_categories)} "
              f"(run tools/patch_stats_for_wildbox.py)")

    # Stale 'zebra' detection (we split into plains/grevys)
    if "zebra" in cat_names and "grevys_zebra" in cat_names:
        _warn("stale 'zebra' entry coexists with 'grevys_zebra'/'plains_zebra' — "
              "may interfere with 'id 1001 already in use' check during patch")

    return not missing_in_names and not missing_in_categories


def audit_gt_json(gt_path: Path, meta: dict, label: str) -> Optional[dict]:
    """Schema + ordering + dimensions convention sanity for one Omni3D JSON."""
    _section(f"[3.{label}] GT JSON: {gt_path}")

    if not gt_path.exists():
        _fail(f"missing: {gt_path}")
        return None

    try:
        gt = json.loads(gt_path.read_text())
    except Exception as e:
        _fail(f"unparseable: {e}")
        return None

    n_images = len(gt.get("images", []))
    n_anns = len(gt.get("annotations", []))
    cats = gt.get("categories", [])
    _info(f"images={n_images}, annotations={n_anns}, categories={len(cats)}")

    # Categories alignment with category_meta
    expected_ids = set(int(k) for k in meta["thing_dataset_id_to_contiguous_id"].keys())
    cat_ids_in_gt = {int(c["id"]) for c in cats}
    if cat_ids_in_gt == expected_ids:
        _ok(f"categories' id-set matches category_meta: {sorted(expected_ids)}")
    elif cat_ids_in_gt.issubset(expected_ids):
        missing = expected_ids - cat_ids_in_gt
        _warn(f"GT JSON has subset of expected categories; missing in GT: {sorted(missing)}")
    else:
        extra = cat_ids_in_gt - expected_ids
        _fail(f"GT has out-of-taxonomy category ids: {sorted(extra)}")

    # Image schema
    required_img = {"id", "file_path", "K", "width", "height"}
    bad_img = sum(1 for im in gt["images"] if not required_img.issubset(im.keys()))
    if bad_img == 0:
        _ok(f"image schema: all {n_images} images have {sorted(required_img)}")
    else:
        _fail(f"{bad_img} images missing required keys")

    # Sample-check K shape (should be 3x3)
    sample_K = gt["images"][0]["K"] if gt["images"] else None
    if sample_K is not None:
        ok_K = (len(sample_K) == 3 and all(len(r) == 3 for r in sample_K))
        if ok_K:
            _ok(f"K is 3×3 (sampled image[0])")
        else:
            _fail(f"K malformed (expected 3×3): {sample_K}")

    # Annotation schema
    required_ann = {"id", "image_id", "category_id", "bbox", "dimensions",
                    "center_cam", "R_cam"}
    bad_ann = 0
    for a in gt["annotations"]:
        if not required_ann.issubset(a.keys()):
            bad_ann += 1
    if bad_ann == 0:
        _ok(f"annotation schema: all {n_anns} annotations have {sorted(required_ann)}")
    else:
        _fail(f"{bad_ann} annotations missing required keys (one of {sorted(required_ann)})")

    # bbox positivity (xywh, w/h > 0)
    bad_bbox = sum(1 for a in gt["annotations"]
                   if len(a["bbox"]) != 4 or a["bbox"][2] <= 0 or a["bbox"][3] <= 0)
    if bad_bbox == 0:
        _ok("all bbox entries positive xywh")
    else:
        _fail(f"{bad_bbox} bbox entries have non-positive w/h")

    # 3D dimensions positivity
    bad_dims = sum(1 for a in gt["annotations"]
                   if len(a.get("dimensions", [])) != 3
                   or any(d <= 0 for d in a["dimensions"]))
    if bad_dims == 0:
        _ok("all dimensions entries are 3-vectors with positive values")
    else:
        _fail(f"{bad_dims} dimensions malformed or non-positive")

    # category_id values must be in expected set
    bad_cat = sum(1 for a in gt["annotations"]
                  if int(a["category_id"]) not in expected_ids)
    if bad_cat == 0:
        _ok(f"all annotation category_ids ∈ {sorted(expected_ids)}")
    else:
        _fail(f"{bad_cat} annotations have category_id outside taxonomy")

    # Per-class annotation counts (paper-relevant)
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    counts = collections.Counter(cat_id_to_name.get(a["category_id"], "?")
                                 for a in gt["annotations"])
    _info(f"per-class annotation counts: {dict(counts)}")

    # Sample file existence (paths must resolve)
    sample = gt["images"][::max(1, n_images // 20)][:20]
    missing_files = sum(1 for im in sample if not Path(im["file_path"]).exists())
    if missing_files == 0:
        _ok(f"sampled {len(sample)}/{n_images} file_paths resolve on disk")
    else:
        _warn(f"{missing_files}/{len(sample)} sampled file_paths do NOT exist; "
              f"data may have been moved (use tools/remap_wildbox_paths.py)")

    return gt


def audit_train_val_leak(gt_train: dict, gt_val: dict):
    """Video-level disjoint check (the core no-leakage guarantee)."""
    _section("[4] Train/val leakage")

    def vids_of(gt):
        return {im["file_path"].split("/")[-3] for im in gt.get("images", [])}

    train_v = vids_of(gt_train)
    val_v = vids_of(gt_val)
    overlap = train_v & val_v
    if not overlap:
        _ok(f"video-level disjoint: {len(train_v)} train videos, "
            f"{len(val_v)} val videos, 0 shared")
    else:
        _fail(f"video-level LEAKAGE: {len(overlap)} shared: "
              f"{sorted(overlap)[:5]}{' ...' if len(overlap) > 5 else ''}")

    # Image-id space (could overlap if 0-indexed independently — that's fine)
    train_imgs = {im["id"] for im in gt_train.get("images", [])}
    val_imgs = {im["id"] for im in gt_val.get("images", [])}
    if train_imgs & val_imgs:
        _info(f"image_id space overlaps {len(train_imgs & val_imgs)} ids "
              f"(typical when each split is 0-indexed independently; "
              f"verified by video disjointness above)")
    else:
        _ok("image_id disjoint between splits (also fine)")


def audit_oracle(oracle_path: Path, gt_val: dict, meta: dict):
    """Oracle JSON ↔ val image-id coverage + class-id sanity."""
    _section("[5] GDino oracle JSON ↔ val GT")

    if not oracle_path.exists():
        _warn(f"oracle JSON not found: {oracle_path}")
        _info("  (skip — paper-protocol zero-shot row will be missing)")
        return

    try:
        oracle = json.loads(oracle_path.read_text())
    except Exception as e:
        _fail(f"oracle JSON unparseable: {e}")
        return

    n_images = len(oracle)
    n_boxes = sum(len(e.get("instances", [])) for e in oracle)
    _info(f"oracle: {n_images} images, {n_boxes} boxes "
          f"({n_boxes/max(1,n_images):.2f} per image)")

    # Image-id coverage vs val GT
    val_ids = {int(im["id"]) for im in gt_val.get("images", [])}
    oracle_ids = {int(e["image_id"]) for e in oracle}

    val_only = val_ids - oracle_ids
    oracle_only = oracle_ids - val_ids
    if not val_only and not oracle_only:
        _ok(f"image_id coverage perfect: {len(val_ids)} val == {len(oracle_ids)} oracle")
    elif not val_only:
        _warn(f"oracle has {len(oracle_only)} extra images not in val GT "
              f"(stale oracle from older val split?)")
    elif not oracle_only:
        _fail(f"oracle MISSING {len(val_only)} val images "
              f"(re-run tools/precompute_gdino_oracle.py on the current val)")
    else:
        _fail(f"oracle/val image_id mismatch: "
              f"val-only={len(val_only)} oracle-only={len(oracle_only)} — STALE oracle")

    # All oracle category_ids must be in our taxonomy
    expected_ids = set(int(k) for k in meta["thing_dataset_id_to_contiguous_id"].keys())
    bad = collections.Counter()
    for e in oracle:
        for inst in e.get("instances", []):
            cid = int(inst["category_id"])
            if cid not in expected_ids:
                bad[cid] += 1
    if not bad:
        _ok(f"all oracle category_ids ∈ {sorted(expected_ids)}")
    else:
        _fail(f"oracle has out-of-taxonomy category_ids: {dict(bad)}")

    # Per-class count distribution (warn on known biases — Grévy's underprediction)
    counts = collections.Counter(inst.get("category_name", "?")
                                 for e in oracle for inst in e.get("instances", []))
    _info(f"per-class oracle counts: {dict(counts)}")
    if "grevys_zebra" in counts and "plains_zebra" in counts:
        if counts["grevys_zebra"] < counts["plains_zebra"] / 10:
            _warn("Grévy's-zebra count is <10% of plains zebra count — known GDino "
                  "fine-grained-class limitation; affects per-class AP for those rows "
                  "but not class-agnostic localization")

    # Sample box quality: median IoU vs GT (only if shapely happens to be importable;
    # otherwise just sanity-check bbox shapes are 4-vectors with positive w/h).
    bad_bbox = sum(1 for e in oracle for inst in e.get("instances", [])
                   if len(inst.get("bbox", [])) != 4
                   or inst["bbox"][2] <= 0 or inst["bbox"][3] <= 0)
    if bad_bbox == 0:
        _ok(f"all {n_boxes} oracle bbox entries are positive xywh")
    else:
        _fail(f"{bad_bbox} oracle bboxes malformed")


def audit_pretrained(pretrained_path: Path):
    """Detect the iteration-skip-training silent failure pattern at the source.
    The pretrained checkpoint stores 'iteration' from Omni3D pre-training; if the
    training code reads that field unconditionally, fine-tunes silently skip.
    """
    _section("[6] Pretrained checkpoint integrity (iteration-skip-bug source)")

    if not pretrained_path.exists():
        _warn(f"pretrained checkpoint not found: {pretrained_path}; skip")
        return

    try:
        torch = _try_load_torch()
        c = torch.load(str(pretrained_path), weights_only=False, map_location="cpu")
    except Exception as e:
        _warn(f"could not load pretrained checkpoint: {e}")
        return

    iteration = c.get("iteration")
    keys = list(c.keys())[:10]
    _info(f"top-level keys (first 10): {keys}")

    if iteration is None:
        _ok("pretrained has NO 'iteration' field — the iteration-skip-bug cannot trigger")
    else:
        _info(f"pretrained's stored iteration = {iteration}")
        _info("  → train_net.py:183 must respect resume=False to avoid setting "
              f"start_iter to {iteration+1} (and silently skipping training).")
        _info("  Verify commit f894ab6 or later is applied. The belt-and-suspenders "
              "fix is to strip the iteration field from the pretrained checkpoint.")


def audit_predictions(pred_dir: Path, gt_val: dict, meta: dict, oracle: Optional[List[dict]]):
    """Schema + value-range + iteration-skip-detection for one prediction directory."""
    _section(f"[7] Predictions: {pred_dir.name}")

    # Eval-side artifact existence
    log_txt = pred_dir / "log.txt"
    log_rel = pred_dir / "log.rel.txt"
    bev_json = pred_dir / "bev_ap.json"
    nhd_txt = pred_dir / "summary_nhd.txt"
    pred_pth = pred_dir / "inference" / "iter_final" / "WildBox_val" / "instances_predictions.pth"

    for f, name, required in [(log_txt, "log.txt", True),
                              (log_rel, "log.rel.txt", False),
                              (bev_json, "bev_ap.json", True),
                              (nhd_txt, "summary_nhd.txt", True),
                              (pred_pth, "instances_predictions.pth", True)]:
        if f.exists():
            _ok(f"{name} present")
        elif required:
            _fail(f"{name} MISSING — eval pipeline likely incomplete")
        else:
            _warn(f"{name} missing (optional; Rel-AP3D may have been --skip-rel-ap3d'd)")

    if not pred_pth.exists():
        return

    # iteration-skip-training detection (only if we can find the training log too)
    train_dir = pred_dir.parent if pred_dir.name == "eval" else pred_dir
    train_log = train_dir / "log.txt"
    train_model = train_dir / "model_final.pth"
    if train_log.exists() and train_model.exists():
        text = train_log.read_text(errors="ignore")
        m = re.search(r"Starting training from iteration (\d+)\s*\((resume=(?:True|False))\)", text)
        if m:
            iter_n = int(m.group(1))
            if iter_n == 0:
                _ok(f"training started from iteration 0 ({m.group(2)})")
            elif iter_n > 50000:
                _fail(f"training started from iteration {iter_n} — the iteration-skip "
                      f"bug fired (model_final.pth is byte-identical to pretrained, "
                      f"NO training happened). git pull to commit f894ab6 or later.")
            else:
                _info(f"training started from iteration {iter_n} (resume from intermediate ckpt)")
        n_iter_lines = len(re.findall(r"iter:\s*\d+", text))
        if n_iter_lines == 0:
            _fail("zero 'iter: N' lines in training log → training was silently skipped")
        else:
            _info(f"{n_iter_lines} iteration log lines (sanity: training did run)")

    # Predictions
    torch = _try_load_torch()
    preds = torch.load(str(pred_pth), weights_only=False, map_location="cpu")
    n_imgs = len(preds)
    n_total = sum(len(p.get("instances", [])) for p in preds)
    _info(f"prediction images={n_imgs}, total instances={n_total} "
          f"({n_total/max(1,n_imgs):.2f} avg per image)")

    # Schema
    required_keys = {"image_id", "category_id", "bbox", "score",
                     "center_cam", "dimensions", "pose"}
    missing = sum(1 for p in preds for inst in p.get("instances", [])
                  if not required_keys.issubset(inst.keys()))
    if missing == 0:
        _ok("all prediction instances have required keys")
    else:
        _fail(f"{missing} prediction instances missing one of {sorted(required_keys)}")

    # category_id space — must be either contiguous {0..N-1} or dataset {1000..1005}
    contig_ids = set(meta["thing_dataset_id_to_contiguous_id"].values())
    ds_ids = set(int(k) for k in meta["thing_dataset_id_to_contiguous_id"].keys())
    pred_cats = collections.Counter(int(inst["category_id"])
                                    for p in preds for inst in p.get("instances", []))
    if all(c in contig_ids for c in pred_cats):
        _ok(f"prediction category_ids in contiguous space {sorted(contig_ids)} "
            f"(typical fine-tuned output)")
    elif all(c in ds_ids for c in pred_cats):
        _ok(f"prediction category_ids in dataset space {sorted(ds_ids)} "
            f"(typical ORACLE2D=True output)")
    else:
        _fail(f"prediction category_ids in UNKNOWN space (partial overlap with both): "
              f"{sorted(pred_cats)[:10]}")
    _info(f"per-class prediction counts: {dict(sorted(pred_cats.items()))}")

    # Score distribution sanity
    scores = [float(inst["score"]) for p in preds for inst in p.get("instances", [])]
    if scores:
        smin, smax = min(scores), max(scores)
        smed = sorted(scores)[len(scores)//2]
        n_unique = len(set(round(s, 5) for s in scores))
        _info(f"score range: min={smin:.4f} median={smed:.4f} max={smax:.4f} "
              f"unique≈{n_unique}")
        if n_unique == 1:
            _warn("all predictions have the same score; per-class AP is order-degenerate")
        elif 0.0 <= smin and smax <= 1.0:
            _ok(f"scores well-formed in [0,1]")
        else:
            _warn(f"scores out of [0,1] range: [{smin}, {smax}]")

    # bbox positivity
    bad_bbox = sum(1 for p in preds for inst in p.get("instances", [])
                   if len(inst["bbox"]) != 4
                   or inst["bbox"][2] <= 0 or inst["bbox"][3] <= 0
                   or inst["bbox"][0] < -1 or inst["bbox"][1] < -1)
    if bad_bbox == 0:
        _ok(f"all {n_total} bbox entries positive xywh")
    else:
        _fail(f"{bad_bbox} bbox entries malformed")

    # bbox3D 8x3 finite (if present)
    bad_3d = 0
    n_with_3d = 0
    for p in preds:
        for inst in p.get("instances", []):
            corners = inst.get("bbox3D_cam") or inst.get("bbox3D")
            if corners is None:
                continue
            n_with_3d += 1
            try:
                if len(corners) != 8 or any(len(c) != 3 for c in corners):
                    bad_3d += 1; continue
                if any(not math.isfinite(v) for c in corners for v in c):
                    bad_3d += 1
            except (TypeError, ValueError):
                bad_3d += 1
    if n_with_3d > 0:
        if bad_3d == 0:
            _ok(f"all {n_with_3d} bbox3D entries are 8×3 with finite values")
        else:
            _fail(f"{bad_3d}/{n_with_3d} bbox3D entries malformed or non-finite")
    else:
        _warn("no predictions carry bbox3D / bbox3D_cam (3D head may have failed)")

    # dimensions positivity
    bad_dims = sum(1 for p in preds for inst in p.get("instances", [])
                   if any(d <= 0 for d in inst.get("dimensions", [1, 1, 1])))
    if bad_dims == 0:
        _ok("all dimensions positive (no degenerate cuboids)")
    else:
        _fail(f"{bad_dims} predictions have non-positive dimensions")

    # pose orthogonality on sample
    n_check = n_pass = 0
    for p in preds[:50]:
        for inst in p.get("instances", []):
            R = inst.get("pose")
            if R is None or len(R) != 3:
                continue
            n_check += 1
            try:
                rt = [[sum(R[k][i] * R[k][j] for k in range(3)) for j in range(3)]
                      for i in range(3)]
                err = sum(abs(rt[i][j] - (1.0 if i == j else 0.0))
                          for i in range(3) for j in range(3))
                if err < 0.05:
                    n_pass += 1
            except (TypeError, ValueError):
                pass
    if n_check > 0:
        if n_pass / n_check >= 0.9:
            _ok(f"pose orthogonal: {n_pass}/{n_check} sampled R^T R within 0.05 of I")
        else:
            _warn(f"only {n_pass}/{n_check} sampled poses orthogonal — verify rotation decoding")

    # Eval-log sanity: per-class AP table populated for all 6 species (catches
    # the symlink bug recurrence)
    if log_txt.exists():
        text = log_txt.read_text(errors="ignore")
        # Per-category bbox AP table (last occurrence)
        m = re.search(r"Per-category bbox AP/AR in 2D mode:\s*\n((?:\|[^\n]*\n)+)",
                      text[::-1][::-1])  # match anywhere
        # Easier: re-find all blocks, take last
        blocks = re.findall(r"Per-category bbox AP/AR in 2D mode:\s*\n((?:\|[^\n]*\n)+)", text)
        if blocks:
            body = blocks[-1]
            species_with_ap = []
            for line in body.splitlines():
                m2 = re.match(r"\|\s*(\w[\w ]*?)\s*\(AP\)\s*\|\s*([-+0-9.]+)", line)
                if m2:
                    species_with_ap.append((m2.group(1).strip(), float(m2.group(2))))
            if species_with_ap:
                n_zero = sum(1 for _, v in species_with_ap if v < 0.001)
                n_total = len(species_with_ap)
                expected_n = len(meta["thing_classes"])
                if n_total != expected_n:
                    _warn(f"per-class AP table has {n_total} rows; expected {expected_n}")
                if n_zero >= n_total - 1 and n_total > 1:
                    _fail(f"per-class 2D AP all 0 except {n_total-n_zero} class — "
                          f"the symlink/category_meta bug recurred. Re-check section [1].")
                elif n_zero == 0:
                    _ok(f"all {n_total} species have non-zero 2D AP — no symlink bug")
                else:
                    _info(f"{n_zero}/{n_total} species have ~0 2D AP "
                          f"(low-data class or true model failure)")
                _info(f"per-class 2D AP: {dict(species_with_ap)}")
        else:
            _warn("no 'Per-category bbox AP/AR in 2D mode' block in log.txt")

    # Rel-AP3D best-scale boundary check
    if log_rel.exists():
        text = log_rel.read_text(errors="ignore")
        scales = re.findall(r"\[rel_ap3d\]\s*best global scale\s*=\s*([-+0-9.]+)", text)
        m = re.search(r"\[rel_ap3d\]\s*best global scale\s*=\s*([-+0-9.]+)", text)
        if m:
            best = float(m.group(1))
            _info(f"Rel-AP3D best global scale = {best:.4f}")
            # Expected near 1.0 for fine-tuned, near 0.05-0.5 for zero-shot
            if 0.85 <= best <= 1.20:
                _ok(f"best-scale {best:.4f} ≈ 1.0 (model in correct VGGT scale; fine-tuned)")
            elif 0.05 <= best <= 0.50:
                _ok(f"best-scale {best:.4f} ∈ [0.05, 0.50] (zero-shot — predictions in "
                    f"metric scale, downscaled to match VGGT)")
            elif best < 0.06 or best > 2.95:
                _warn(f"best-scale {best:.4f} near grid boundary; widen "
                      f"REL_AP3D_SEARCH and re-run check_rel_ap3d_boundary.py")
            else:
                _info(f"best-scale {best:.4f} interior, neither 1.0 nor 0.05-0.5 range "
                      f"(unusual; investigate)")

    # NHD components from summary_nhd.txt
    if nhd_txt.exists():
        text = nhd_txt.read_text(errors="ignore")
        m = re.search(r"NHD-z.*?([-+0-9.]+)", text)
        if m:
            nhd_z = float(m.group(1))
            _info(f"NHD-z (depth error) = {nhd_z:.2f} "
                  f"({'fine-tuned-quality' if nhd_z < 50 else 'zero-shot-quality (>>50, scale-mismatch)'})")

    # BEV per-class population
    if bev_json.exists():
        try:
            bev = json.loads(bev_json.read_text())
            for iou_str in ("0.5", "0.25"):
                iou_block = bev.get(iou_str, {})
                pc = iou_block.get("ap_per_class", {})
                if pc:
                    n_zero = sum(1 for v in pc.values() if v < 0.001)
                    if n_zero == len(pc) and iou_block.get("ap_micro", 0) > 0:
                        _fail(f"BEV@{iou_str}: per-class all 0 but micro > 0 — "
                              f"dataset_id↔contiguous_id normalization missing")
                    else:
                        _ok(f"BEV@{iou_str}: per-class populated, "
                            f"micro={iou_block.get('ap_micro',0):.2f}, "
                            f"macro={iou_block.get('ap_macro',0):.2f}")
        except Exception as e:
            _warn(f"bev_ap.json unparseable: {e}")


def audit_multiseed(multiseed_dir: Path):
    """Multi-seed completeness + cross-seed sanity."""
    _section(f"[8] Multi-seed consistency: {multiseed_dir}")

    if not multiseed_dir.exists():
        _warn(f"directory does not exist: {multiseed_dir}; skip")
        return

    seed_dirs = sorted([p for p in multiseed_dir.iterdir() if p.is_dir()
                        and re.match(r"seed\d+", p.name)])
    if not seed_dirs:
        _warn(f"no seedN/ subdirectories under {multiseed_dir}; skip")
        return

    _info(f"found seeds: {[p.name for p in seed_dirs]}")

    # Per-seed required artifacts
    for sd in seed_dirs:
        model_final = sd / "model_final.pth"
        eval_log = sd / "eval" / "log.txt"
        eval_bev = sd / "eval" / "bev_ap.json"
        eval_pred = sd / "eval" / "inference" / "iter_final" / "WildBox_val" / "instances_predictions.pth"
        ok = all(p.exists() for p in [model_final, eval_log, eval_bev, eval_pred])
        if ok:
            _ok(f"{sd.name}: model_final + eval/log + eval/bev + eval/predictions all present")
        else:
            missing = [p.name for p in [model_final, eval_log, eval_bev, eval_pred]
                       if not p.exists()]
            _fail(f"{sd.name}: missing {missing}")

    # Cross-seed model_final sizes (training output should differ across seeds even
    # by a few bytes due to optimizer state; warn if all are byte-identical)
    sizes = {}
    for sd in seed_dirs:
        mf = sd / "model_final.pth"
        if mf.exists():
            sizes[sd.name] = mf.stat().st_size
    if len(set(sizes.values())) > 1:
        _ok(f"model_final.pth sizes differ across seeds: {sizes}")
    elif len(sizes) > 1:
        _info(f"all model_final.pth sizes identical: {next(iter(sizes.values()))} "
              f"(could be coincidence — checkpoints contain only weights, no random state)")

    # If all seeds finished, scrape per-class 3D AP and report std
    per_class_3d = collections.defaultdict(list)
    for sd in seed_dirs:
        log = sd / "eval" / "log.txt"
        if not log.exists():
            continue
        text = log.read_text(errors="ignore")
        blocks = re.findall(r"Per-category bbox AP/AR in 3D mode:\s*\n((?:\|[^\n]*\n)+)", text)
        if blocks:
            for line in blocks[-1].splitlines():
                m = re.match(r"\|\s*(\w[\w ]*?)\s*\(AP\)\s*\|\s*([-+0-9.]+)", line)
                if m:
                    per_class_3d[m.group(1).strip()].append(float(m.group(2)))

    if per_class_3d:
        _info("per-class 3D AP across seeds (mean ± std):")
        for cls, vals in sorted(per_class_3d.items()):
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
                rel_std = std / max(mean, 1e-6)
                marker = " (HIGH variance)" if rel_std > 0.5 else ""
                _info(f"  {cls:14s} {mean:6.2f} ± {std:5.2f}  ({len(vals)} seeds){marker}")
            else:
                _info(f"  {cls:14s} {mean:6.2f}  ({len(vals)} seed)")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-train", type=Path,
                    default=Path("datasets/Omni3D/WildBox_train.json"))
    ap.add_argument("--gt-val", type=Path,
                    default=Path("datasets/Omni3D/WildBox_val.json"))
    ap.add_argument("--stats", type=Path,
                    default=Path("datasets/Omni3D/stats.json"))
    ap.add_argument("--oracle", type=Path,
                    default=Path("datasets/Omni3D/gdino_WildBox_val_oracle_2d.json"))
    ap.add_argument("--category-meta-top", type=Path,
                    default=Path("configs/category_meta.json"),
                    help="Top-level category_meta symlink (read by Omni3D evaluator).")
    ap.add_argument("--category-meta-wbx", type=Path,
                    default=Path("configs/wildbox/category_meta.json"),
                    help="Wildbox-dir category_meta symlink (read by train_net.py --eval-only).")
    ap.add_argument("--pretrained", type=Path,
                    default=Path("checkpoints/ovmono3d_lift.pth"),
                    help="Pretrained checkpoint to inspect for the iteration-skip-bug.")
    ap.add_argument("--pred-dir", type=Path, action="append", default=[],
                    help="Per-row prediction directory under output/. Repeatable.")
    ap.add_argument("--multiseed-dir", type=Path, default=None,
                    help="Multi-seed parent dir (e.g. output/wl6_init5sp_multiseed).")
    args = ap.parse_args(argv)

    print("=" * 72)
    print("OVMono3D × WildBox — methodology audit")
    print("=" * 72)

    meta = audit_category_meta(args.category_meta_top, args.category_meta_wbx)
    if meta is None:
        print("\nFATAL: category_meta is broken; downstream checks would be misleading.")
        return 2

    audit_stats_json(args.stats, meta)

    gt_val = audit_gt_json(args.gt_val, meta, "val")
    gt_train = audit_gt_json(args.gt_train, meta, "train")

    if gt_train and gt_val:
        audit_train_val_leak(gt_train, gt_val)

    if gt_val:
        audit_oracle(args.oracle, gt_val, meta)

    audit_pretrained(args.pretrained)

    oracle_data = None
    if args.oracle.exists():
        try:
            oracle_data = json.loads(args.oracle.read_text())
        except Exception:
            pass

    for pd in args.pred_dir:
        audit_predictions(pd, gt_val or {}, meta, oracle_data)

    if args.multiseed_dir:
        audit_multiseed(args.multiseed_dir)

    print()
    print("=" * 72)
    print("Audit done. Review FAIL and WARN lines before paper submission.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())

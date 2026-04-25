#!/usr/bin/env python
"""Single-sheet paper-results assembler for the WildBox 3D wildlife
detection experiments.

Reads all the eval dirs we produced — multi-seed fine-tuned, single-seed
ablation, zero-shot RPN-transfer, zero-shot GDino oracle — and emits ONE
markdown file with every number a paper experiments/benchmarks section
needs. Plus LaTeX exports of the headline tables and a single JSON with
all values for downstream tooling.

Sections in the output:
  1. Dataset facts (embedded from dataset_stats.md if present)
  2. Main metrics table (standard 2D/3D + Rel-AP3D + BEV)
  3. Per-class 2D AP (with mean±std for multi-seed rows)
  4. Per-class 3D AP (with mean±std for multi-seed rows)
  5. Per-class BEV AP (primary IoU 0.50, supplementary 0.25)
  6. Disentangled NHD components (motivates BEV as primary metric)
  7. Class-agnostic 2D AP (zero-shot diagnostic)
  8. Reproducibility (commit, configs, dataset version)
  9. Per-run raw numbers (appendix)

Usage:
    python tools/assemble_paper_results.py \\
        --finetuned-seeds output/wl6_rt0.5_multiseed/seed0/eval \\
                          output/wl6_rt0.5_multiseed/seed1/eval \\
                          output/wl6_rt0.5_multiseed/seed2/eval \\
        --ablation-rt035  output/wl6_rt0.35_seed0/seed0/eval \\
        --zeroshot-rpn    output/wl6_zeroshot_rpn \\
        --zeroshot-oracle output/wl6_zeroshot_oracle2d \\
        --dataset-stats   datasets/Omni3D/dataset_stats/dataset_stats.md \\
        --classes         giraffe grevys_zebra elephant plains_zebra rhino gazelle \\
        --rare-classes    giraffe gazelle grevys_zebra \\
        --out             output/paper_results

Any of --ablation-rt035 / --zeroshot-rpn / --zeroshot-oracle /
--dataset-stats can be omitted; the corresponding sections are skipped.
"""
import argparse
import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------
# Shared parsers (kept self-contained — no import from make_report.py)
# ---------------------------------------------------------------

PAIR_RE = re.compile(r"(\w[\w ]*?)\s*\((AP|AR)\)\s*\|\s*([-+0-9.]+)")


def _parse_per_class_blocks(text: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Returns {mode: {class_name: {AP: v, AR: v}}}."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m in re.finditer(
        r"Per-category bbox AP/AR in (\S+) mode:\s*\n((?:\|[^\n]*\n)+)", text
    ):
        mode = m.group(1)
        body = m.group(2)
        cls_dict: Dict[str, Dict[str, float]] = {}
        for pair in PAIR_RE.finditer(body):
            cname, metric, val = pair.groups()
            try:
                cls_dict.setdefault(cname.strip(), {})[metric] = float(val)
            except ValueError:
                pass
        out[mode] = cls_dict
    return out


def _parse_summary_blocks(text: str) -> Dict[str, Dict[str, float]]:
    """Pulls the 'Evaluation results for bbox in <MODE> mode:' table values."""
    out: Dict[str, Dict[str, float]] = {}
    blocks: Dict[str, Tuple[str, str]] = {}
    for m in re.finditer(
        r"Evaluation results for bbox in (\S+) mode:\s*\n"
        r"\|([^\n]*)\|\s*\n"
        r"\|([^\n]*)\|\s*\n"
        r"\|([^\n]*)\|",
        text,
        re.MULTILINE,
    ):
        mode = m.group(1)
        header = m.group(2)
        values = m.group(4)
        blocks[mode] = (header, values)
    for mode, (header, values) in blocks.items():
        hdrs = [h.strip() for h in header.split("|") if h.strip()]
        vals = [v.strip() for v in values.split("|") if v.strip()]
        out[mode] = {}
        for h, v in zip(hdrs, vals):
            try:
                out[mode][h] = float(v)
            except ValueError:
                pass
    return out


def _parse_summary_nhd(text: str) -> Dict[str, Any]:
    """Pull class-agnostic 2D AP, NHD best-scale, NHD-z, mean-NHD@best from
    summary_nhd.txt (output of class_agnostic_eval.py)."""
    out: Dict[str, Any] = {}
    for ap_thr in ("0.25", "0.50", "0.75"):
        m = re.search(rf"AP@{re.escape(ap_thr)}\s*=\s*([-+0-9.]+)", text)
        if m:
            out[f"ca_2d_ap_{ap_thr}"] = float(m.group(1))
    m = re.search(r"macro AP@0\.50\s*=\s*([-+0-9.]+)", text)
    if m:
        out["ca_macro_2d_ap_0.50"] = float(m.group(1))
    m = re.search(r"best global scale:\s*([-+0-9.]+)", text)
    if m:
        out["nhd_best_scale"] = float(m.group(1))
    m = re.search(r"mean NHD @ best s:\s*([-+0-9.]+)", text)
    if m:
        out["mean_nhd_best_s"] = float(m.group(1))
    m = re.search(r"mean NHD @ s=1:\s*([-+0-9.]+)", text)
    if m:
        out["mean_nhd_at_s1"] = float(m.group(1))
    return out


def _parse_disent_nhd(text: str) -> Dict[str, float]:
    """From log.txt or log.rel.txt — extracts the 4-component NHD breakdown."""
    out: Dict[str, float] = {}
    pat = {
        "overall_NHD": r"overall:\s*([-+0-9.]+)",
        "disent_xy_NHD": r"xy:\s*([-+0-9.]+)",
        "disent_z_NHD": r"z:\s*([-+0-9.]+)",
        "disent_dimensions_NHD": r"dimensions:\s*([-+0-9.]+)",
        "disent_pose_NHD": r"pose:\s*([-+0-9.]+)",
    }
    for key, p in pat.items():
        m = re.search(p, text)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def gather_run(run_dir: Path) -> Dict[str, Any]:
    """Read everything we know about a single eval dir."""
    log = (run_dir / "log.txt")
    log_rel = (run_dir / "log.rel.txt")
    bev = (run_dir / "bev_ap.json")
    nhd = (run_dir / "summary_nhd.txt")
    config = (run_dir / "config.yaml")

    text = (log.read_text(errors="ignore") if log.exists() else "") + "\n" + \
           (log_rel.read_text(errors="ignore") if log_rel.exists() else "")
    nhd_text = nhd.read_text(errors="ignore") if nhd.exists() else ""

    out = {
        "run_dir": str(run_dir),
        "summary": _parse_summary_blocks(text),
        "per_class": _parse_per_class_blocks(text),
        "ca_nhd": _parse_summary_nhd(nhd_text),
        "disent_nhd": _parse_disent_nhd(text),
        "bev": json.load(open(bev)) if bev.exists() else {},
        "config_path": str(config) if config.exists() else None,
    }
    return out


def normalize_single_run(r: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap raw scalar values from a single-seed run as (value, 0.0) tuples
    so the renderer can treat single-seed and multi-seed rows uniformly.

    Also re-keys the raw BEV JSON ('ap_micro', 'ap_macro', 'ap_per_class')
    to the post-aggregation shape ('micro', 'macro', 'per_class').

    Returns a NEW dict; doesn't mutate input.
    """
    def to_tup(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return (float("nan"), 0.0)
        if isinstance(v, (int, float)):
            return (float(v), 0.0)
        return v  # already a tuple, leave alone

    out = dict(r)
    out["summary"] = {
        mode: {k: to_tup(v) for k, v in (r.get("summary", {}).get(mode) or {}).items()}
        for mode in (r.get("summary") or {})
    }
    out["per_class"] = {
        mode: {
            cname: {mk: to_tup(mv) for mk, mv in (r.get("per_class", {}).get(mode, {}).get(cname) or {}).items()}
            for cname in (r.get("per_class", {}).get(mode) or {})
        }
        for mode in (r.get("per_class") or {})
    }
    out["disent_nhd"] = {k: to_tup(v) for k, v in (r.get("disent_nhd") or {}).items()}
    out["ca_nhd"] = {k: to_tup(v) for k, v in (r.get("ca_nhd") or {}).items()}

    # BEV: raw json uses ap_micro / ap_macro / ap_per_class with float values.
    # Aggregator uses micro / macro / per_class with (mean, std) tuples.
    # Translate raw -> aggregator shape so the renderer can treat both the same.
    bev_in = r.get("bev") or {}
    bev_out: Dict[str, Any] = {}
    for k, v in bev_in.items():
        # Only iterate the per-IoU dicts (skip top-level scalar keys like 'iou_thresholds')
        if not isinstance(v, dict):
            continue
        if "ap_per_class" not in v and "ap_micro" not in v:
            # Already in aggregator shape (or unknown); pass through
            bev_out[k] = v
            continue
        bev_out[k] = {
            "micro": to_tup(v.get("ap_micro")),
            "macro": to_tup(v.get("ap_macro")),
            "per_class": {c: to_tup(p) for c, p in (v.get("ap_per_class") or {}).items()},
        }
    out["bev"] = bev_out
    return out


# ---------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------

def _meanstd(xs: List[float]) -> Tuple[float, float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    m = statistics.mean(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def aggregate_runs(runs: List[Dict[str, Any]], mode_name: str = "fine-tuned (multi-seed)"
                  ) -> Dict[str, Any]:
    """Produce mean±std views across N runs. Each run was parsed by gather_run."""
    if not runs:
        return {"label": mode_name, "n_seeds": 0}
    n = len(runs)

    # 2D / 3D summary
    summary_agg: Dict[str, Dict[str, Tuple[float, float]]] = {}
    modes = set()
    for r in runs:
        modes.update(r["summary"].keys())
    for mode in modes:
        keys = set()
        for r in runs:
            keys.update(r["summary"].get(mode, {}).keys())
        per_key: Dict[str, Tuple[float, float]] = {}
        for k in keys:
            vals = [r["summary"].get(mode, {}).get(k) for r in runs]
            per_key[k] = _meanstd([v for v in vals if v is not None])
        summary_agg[mode] = per_key

    # per-class
    per_class_agg: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    pc_modes = set()
    for r in runs:
        pc_modes.update(r["per_class"].keys())
    for mode in pc_modes:
        cls_set = set()
        for r in runs:
            cls_set.update(r["per_class"].get(mode, {}).keys())
        per_class_agg[mode] = {}
        for cname in cls_set:
            metric_keys = set()
            for r in runs:
                metric_keys.update(r["per_class"].get(mode, {}).get(cname, {}).keys())
            per_class_agg[mode][cname] = {}
            for mk in metric_keys:
                vals = [r["per_class"].get(mode, {}).get(cname, {}).get(mk) for r in runs]
                per_class_agg[mode][cname][mk] = _meanstd([v for v in vals if v is not None])

    # BEV
    bev_agg: Dict[str, Any] = {}
    iou_keys = set()
    for r in runs:
        iou_keys.update((r.get("bev", {}).get("ap_per_class") or {}).keys())
    for iou in iou_keys:
        bev_agg[iou] = {"per_class": {}, "micro": _meanstd(
            [r.get("bev", {}).get("ap_micro", {}).get(iou) for r in runs]
        ), "macro": _meanstd(
            [r.get("bev", {}).get("ap_macro", {}).get(iou) for r in runs]
        )}
        cls_set = set()
        for r in runs:
            cls_set.update((r.get("bev", {}).get("ap_per_class", {}) or {}).get(iou, {}).keys())
        for c in cls_set:
            vals = [(r.get("bev", {}).get("ap_per_class", {}) or {}).get(iou, {}).get(c) for r in runs]
            bev_agg[iou]["per_class"][c] = _meanstd([v for v in vals if v is not None])

    # NHD components + class-agnostic
    disent_agg = {}
    for k in ("overall_NHD", "disent_xy_NHD", "disent_z_NHD",
              "disent_dimensions_NHD", "disent_pose_NHD"):
        vals = [r["disent_nhd"].get(k) for r in runs]
        disent_agg[k] = _meanstd([v for v in vals if v is not None])

    ca_agg = {}
    for k in ("ca_2d_ap_0.25", "ca_2d_ap_0.50", "ca_2d_ap_0.75",
              "ca_macro_2d_ap_0.50", "nhd_best_scale", "mean_nhd_best_s"):
        vals = [r["ca_nhd"].get(k) for r in runs]
        ca_agg[k] = _meanstd([v for v in vals if v is not None])

    return {
        "label": mode_name, "n_seeds": n,
        "run_dirs": [r["run_dir"] for r in runs],
        "summary": summary_agg, "per_class": per_class_agg,
        "bev": bev_agg, "disent_nhd": disent_agg, "ca_nhd": ca_agg,
    }


# ---------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------

def _fmt(mean: float, std: float, force_std: bool = False) -> str:
    if math.isnan(mean):
        return "—"
    if std == 0.0 and not force_std:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def _as_tuple(v) -> Tuple[float, float]:
    """Coerce a value into (mean, std). Accepts already-tuple, raw scalar,
    or None/NaN. Single defensive helper that the whole renderer uses."""
    if v is None:
        return (float("nan"), 0.0)
    if isinstance(v, tuple) and len(v) == 2:
        return (float(v[0]) if v[0] is not None else float("nan"),
                float(v[1]) if v[1] is not None else 0.0)
    if isinstance(v, (int, float)):
        return (float(v), 0.0)
    return (float("nan"), 0.0)


def _cell(getter, agg: Dict[str, Any]) -> str:
    try:
        v = getter(agg)
        m, s = _as_tuple(v)
        if math.isnan(m):
            return "—"
        return _fmt(m, s)
    except (KeyError, TypeError, ValueError):
        return "—"


def _bev_pc(agg: Dict[str, Any], iou: str, cls: str) -> Tuple[float, float]:
    return agg["bev"][iou]["per_class"][cls]


# ---------------------------------------------------------------
# Markdown / LaTeX builders
# ---------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "(not a git repo)"


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip())
    except Exception:
        return False


def render_markdown(rows: List[Dict[str, Any]],
                    classes: List[str], rare_classes: List[str],
                    dataset_stats_md: Optional[str], gt_facts: Dict[str, Any]) -> str:
    out = []
    from datetime import date
    out.append(f"# WildBox Paper Results — assembled {date.today().isoformat()}\n")
    out.append(f"git: `{_git_commit()}`{' (DIRTY)' if _git_dirty() else ''}  \n")
    out.append(f"GT: `{gt_facts.get('path', 'n/a')}` "
               f"({gt_facts.get('n_images', '?')} images, "
               f"{gt_facts.get('n_anns', '?')} annotations, "
               f"{gt_facts.get('n_videos', '?')} videos)\n\n")

    # ----- 1) Dataset facts
    if dataset_stats_md:
        out.append("## 1. Dataset facts\n")
        out.append("> Auto-generated from `tools/dataset_stats.py` — regenerate any time the prep changes.\n\n")
        # Embed the dataset_stats.md but strip its top-level heading
        embed = re.sub(r"^# .*\n", "", dataset_stats_md, count=1)
        out.append(embed)
        out.append("\n")

    # ----- 2) Main metrics table
    out.append("## 2. Main metrics table (paper Table 1)\n")
    out.append("> Each row is one experimental condition. For multi-seed conditions, cells are mean ± sample-std across N seeds.\n\n")

    # micro AP50 (2D) and overall 3D AP and Rel-AP3D and BEV@0.5/0.25 micro
    headers = ["method", "n_seeds", "2D AP@0.5", "2D AP@0.5:0.95", "3D AP",
               "Rel-AP3D", "BEV AP@0.5", "BEV AP@0.25",
               "ca-2D AP@0.5", "best-scale"]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells = [r["label"], str(r.get("n_seeds", 1))]
        cells.append(_cell(lambda a: a["summary"].get("2D", {}).get("AP50", (float("nan"), 0.0)), r))
        cells.append(_cell(lambda a: a["summary"].get("2D", {}).get("AP", (float("nan"), 0.0)), r))
        cells.append(_cell(lambda a: a["summary"].get("3D", {}).get("AP", (float("nan"), 0.0)), r))
        cells.append(_cell(lambda a: a["summary"].get("3D-Rel", {}).get("AP", (float("nan"), 0.0)), r))
        cells.append(_cell(lambda a: a["bev"].get("0.5", {}).get("micro", (float("nan"), 0.0)), r))
        cells.append(_cell(lambda a: a["bev"].get("0.25", {}).get("micro", (float("nan"), 0.0)), r))
        cells.append(_cell(lambda a: a["ca_nhd"].get("ca_2d_ap_0.50", (float("nan"), 0.0)), r))
        cells.append(_cell(lambda a: a["ca_nhd"].get("nhd_best_scale", (float("nan"), 0.0)), r))
        out.append("| " + " | ".join(cells) + " |")
    out.append("\nNotes: BEV AP @ 0.50 is our primary 3D metric. Rel-AP3D is the LabelAny3D scale-aligned 3D AP. "
               "ca-2D AP is class-agnostic (ignores label correctness — measures pure localization).\n\n")

    # ----- 3) Per-class 2D AP
    out.append("## 3. Per-class 2D AP (COCO 0.50:0.95 mean)\n")
    out.append("| method | " + " | ".join(classes) + " | macro |")
    out.append("|" + "|".join(["---"] * (len(classes) + 2)) + "|")
    for r in rows:
        row = [r["label"]]
        per_cls = []
        for c in classes:
            m, s = _as_tuple(r.get("per_class", {}).get("2D", {}).get(c, {}).get("AP"))
            row.append(_fmt(m, s))
            if not math.isnan(m):
                per_cls.append(m)
        macro = sum(per_cls) / len(per_cls) if per_cls else float("nan")
        row.append(f"{macro:.2f}" if not math.isnan(macro) else "—")
        out.append("| " + " | ".join(row) + " |")
    out.append("\n")

    # ----- 4) Per-class 3D AP
    out.append("## 4. Per-class 3D AP (0.05:0.50 mean) — flag rare classes\n")
    out.append("Rare classes (≤5 train videos, wide single-seed variance): " + ", ".join(rare_classes) + "\n\n")
    out.append("| method | " + " | ".join(classes) + " | macro |")
    out.append("|" + "|".join(["---"] * (len(classes) + 2)) + "|")
    for r in rows:
        row = [r["label"]]
        per_cls = []
        for c in classes:
            m, s = _as_tuple(r.get("per_class", {}).get("3D", {}).get(c, {}).get("AP"))
            label = "**" if c in rare_classes else ""
            row.append(label + _fmt(m, s) + label)
            if not math.isnan(m):
                per_cls.append(m)
        macro = sum(per_cls) / len(per_cls) if per_cls else float("nan")
        row.append(f"{macro:.2f}" if not math.isnan(macro) else "—")
        out.append("| " + " | ".join(row) + " |")
    out.append("\n**bold** = rare class (multi-seed mean±std on these is the meaningful number; single-seed values are unreliable).\n\n")

    # ----- 5) Per-class BEV AP
    out.append("## 5. Per-class BEV AP (primary 3D metric)\n")
    for iou in ("0.5", "0.25"):
        out.append(f"\n### BEV AP @ IoU {iou}\n")
        out.append("| method | " + " | ".join(classes) + " | micro | macro |")
        out.append("|" + "|".join(["---"] * (len(classes) + 3)) + "|")
        for r in rows:
            row = [r["label"]]
            bev_at_iou = r.get("bev", {}).get(iou, {}) or {}
            for c in classes:
                m, s = _as_tuple(bev_at_iou.get("per_class", {}).get(c))
                row.append(_fmt(m, s))
            row.append(_fmt(*_as_tuple(bev_at_iou.get("micro"))))
            row.append(_fmt(*_as_tuple(bev_at_iou.get("macro"))))
            out.append("| " + " | ".join(row) + " |")
    out.append("\n")

    # ----- 6) Disentangled NHD
    out.append("## 6. Disentangled NHD components (motivates BEV as primary)\n")
    out.append("> z-component dominates → depth is the primary 3D error source → BEV (which projects out z) is a fairer headline metric than raw 3D AP for our domain.\n\n")
    out.append("| method | overall | xy | z | dims | pose |")
    out.append("|" + "|".join(["---"] * 6) + "|")
    for r in rows:
        row = [r["label"]]
        nhd = r.get("disent_nhd", {}) or {}
        for k in ("overall_NHD", "disent_xy_NHD", "disent_z_NHD",
                  "disent_dimensions_NHD", "disent_pose_NHD"):
            row.append(_fmt(*_as_tuple(nhd.get(k))))
        out.append("| " + " | ".join(row) + " |")
    out.append("\n")

    # ----- 7) Class-agnostic 2D AP
    out.append("## 7. Class-agnostic 2D AP (zero-shot localization diagnostic)\n")
    out.append("> Ignores class labels — measures pure 'did any prediction overlap any GT box'. The relevant signal when standard per-class AP is 0 (closed-vocab zero-shot, scale-mismatched 3D, etc.).\n\n")
    out.append("| method | AP@0.25 | AP@0.50 | AP@0.75 | macro@0.50 |")
    out.append("|" + "|".join(["---"] * 5) + "|")
    for r in rows:
        row = [r["label"]]
        ca = r.get("ca_nhd", {}) or {}
        for k in ("ca_2d_ap_0.25", "ca_2d_ap_0.50", "ca_2d_ap_0.75", "ca_macro_2d_ap_0.50"):
            row.append(_fmt(*_as_tuple(ca.get(k))))
        out.append("| " + " | ".join(row) + " |")
    out.append("\n")

    # ----- 8) Reproducibility
    out.append("## 8. Reproducibility\n")
    out.append(f"- **git commit**: `{_git_commit()}`{' (working tree DIRTY)' if _git_dirty() else ''}\n")
    out.append("- **environment**: `/storage3/3DOM/vshukla/envs/ovmono3d` (Python 3.8.20, PyTorch 2.4.1+cu121, detectron2 fork, pytorch3d CPU pinned 055ab3a)\n")
    out.append("- **hardware**: 1× NVIDIA A40 (node81)\n")
    out.append("- **dataset prep**: see `datasets/Omni3D/dataset_stats/dataset_stats.md` for the canonical inventory; regenerated every prep.\n\n")
    out.append("**Per-run config snapshots:**\n")
    for r in rows:
        cfg = (r.get("config_path") or
               (r.get("runs", [{}])[0].get("config_path") if r.get("n_seeds", 1) > 1 else None))
        if cfg:
            out.append(f"- {r['label']}: `{cfg}`\n")
    out.append("\n")

    # ----- 9) Per-run raw appendix
    out.append("## 9. Per-run raw numbers (appendix)\n")
    for r in rows:
        out.append(f"\n### {r['label']}\n")
        out.append(f"- n_seeds: {r.get('n_seeds', 1)}\n")
        if r.get("run_dirs"):
            out.append("- run_dirs:\n")
            for rd in r["run_dirs"]:
                out.append(f"  - `{rd}`\n")
        elif r.get("run_dir"):
            out.append(f"- run_dir: `{r['run_dir']}`\n")
        # Compact summary block
        ap2d = r.get("summary", {}).get("2D", {}).get("AP", ("—", 0.0))
        ap3d = r.get("summary", {}).get("3D", {}).get("AP", ("—", 0.0))
        relap3d = r.get("summary", {}).get("3D-Rel", {}).get("AP", ("—", 0.0))
        out.append(f"- 2D AP: {_fmt(*ap2d) if isinstance(ap2d, tuple) else ap2d}\n")
        out.append(f"- 3D AP: {_fmt(*ap3d) if isinstance(ap3d, tuple) else ap3d}\n")
        out.append(f"- Rel-AP3D: {_fmt(*relap3d) if isinstance(relap3d, tuple) else relap3d}\n")

    return "".join(out) if all(isinstance(x, str) for x in out) else "\n".join(out)


def render_latex_main(rows: List[Dict[str, Any]], classes: List[str]) -> str:
    """Booktabs-style main table (subset suitable for paper)."""
    lines = [r"\begin{tabular}{l" + "r" * 7 + "}", r"\toprule",
             r"Method & 2D AP@0.5 & 2D AP & 3D AP & Rel-AP3D & BEV@0.5 & BEV@0.25 & ca-2D AP@0.5 \\",
             r"\midrule"]
    for r in rows:
        cells = [r["label"]]
        cells.append(_cell(lambda a: a["summary"].get("2D", {}).get("AP50", (float("nan"), 0.0)), r))
        cells.append(_cell(lambda a: a["summary"].get("2D", {}).get("AP", (float("nan"), 0.0)), r))
        cells.append(_cell(lambda a: a["summary"].get("3D", {}).get("AP", (float("nan"), 0.0)), r))
        cells.append(_cell(lambda a: a["summary"].get("3D-Rel", {}).get("AP", (float("nan"), 0.0)), r))
        cells.append(_cell(lambda a: a["bev"].get("0.5", {}).get("micro", (float("nan"), 0.0)), r))
        cells.append(_cell(lambda a: a["bev"].get("0.25", {}).get("micro", (float("nan"), 0.0)), r))
        cells.append(_cell(lambda a: a["ca_nhd"].get("ca_2d_ap_0.50", (float("nan"), 0.0)), r))
        cells_tex = [c.replace("±", r"$\pm$").replace("—", r"--") for c in cells]
        lines.append(" & ".join(cells_tex) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def render_latex_per_class_3d(rows: List[Dict[str, Any]],
                               classes: List[str], rare_classes: List[str]) -> str:
    lines = [r"\begin{tabular}{l" + "r" * (len(classes) + 1) + "}",
             r"\toprule",
             "Method & " + " & ".join(
                 (r"\textbf{" + c + r"}\textsuperscript{*}") if c in rare_classes else c
                 for c in classes) + r" & macro \\",
             r"\midrule"]
    for r in rows:
        row = [r["label"]]
        per_cls = []
        for c in classes:
            m, s = _as_tuple(r.get("per_class", {}).get("3D", {}).get(c, {}).get("AP"))
            if math.isnan(m):
                row.append("--")
            else:
                row.append(_fmt(m, s))
                per_cls.append(m)
        macro = sum(per_cls) / len(per_cls) if per_cls else float("nan")
        row.append(f"{macro:.2f}" if not math.isnan(macro) else "--")
        row_tex = [c.replace("±", r"$\pm$").replace("—", r"--") for c in row]
        lines.append(" & ".join(row_tex) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\textsuperscript{*} rare classes; mean$\pm$std across seeds"]
    return "\n".join(lines)


# ---------------------------------------------------------------
# JSON sanitization for output
# ---------------------------------------------------------------

def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        # mean ± std tuples flattened
        if len(obj) == 2 and all(isinstance(v, (int, float)) for v in obj):
            return {"mean": float(obj[0]) if not math.isnan(obj[0]) else None,
                    "std": float(obj[1])}
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    return obj


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--finetuned-seeds", nargs="+", type=Path, default=[],
                   help="Eval dirs of the multi-seed fine-tuned run (one per seed).")
    p.add_argument("--ablation-rt035", type=Path, default=None,
                   help="Eval dir of the REPEAT_THRESHOLD=0.35 single-seed ablation.")
    p.add_argument("--zeroshot-rpn", type=Path, default=None,
                   help="Eval dir of the closed-vocab RPN-transfer zero-shot.")
    p.add_argument("--zeroshot-oracle", type=Path, default=None,
                   help="Eval dir of the GDino-oracle zero-shot (paper protocol).")
    p.add_argument("--gt", type=Path,
                   default=Path("datasets/Omni3D/WildBox_val.json"),
                   help="GT JSON for dataset facts (image/annotation/video count).")
    p.add_argument("--dataset-stats", type=Path,
                   default=Path("datasets/Omni3D/dataset_stats/dataset_stats.md"),
                   help="Dataset inventory markdown produced by tools/dataset_stats.py.")
    p.add_argument("--classes", nargs="+", required=True,
                   help="Full class list, in display order. e.g. giraffe grevys_zebra elephant plains_zebra rhino gazelle")
    p.add_argument("--rare-classes", nargs="+", default=[],
                   help="Subset of --classes treated as rare (≤5 train videos).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory (will be created).")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    # zero-shot RPN
    if args.zeroshot_rpn and args.zeroshot_rpn.exists():
        r = normalize_single_run(gather_run(args.zeroshot_rpn))
        r["label"] = "zero-shot (RPN-transfer, closed-vocab)"
        r["n_seeds"] = 1
        rows.append(r)

    # zero-shot GDino oracle
    if args.zeroshot_oracle and args.zeroshot_oracle.exists():
        r = normalize_single_run(gather_run(args.zeroshot_oracle))
        r["label"] = "zero-shot (GDino oracle, paper protocol)"
        r["n_seeds"] = 1
        rows.append(r)

    # multi-seed fine-tuned
    if args.finetuned_seeds:
        seed_runs = [gather_run(d) for d in args.finetuned_seeds]
        agg = aggregate_runs(seed_runs,
                             mode_name=f"fine-tuned ({len(seed_runs)}-seed mean ± std)")
        agg["runs"] = seed_runs
        rows.append(agg)

    # ablation rt=0.35
    if args.ablation_rt035 and args.ablation_rt035.exists():
        r = normalize_single_run(gather_run(args.ablation_rt035))
        r["label"] = "ablation: REPEAT_THRESHOLD=0.35 (1 seed)"
        r["n_seeds"] = 1
        rows.append(r)

    # GT facts
    gt_facts = {}
    if args.gt.exists():
        gt = json.load(open(args.gt))
        vids = {im.get("file_path", "/").split("/")[-3] for im in gt.get("images", [])}
        gt_facts = {
            "path": str(args.gt),
            "n_images": len(gt.get("images", [])),
            "n_anns": len(gt.get("annotations", [])),
            "n_videos": len(vids),
        }

    # Dataset stats embed
    ds_md = args.dataset_stats.read_text(errors="ignore") if args.dataset_stats.exists() else None

    # Render markdown + LaTeX + JSON
    md = render_markdown(rows, args.classes, args.rare_classes, ds_md, gt_facts)
    (args.out / "paper_results.md").write_text(md)

    tex_main = render_latex_main(rows, args.classes)
    (args.out / "paper_table_main.tex").write_text(tex_main + "\n")

    tex_3d = render_latex_per_class_3d(rows, args.classes, args.rare_classes)
    (args.out / "paper_table_perclass_3d.tex").write_text(tex_3d + "\n")

    # JSON dump (full)
    json_out = {
        "meta": {
            "git": _git_commit(),
            "git_dirty": _git_dirty(),
            "gt_facts": gt_facts,
            "classes": args.classes,
            "rare_classes": args.rare_classes,
        },
        "rows": [
            {
                "label": r["label"],
                "n_seeds": r.get("n_seeds", 1),
                "run_dirs": r.get("run_dirs", [r.get("run_dir")]) if r.get("run_dirs") or r.get("run_dir") else [],
                "summary": to_jsonable(r.get("summary", {})),
                "per_class": to_jsonable(r.get("per_class", {})),
                "bev": to_jsonable(r.get("bev", {})),
                "disent_nhd": to_jsonable(r.get("disent_nhd", {})),
                "ca_nhd": to_jsonable(r.get("ca_nhd", {})),
            }
            for r in rows
        ],
    }
    (args.out / "paper_results.json").write_text(json.dumps(json_out, indent=2))

    print(f"\nWrote:")
    print(f"  {args.out}/paper_results.md          (single markdown — paper text source)")
    print(f"  {args.out}/paper_table_main.tex      (LaTeX main table)")
    print(f"  {args.out}/paper_table_perclass_3d.tex (LaTeX per-class 3D AP)")
    print(f"  {args.out}/paper_results.json        (machine-readable everything)")
    print()
    print("Preview (first 80 lines of paper_results.md):")
    print("-" * 60)
    print("\n".join(md.splitlines()[:80]))


if __name__ == "__main__":
    main()

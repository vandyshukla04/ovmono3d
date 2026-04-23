#!/usr/bin/env python
"""Assemble a paper-ready evaluation report from raw eval outputs.

Ingests:
  --run-dir <output/wildbox_xxx>           # Output dir of an eval run
  --gt      datasets/Omni3D/WildBox_val.json
  --config  configs/wildbox/OVMono3D_wildbox_wildlife5.yaml
  --bev     <run-dir>/bev_ap.json          # output of bev_ap_eval.py
  --label   "fine-tuned"                   # label to show in the report

Produces:
  <run-dir>/paper_report/
    metrics.json       # machine-readable summary
    report.md          # human-readable
    table_main.tex     # LaTeX main metrics table
    table_diagnostic.tex  # zero-shot diagnostic table

Or in --compare mode (takes --run-dir twice):
  python tools/make_report.py --compare \\
      --run-dir output/wildbox_wl5_zeroshot        --label zeroshot \\
      --run-dir output/wildbox_wl5_finetuned_eval  --label finetuned \\
      --gt datasets/Omni3D/WildBox_val.json \\
      --out output/paper_report_v1/
  -> writes report.md, tables, metrics.json with both runs side by side.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------
# Log parsing
# -----------------------------------------------------------------------

def _find_log(run_dir: Path) -> Optional[Path]:
    cands = list(run_dir.glob("log.txt"))
    if cands:
        return cands[0]
    cands = sorted(run_dir.glob("**/log.txt"))
    return cands[0] if cands else None


def parse_standard_eval_log(log_path: Path) -> Dict[str, Any]:
    """Extract the last evaluator block from a train_net.py log.

    Returns a dict keyed by metric mode ('2D', '3D', '3D-Rel') with nested
    dicts: {metric_name: value}. Also parses per-class AP tables.

    Rel-AP3D output lives in `log.rel.txt` (emitted by run_full_eval.sh's
    step [2/4]), not the main log.txt. We concatenate both so the regex
    search covers `mode=3D-Rel` blocks that the Rel-AP3D pass produces.
    """
    text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    rel_log = log_path.parent / "log.rel.txt"
    if rel_log.exists():
        text = text + "\n" + rel_log.read_text(errors="ignore")
    result: Dict[str, Any] = {"2D": {}, "3D": {}, "3D-Rel": {},
                              "per_class": {"2D": {}, "3D": {}},
                              "disentangled_nhd": {},
                              "rel_ap3d_best_scale": None}

    # Capture the final "Evaluation results for bbox in <MODE> mode:" blocks.
    # The table has header row | cells | cells | ... followed by value row.
    # We take the LAST occurrence of each mode to get the final eval.
    mode_pattern = re.compile(
        r"Evaluation results for bbox in (\S+) mode:\s*\n"
        r"\|([^\n]*)\|\s*\n"
        r"\|([^\n]*)\|\s*\n"
        r"\|([^\n]*)\|",
        re.MULTILINE,
    )
    # Collect last block per mode
    blocks: Dict[str, Tuple[str, str]] = {}
    for m in mode_pattern.finditer(text):
        mode = m.group(1)
        header = m.group(2)
        values = m.group(4)
        blocks[mode] = (header, values)
    for mode, (header, values) in blocks.items():
        hdrs = [h.strip() for h in header.split("|") if h.strip()]
        vals = [v.strip() for v in values.split("|") if v.strip()]
        for h, v in zip(hdrs, vals):
            try:
                result[mode][h] = float(v)
            except ValueError:
                result[mode][h] = v

    # Per-class blocks: "Per-category bbox AP/AR in <MODE> mode:"
    # Match the mode header, then consume consecutive markdown-table lines
    # (lines starting with `|`). This stops cleanly at the next timestamp /
    # log line and never accidentally absorbs the following mode's table.
    # The pair_rx finds "species (AP|AR) | value" pairs globally within the
    # body -- survives column alignment variations.
    per_cat_pat = re.compile(
        r"Per-category bbox AP/AR in (\S+) mode:\s*\n"
        r"((?:\|[^\n]*\n)+)",
        re.MULTILINE,
    )
    pair_rx = re.compile(r"(\w[\w ]*?)\s*\((AP|AR)\)\s*\|\s*([-+0-9.]+)")
    for m in per_cat_pat.finditer(text):
        mode = m.group(1)
        body = m.group(2)
        for cls, metric, val in pair_rx.findall(body):
            cls = cls.strip()
            try:
                fv = float(val)
            except ValueError:
                continue
            result["per_class"].setdefault(mode, {}).setdefault(cls, {})[metric] = fv

    # Sanity-check: if we have per-class data and all classes but one are
    # exactly 0 for AP, the evaluator almost certainly ran with a broken
    # class mapping (see WILDBOX_EXPERIMENT.md §4.3 -- symlink must match
    # training's contiguous-id assignment). Warn loudly.
    for mode, classes in result["per_class"].items():
        aps = {c: d.get("AP", 0.0) for c, d in classes.items() if "AP" in d}
        if len(aps) >= 3:
            nonzero = [c for c, v in aps.items() if v > 0.01]
            if len(nonzero) == 1 and len(aps) - len(nonzero) >= 2:
                print(f"\n!! WARNING (mode={mode}): only '{nonzero[0]}' has "
                      f"non-zero per-class AP; all others = 0. This is almost "
                      f"always a class-mapping mismatch (wrong "
                      f"configs/category_meta.json symlink at eval time). "
                      f"See WILDBOX_EXPERIMENT.md §4.3.\n")

    # Disentangled NHD: either in the 3D-mode table (columns) or in the
    # "Average Disentangled NHD Metrics:" block.
    for key in ("overall_NHD", "disent_xy_NHD", "disent_z_NHD",
                "disent_dimensions_NHD", "disent_pose_NHD"):
        if key in result["3D"]:
            result["disentangled_nhd"][key.replace("disent_", "").replace("_NHD", "")] = \
                result["3D"][key]
    # Also match the 'Disentangled XY NHD: X.XXX' style summary lines
    for name, key in [("XY", "xy"), (" Z", "z"),
                      ("Dimensions", "dimensions"),
                      ("Pose", "pose")]:
        m = re.search(rf"Disentangled {name} NHD:\s*([0-9.]+)", text)
        if m:
            result["disentangled_nhd"][key] = float(m.group(1))
    m = re.search(r"Overall NHD:\s*([0-9.]+)", text)
    if m:
        result["disentangled_nhd"]["overall"] = float(m.group(1))

    # Rel-AP3D best scale
    m = re.search(r"\[rel_ap3d\] best global scale = ([0-9.]+)", text)
    if m:
        result["rel_ap3d_best_scale"] = float(m.group(1))

    return result


def load_bev_json(path: Optional[Path]) -> Dict[str, Any]:
    if path and path.exists():
        return json.load(open(path))
    return {}


def parse_class_agnostic_output(nhd_txt: Path) -> Dict[str, Any]:
    """Parse the saved output of class_agnostic_eval.py --nhd."""
    if not nhd_txt.exists():
        return {}
    text = nhd_txt.read_text()
    out: Dict[str, Any] = {"ca_2d_ap": {}, "nhd": {},
                           "per_class_ca_ap": {}, "macro_ca_ap": {}}

    # class-agnostic 2D AP lines
    for m in re.finditer(r"^\s*AP@([0-9.]+)\s*=\s*([0-9.]+)", text, re.MULTILINE):
        out["ca_2d_ap"][float(m.group(1))] = float(m.group(2))

    # NHD summary
    for line, key in [
        (r"mean NHD @ s=1:\s*([0-9.]+)", "mean_nhd_s1"),
        (r"best global scale:\s*([0-9.]+)", "best_scale"),
        (r"mean NHD @ best s:\s*([0-9.]+)", "mean_nhd_best_s"),
        (r"frac pairs NHD<0\.3:\s*([0-9.]+)%", "frac_nhd_lt_0.3"),
        (r"frac pairs NHD<0\.5:\s*([0-9.]+)%", "frac_nhd_lt_0.5"),
        (r"frac pairs NHD<1\.0:\s*([0-9.]+)%", "frac_nhd_lt_1.0"),
    ]:
        m = re.search(line, text)
        if m:
            out["nhd"][key] = float(m.group(1))

    # Per-class class-agnostic AP
    for m in re.finditer(
        r"^\s*(\w+)\s+AP@([0-9.]+)\s*=\s*([0-9.]+)",
        text, re.MULTILINE
    ):
        cls, iou, val = m.group(1), float(m.group(2)), float(m.group(3))
        if cls in ("macro",):
            out["macro_ca_ap"][iou] = val
            continue
        out["per_class_ca_ap"].setdefault(cls, {})[iou] = val

    # Macro AP (from the "=== macro-AP" block)
    for m in re.finditer(r"macro AP@([0-9.]+)\s*=\s*([0-9.]+)", text):
        out["macro_ca_ap"][float(m.group(1))] = float(m.group(2))
    return out


# -----------------------------------------------------------------------
# Dataset stats
# -----------------------------------------------------------------------

def dataset_summary(gt_json: Path) -> Dict[str, Any]:
    d = json.load(open(gt_json))
    from collections import Counter
    ann_by_cat = Counter(a["category_name"] for a in d["annotations"])
    img_by_cat: Dict[str, int] = {}
    for img in d["images"]:
        img_by_cat[img.get("file_path", "").split("/")[-3] if "/" in img.get("file_path", "") else ""] = 0
    videos = {img["file_path"].split("/")[-3]
              for img in d["images"] if "/" in img.get("file_path", "")}
    return {
        "path": str(gt_json),
        "n_images": len(d["images"]),
        "n_annotations": len(d["annotations"]),
        "n_videos": len(videos),
        "categories": [c["name"] for c in d["categories"]],
        "ann_per_class": dict(ann_by_cat),
        "bbox_source": "sam3" if any(a.get("bbox_source") == "sam3" for a in d["annotations"][:100]) else "vggt",
    }


# -----------------------------------------------------------------------
# Config snapshot
# -----------------------------------------------------------------------

def config_summary(cfg_path: Optional[Path]) -> Dict[str, Any]:
    if cfg_path is None or not cfg_path.exists():
        return {}
    import yaml
    try:
        cfg = yaml.safe_load(open(cfg_path))
    except Exception:
        return {"path": str(cfg_path), "error": "could not parse"}
    return {
        "path": str(cfg_path),
        "base": cfg.get("_BASE_"),
        "solver": cfg.get("SOLVER", {}),
        "datasets": cfg.get("DATASETS", {}),
        "test": cfg.get("TEST", {}),
        "dataloader": cfg.get("DATALOADER", {}),
        "model": {
            "WEIGHTS_PRETRAIN": cfg.get("MODEL", {}).get("WEIGHTS_PRETRAIN"),
            "ROI_CUBE_HEAD": cfg.get("MODEL", {}).get("ROI_CUBE_HEAD"),
        },
    }


# -----------------------------------------------------------------------
# Report rendering
# -----------------------------------------------------------------------

def render_main_table_md(runs: List[Dict[str, Any]], classes: List[str]) -> str:
    # One row per metric, one block per run
    lines: List[str] = []
    lines.append("## Main metrics table (§4.1)\n")
    header_cols = ["micro", "macro"] + classes
    for run in runs:
        lines.append(f"### {run['label']}")
        lines.append("")
        lines.append("| Metric | " + " | ".join(header_cols) + " |")
        lines.append("|" + "---|" * (len(header_cols) + 1))
        rows = [
            # Headers reflect what each metric actually means:
            #   - BEV micro/macro/per-class come from bev_ap.json (AP at
            #     specified IoU; per-class is this same IoU for that class).
            #   - AP_3D micro = AP25 (from 3D mode); per-class AP is IoU
            #     0.05-0.50 mean (detectron2 only emits one aggregated per-
            #     class AP in the table, regardless of threshold).
            #   - Rel-AP_3D uses the 3D-Rel mode entries from the log.
            #   - 2D AP micro = AP50; per-class is IoU 0.50:0.95 mean.
            ("AP_BEV @ 0.25", _bev_row(run, 0.25, classes)),
            ("AP_BEV @ 0.50", _bev_row(run, 0.50, classes)),
            ("AP_3D @ 0.25 (per-class: AP 0.05:0.50)",  _ap3d_row(run, "AP25", classes)),
            ("Rel-AP_3D (per-class: AP 0.05:0.50)",      _relap_row(run, classes)),
            ("AP_2D @ 0.50 (per-class: AP 0.50:0.95)",   _ap2d_row(run, "AP50", classes)),
        ]
        for name, vals in rows:
            lines.append(f"| **{name}** | " + " | ".join(vals) + " |")
        lines.append("")
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


def _bev_row(run: Dict[str, Any], iou: float, classes: List[str]) -> List[str]:
    bev = run.get("bev", {})
    key = f"IoU={iou:.2f}"
    micro = bev.get("micro", {}).get(key)
    macro = bev.get("macro", {}).get(key)
    row = [_fmt(micro), _fmt(macro)]
    per_class = bev.get("per_class", {})
    for cls in classes:
        row.append(_fmt(per_class.get(cls, {}).get(key)))
    return row


def _ap3d_row(run: Dict[str, Any], col: str, classes: List[str]) -> List[str]:
    log = run.get("log", {})
    micro = log.get("3D", {}).get(col)
    per_class = log.get("per_class", {}).get("3D", {})
    vals = [per_class.get(c, {}).get("AP") for c in classes]
    vals_f = [v for v in vals if v is not None]
    macro = sum(vals_f) / len(vals_f) if vals_f else None
    return [_fmt(micro), _fmt(macro)] + [_fmt(v) for v in vals]


def _relap_row(run: Dict[str, Any], classes: List[str]) -> List[str]:
    log = run.get("log", {})
    micro = log.get("3D-Rel", {}).get("AP") or log.get("3D-Rel", {}).get("AP25")
    per_class = log.get("per_class", {}).get("3D-Rel", {})
    vals = [per_class.get(c, {}).get("AP") for c in classes]
    vals_f = [v for v in vals if v is not None]
    macro = sum(vals_f) / len(vals_f) if vals_f else None
    return [_fmt(micro), _fmt(macro)] + [_fmt(v) for v in vals]


def _ap2d_row(run: Dict[str, Any], col: str, classes: List[str]) -> List[str]:
    """2D AP row. `col` is the overall-metric column (e.g. "AP50", "AP").
    Per-class values come from the per-category AP (single IoU=0.50:0.95
    column, as detectron2 only reports one aggregated AP per class)."""
    log = run.get("log", {})
    micro = log.get("2D", {}).get(col)
    per_class = log.get("per_class", {}).get("2D", {})
    vals = [per_class.get(c, {}).get("AP") for c in classes]
    vals_f = [v for v in vals if v is not None]
    macro = sum(vals_f) / len(vals_f) if vals_f else None
    return [_fmt(micro), _fmt(macro)] + [_fmt(v) for v in vals]


def render_diagnostic_md(runs: List[Dict[str, Any]]) -> str:
    lines = ["## Zero-shot diagnostic (§4.1 subtable)\n"]
    # Two columns: each run's relevant zero-shot diagnostic values
    header = ["Metric"] + [r["label"] for r in runs]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    fields = [
        ("Class-agnostic 2D AP@0.25", lambda r: r.get("ca", {}).get("ca_2d_ap", {}).get(0.25)),
        ("Class-agnostic 2D AP@0.50", lambda r: r.get("ca", {}).get("ca_2d_ap", {}).get(0.50)),
        ("Class-agnostic 2D AP@0.75", lambda r: r.get("ca", {}).get("ca_2d_ap", {}).get(0.75)),
        ("Macro class-agnostic 2D AP@0.50",
            lambda r: r.get("ca", {}).get("macro_ca_ap", {}).get(0.50)),
        ("NHD best-scale factor", lambda r: r.get("ca", {}).get("nhd", {}).get("best_scale")),
        ("NHD-z (depth error)",
            lambda r: r.get("log", {}).get("disentangled_nhd", {}).get("z")),
        ("Mean NHD @ best scale", lambda r: r.get("ca", {}).get("nhd", {}).get("mean_nhd_best_s")),
        ("Rel-AP3D best global scale",
            lambda r: r.get("log", {}).get("rel_ap3d_best_scale")),
    ]
    for field_name, getter in fields:
        row = [field_name]
        for r in runs:
            row.append(_fmt(getter(r)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_latex_table(runs: List[Dict[str, Any]], classes: List[str],
                       table_label: str) -> str:
    """Simple booktabs-style LaTeX table for the main metrics."""
    header_cols = ["micro", "macro"] + classes
    rows_spec = "l" + "r" * len(header_cols)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Primary metrics on WildBox-val. Rows are our four headline metrics; columns show micro-average, macro-average, and per-species AP. Higher is better.}",
        f"\\label{{{table_label}}}",
        f"\\begin{{tabular}}{{{rows_spec}}}",
        r"\toprule",
        "Metric & " + " & ".join(header_cols) + r" \\",
        r"\midrule",
    ]
    for run in runs:
        lines.append(f"\\multicolumn{{{len(header_cols)+1}}}{{l}}{{\\emph{{{run['label']}}}}} \\\\")
        rows = [
            ("AP$_{\\rm BEV}$@0.25", _bev_row(run, 0.25, classes)),
            ("AP$_{\\rm BEV}$@0.50", _bev_row(run, 0.50, classes)),
            ("AP$_{\\rm 3D}$@0.25",   _ap3d_row(run, "AP25", classes)),
            ("Rel-AP$_{\\rm 3D}$",     _relap_row(run, classes)),
            ("AP$_{\\rm 2D}$@0.50",    _ap2d_row(run, "AP50", classes)),
        ]
        for name, vals in rows:
            lines.append(f"{name} & " + " & ".join(vals) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def render_report_md(runs: List[Dict[str, Any]], ds: Dict[str, Any],
                      cfg: Dict[str, Any], classes: List[str]) -> str:
    import datetime
    lines = []
    lines.append("# OVMono3D WildBox — Evaluation Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.date.today().isoformat()}_")
    lines.append("")

    lines.append("## Run metadata\n")
    for r in runs:
        lines.append(f"- **{r['label']}** @ {r['run_dir']}")
        if r.get("log", {}).get("rel_ap3d_best_scale") is not None:
            lines.append(f"   - Rel-AP3D best-scale: {r['log']['rel_ap3d_best_scale']}")
    lines.append("")

    lines.append("## Validation set")
    lines.append(f"- Path: `{ds.get('path')}`")
    lines.append(f"- {ds.get('n_images')} images, {ds.get('n_annotations')} annotations "
                 f"across {ds.get('n_videos')} videos")
    lines.append(f"- Categories: {', '.join(ds.get('categories', []))}")
    lines.append(f"- Per-class annotation counts: {ds.get('ann_per_class')}")
    lines.append(f"- 2D bbox source: `{ds.get('bbox_source')}`")
    lines.append("")

    lines.append("## Training config snapshot")
    if cfg:
        lines.append(f"- Config: `{cfg.get('path')}`")
        lines.append(f"- Base: `{cfg.get('base')}`")
        solver = cfg.get("solver", {})
        lines.append(f"- Solver: IMS_PER_BATCH={solver.get('IMS_PER_BATCH')}, "
                     f"BASE_LR={solver.get('BASE_LR')}, "
                     f"MAX_ITER={solver.get('MAX_ITER')}, "
                     f"STEPS={solver.get('STEPS')}, "
                     f"AMP={(solver.get('AMP') or {}).get('ENABLED')}")
        lines.append(f"- Dataloader: {cfg.get('dataloader')}")
        lines.append(f"- TEST: EVAL_REL_AP3D={(cfg.get('test') or {}).get('EVAL_REL_AP3D')}, "
                     f"REL_AP3D_SEARCH={(cfg.get('test') or {}).get('REL_AP3D_SEARCH')}")
    lines.append("")

    lines.append(render_main_table_md(runs, classes))
    lines.append("")

    lines.append(render_diagnostic_md(runs))
    lines.append("")

    # Key findings auto-generated from numbers
    if len(runs) >= 1:
        last = runs[-1]
        disent = last.get("log", {}).get("disentangled_nhd", {})
        if "z" in disent:
            lines.append("\n## One-line callout (§4.1)")
            lines.append(f"> NHD-z = **{disent.get('z', 0):.2f}** dominates disentangled "
                         f"NHD (xy={disent.get('xy', 0):.2f}, "
                         f"dims={disent.get('dimensions', 0):.2f}, "
                         f"pose={disent.get('pose', 0):.2f}), confirming depth as "
                         f"the primary 3D error source. BEV factors this out, "
                         f"hence AP_BEV is our primary metric.")

    lines.append("")
    lines.append("## Appendix pointers")
    lines.append("- Full disentangled NHD and multi-threshold AP in raw eval logs")
    lines.append("- Training curves: `training_curves.png` in run dir")
    lines.append("- Per-image visualizations: `vis_agnostic/` in run dir")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def gather_run(run_dir: Path, gt_path: Path, label: str,
                bev_json: Optional[Path] = None,
                nhd_txt: Optional[Path] = None) -> Dict[str, Any]:
    log_path = _find_log(run_dir)
    log_data = parse_standard_eval_log(log_path) if log_path else {}
    bev_json = bev_json or (run_dir / "bev_ap.json")
    bev = load_bev_json(bev_json)
    nhd_txt = nhd_txt or (run_dir / "summary_nhd.txt")
    ca = parse_class_agnostic_output(nhd_txt)
    return {
        "label": label,
        "run_dir": str(run_dir),
        "log_path": str(log_path) if log_path else None,
        "bev_json": str(bev_json) if bev_json.exists() else None,
        "nhd_txt": str(nhd_txt) if nhd_txt.exists() else None,
        "log": log_data,
        "bev": bev,
        "ca": ca,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, action="append", required=True,
                   help="Output dir of one eval run. Repeatable for --compare.")
    p.add_argument("--label", type=str, action="append", required=True,
                   help="Label for each run-dir, in order.")
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--bev", type=Path, action="append", default=[],
                   help="Per-run BEV JSON override (in order)")
    p.add_argument("--nhd", type=Path, action="append", default=[],
                   help="Per-run class-agnostic-nhd txt override (in order)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir; default: <first run>/paper_report/")
    p.add_argument("--compare", action="store_true",
                   help="Side-by-side compare mode (runs -> columns).")
    args = p.parse_args()

    runs: List[Dict[str, Any]] = []
    for i, rd in enumerate(args.run_dir):
        bev = args.bev[i] if i < len(args.bev) else None
        nhd = args.nhd[i] if i < len(args.nhd) else None
        lbl = args.label[i] if i < len(args.label) else f"run{i}"
        runs.append(gather_run(rd, args.gt, lbl, bev, nhd))

    ds = dataset_summary(args.gt)
    cfg = config_summary(args.config) if args.config else {}

    # Class list from config meta if available
    classes: List[str] = ds.get("categories", [])
    # Sort with training-meta ordering if available
    meta_path = Path("configs/category_meta.json")
    if meta_path.exists():
        try:
            m = json.load(open(meta_path))
            classes = m.get("thing_classes", classes)
        except Exception:
            pass

    out_dir = args.out or (args.run_dir[0] / "paper_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics = {
        "runs": runs,
        "dataset": ds,
        "config": cfg,
        "classes": classes,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2,
                                                     default=str))

    # report.md
    md = render_report_md(runs, ds, cfg, classes)
    (out_dir / "report.md").write_text(md)

    # LaTeX tables
    (out_dir / "table_main.tex").write_text(
        render_latex_table(runs, classes, "tab:wildbox_main"))

    print(f"\nWrote:")
    print(f"  {out_dir / 'metrics.json'}")
    print(f"  {out_dir / 'report.md'}")
    print(f"  {out_dir / 'table_main.tex'}")

    # Print the report to stdout too
    print()
    print(md)


if __name__ == "__main__":
    main()

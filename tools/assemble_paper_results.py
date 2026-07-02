#!/usr/bin/env python
"""Assemble all WildBox paper results into a single markdown sheet.

One section per run. Every metric we have. No LaTeX, no JSON, no clever
aggregation — just a flat dump. The 'one place to look at everything'.

Usage:
    python tools/assemble_paper_results.py \\
        --runs label="path" label="path" ... \\
        --out output/paper_results.md

Conveniences for the standard layout:
    python tools/assemble_paper_results.py \\
        --finetuned-seeds  output/wl6_rt0.5_multiseed/seed0/eval \\
                           output/wl6_rt0.5_multiseed/seed1/eval \\
                           output/wl6_rt0.5_multiseed/seed2/eval \\
        --ablation-rt035   output/wl6_rt0.35_seed0/seed0/eval \\
        --zeroshot-rpn     output/wl6_zeroshot_rpn \\
        --zeroshot-oracle  output/wl6_zeroshot_oracle2d \\
        --out              output/paper_results.md
"""
import argparse
import json
import math
import re
import statistics
from pathlib import Path
from datetime import date


# ---------- Parsers (return raw floats, no tuple wrapping) ----------

def parse_run(run_dir: Path):
    """Read every metric file in run_dir and return a flat dict of values.
    Missing keys → None. Caller decides how to display None."""
    log = (run_dir / "log.txt")
    log_rel = (run_dir / "log.rel.txt")
    bev = (run_dir / "bev_ap.json")
    nhd = (run_dir / "summary_nhd.txt")

    text = ""
    if log.exists():
        text += log.read_text(errors="ignore")
    text += "\n"
    if log_rel.exists():
        text += log_rel.read_text(errors="ignore")
    nhd_text = nhd.read_text(errors="ignore") if nhd.exists() else ""

    out = {"run_dir": str(run_dir)}

    # ---- summary AP tables for each mode ----
    for m in re.finditer(
        r"Evaluation results for bbox in (\S+) mode:\s*\n"
        r"\|([^\n]*)\|\s*\n"
        r"\|([^\n]*)\|\s*\n"
        r"\|([^\n]*)\|",
        text, re.MULTILINE,
    ):
        mode = m.group(1)
        hdrs = [h.strip() for h in m.group(2).split("|") if h.strip()]
        vals = [v.strip() for v in m.group(4).split("|") if v.strip()]
        for h, v in zip(hdrs, vals):
            try:
                out[f"{mode}__{h}"] = float(v)
            except ValueError:
                pass

    # ---- per-class AP tables ----
    for m in re.finditer(
        r"Per-category bbox AP/AR in (\S+) mode:\s*\n((?:\|[^\n]*\n)+)",
        text,
    ):
        mode = m.group(1)
        body = m.group(2)
        for pair in re.finditer(r"(\w[\w ]*?)\s*\((AP|AR)\)\s*\|\s*([-+0-9.]+)", body):
            cname, metric, val = pair.groups()
            try:
                out[f"{mode}__{cname.strip()}__{metric}"] = float(val)
            except ValueError:
                pass

    # ---- disentangled NHD ----
    for k, p in (
        ("nhd_overall", r"overall:\s*([-+0-9.]+)"),
        ("nhd_xy", r"xy:\s*([-+0-9.]+)"),
        ("nhd_z", r"z:\s*([-+0-9.]+)"),
        ("nhd_dims", r"dimensions:\s*([-+0-9.]+)"),
        ("nhd_pose", r"pose:\s*([-+0-9.]+)"),
    ):
        m = re.search(p, text)
        if m:
            try:
                out[k] = float(m.group(1))
            except ValueError:
                pass

    # ---- class-agnostic 2D AP from summary_nhd.txt ----
    for thr in ("0.25", "0.50", "0.75"):
        m = re.search(rf"AP@{re.escape(thr)}\s*=\s*([-+0-9.]+)", nhd_text)
        if m:
            try:
                out[f"ca_2d_ap_{thr}"] = float(m.group(1))
            except ValueError:
                pass
    m = re.search(r"macro AP@0\.50\s*=\s*([-+0-9.]+)", nhd_text)
    if m:
        out["ca_macro_ap_0.50"] = float(m.group(1))
    m = re.search(r"best global scale:\s*([-+0-9.]+)", nhd_text)
    if m:
        out["nhd_best_scale"] = float(m.group(1))
    m = re.search(r"mean NHD @ best s:\s*([-+0-9.]+)", nhd_text)
    if m:
        out["mean_nhd_best_s"] = float(m.group(1))

    # ---- BEV AP ----
    if bev.exists():
        try:
            bd = json.load(open(bev))
            for iou_key in ("0.25", "0.5"):
                bv = bd.get(iou_key, {})
                if "ap_micro" in bv:
                    out[f"bev_{iou_key}_micro"] = float(bv["ap_micro"])
                if "ap_macro" in bv:
                    out[f"bev_{iou_key}_macro"] = float(bv["ap_macro"])
                for c, v in (bv.get("ap_per_class") or {}).items():
                    try:
                        out[f"bev_{iou_key}__{c}"] = float(v)
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass

    return out


# ---------- Formatting helpers ----------

def fmt(v, prec=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{prec}f}"


def fmt_meanstd(values, prec=2):
    """Given a list of floats (or Nones), return 'mean ± std' or '—'."""
    xs = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.{prec}f}"
    m = statistics.mean(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return f"{m:.{prec}f} ± {s:.{prec}f}"


# ---------- One section per run ----------

def render_run_section(label, run, classes):
    """One section dumping all metrics of one run."""
    L = []
    L.append(f"### {label}")
    L.append("")
    L.append(f"`{run['run_dir']}`")
    L.append("")

    # 2D summary
    L.append("**2D detection (mode=2D):**")
    L.append("")
    L.append("| AP | AP50 | AP75 | AP small | AP medium | AP large |")
    L.append("|---|---|---|---|---|---|")
    L.append("| " + " | ".join([
        fmt(run.get("2D__AP")),
        fmt(run.get("2D__AP50")),
        fmt(run.get("2D__AP75")),
        fmt(run.get("2D__APs")),
        fmt(run.get("2D__APm")),
        fmt(run.get("2D__APl")),
    ]) + " |")
    L.append("")

    # 2D per class
    L.append("**Per-class 2D AP (COCO 0.50:0.95 mean):**")
    L.append("")
    L.append("| " + " | ".join(classes) + " |")
    L.append("|" + "|".join(["---"] * len(classes)) + "|")
    L.append("| " + " | ".join(fmt(run.get(f"2D__{c}__AP")) for c in classes) + " |")
    L.append("")

    # 3D summary
    L.append("**3D detection (mode=3D):**")
    L.append("")
    L.append("| AP | AP15 | AP25 | AP50 | overall NHD | NHD xy | NHD z | NHD dims | NHD pose |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append("| " + " | ".join([
        fmt(run.get("3D__AP")),
        fmt(run.get("3D__AP15")),
        fmt(run.get("3D__AP25")),
        fmt(run.get("3D__AP50")),
        fmt(run.get("nhd_overall")),
        fmt(run.get("nhd_xy")),
        fmt(run.get("nhd_z")),
        fmt(run.get("nhd_dims")),
        fmt(run.get("nhd_pose")),
    ]) + " |")
    L.append("")

    # 3D per class
    L.append("**Per-class 3D AP (0.05:0.50 mean):**")
    L.append("")
    L.append("| " + " | ".join(classes) + " |")
    L.append("|" + "|".join(["---"] * len(classes)) + "|")
    L.append("| " + " | ".join(fmt(run.get(f"3D__{c}__AP")) for c in classes) + " |")
    L.append("")

    # Rel-AP3D
    if run.get("3D-Rel__AP") is not None:
        L.append("**Rel-AP3D (LabelAny3D scale-aligned):**")
        L.append("")
        L.append("| AP | AP15 | AP25 | AP50 |")
        L.append("|---|---|---|---|")
        L.append("| " + " | ".join([
            fmt(run.get("3D-Rel__AP")),
            fmt(run.get("3D-Rel__AP15")),
            fmt(run.get("3D-Rel__AP25")),
            fmt(run.get("3D-Rel__AP50")),
        ]) + " |")
        L.append("")
        L.append("**Per-class Rel-AP3D:**")
        L.append("")
        L.append("| " + " | ".join(classes) + " |")
        L.append("|" + "|".join(["---"] * len(classes)) + "|")
        L.append("| " + " | ".join(fmt(run.get(f"3D-Rel__{c}__AP")) for c in classes) + " |")
        L.append("")

    # BEV
    L.append("**BEV AP (primary 3D metric):**")
    L.append("")
    L.append("| IoU | micro | macro | " + " | ".join(classes) + " |")
    L.append("|" + "|".join(["---"] * (3 + len(classes))) + "|")
    for iou in ("0.5", "0.25"):
        cells = [iou,
                 fmt(run.get(f"bev_{iou}_micro")),
                 fmt(run.get(f"bev_{iou}_macro"))]
        for c in classes:
            cells.append(fmt(run.get(f"bev_{iou}__{c}")))
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # Class-agnostic 2D AP
    L.append("**Class-agnostic 2D AP (zero-shot diagnostic — ignores labels):**")
    L.append("")
    L.append("| AP@0.25 | AP@0.50 | AP@0.75 | macro@0.50 | best-scale | mean NHD @ best-s |")
    L.append("|---|---|---|---|---|---|")
    L.append("| " + " | ".join([
        fmt(run.get("ca_2d_ap_0.25")),
        fmt(run.get("ca_2d_ap_0.50")),
        fmt(run.get("ca_2d_ap_0.75")),
        fmt(run.get("ca_macro_ap_0.50")),
        fmt(run.get("nhd_best_scale"), prec=3),
        fmt(run.get("mean_nhd_best_s"), prec=3),
    ]) + " |")
    L.append("")

    return "\n".join(L) + "\n"


def render_multiseed_summary(label, seed_runs, classes):
    """Mean ± std across N seeds. Single section showing the across-seed
    aggregation for fine-tuned multi-seed."""
    L = []
    L.append(f"### {label} — mean ± std across {len(seed_runs)} seeds")
    L.append("")

    # Per-class 2D AP
    L.append("**Per-class 2D AP (mean ± std):**")
    L.append("")
    L.append("| metric | " + " | ".join(classes) + " |")
    L.append("|" + "|".join(["---"] * (1 + len(classes))) + "|")
    L.append("| AP (COCO) | " + " | ".join(
        fmt_meanstd([r.get(f"2D__{c}__AP") for r in seed_runs])
        for c in classes
    ) + " |")
    L.append("")

    # Per-class 3D AP
    L.append("**Per-class 3D AP (mean ± std):**")
    L.append("")
    L.append("| metric | " + " | ".join(classes) + " |")
    L.append("|" + "|".join(["---"] * (1 + len(classes))) + "|")
    L.append("| 3D AP | " + " | ".join(
        fmt_meanstd([r.get(f"3D__{c}__AP") for r in seed_runs])
        for c in classes
    ) + " |")
    L.append("| Rel-AP3D | " + " | ".join(
        fmt_meanstd([r.get(f"3D-Rel__{c}__AP") for r in seed_runs])
        for c in classes
    ) + " |")
    L.append("")

    # Per-class BEV AP
    L.append("**Per-class BEV AP @ IoU 0.5 (mean ± std):**")
    L.append("")
    L.append("| metric | micro | macro | " + " | ".join(classes) + " |")
    L.append("|" + "|".join(["---"] * (3 + len(classes))) + "|")
    micro = fmt_meanstd([r.get("bev_0.5_micro") for r in seed_runs])
    macro = fmt_meanstd([r.get("bev_0.5_macro") for r in seed_runs])
    cells = ["BEV@0.5", micro, macro]
    for c in classes:
        cells.append(fmt_meanstd([r.get(f"bev_0.5__{c}") for r in seed_runs]))
    L.append("| " + " | ".join(cells) + " |")
    L.append("")

    L.append("**Per-class BEV AP @ IoU 0.25 (mean ± std):**")
    L.append("")
    L.append("| metric | micro | macro | " + " | ".join(classes) + " |")
    L.append("|" + "|".join(["---"] * (3 + len(classes))) + "|")
    micro = fmt_meanstd([r.get("bev_0.25_micro") for r in seed_runs])
    macro = fmt_meanstd([r.get("bev_0.25_macro") for r in seed_runs])
    cells = ["BEV@0.25", micro, macro]
    for c in classes:
        cells.append(fmt_meanstd([r.get(f"bev_0.25__{c}") for r in seed_runs]))
    L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # Overall (micro) AP summary
    L.append("**Overall metrics (mean ± std):**")
    L.append("")
    L.append("| 2D AP | 2D AP50 | 3D AP | Rel-AP3D | BEV@0.5 micro | BEV@0.25 micro | NHD overall | NHD z | ca-2D AP@0.5 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append("| " + " | ".join([
        fmt_meanstd([r.get("2D__AP") for r in seed_runs]),
        fmt_meanstd([r.get("2D__AP50") for r in seed_runs]),
        fmt_meanstd([r.get("3D__AP") for r in seed_runs]),
        fmt_meanstd([r.get("3D-Rel__AP") for r in seed_runs]),
        fmt_meanstd([r.get("bev_0.5_micro") for r in seed_runs]),
        fmt_meanstd([r.get("bev_0.25_micro") for r in seed_runs]),
        fmt_meanstd([r.get("nhd_overall") for r in seed_runs]),
        fmt_meanstd([r.get("nhd_z") for r in seed_runs]),
        fmt_meanstd([r.get("ca_2d_ap_0.50") for r in seed_runs]),
    ]) + " |")
    L.append("")

    return "\n".join(L) + "\n"


# ---------- Main ----------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--finetuned-seeds", nargs="+", type=Path, default=[])
    p.add_argument("--ablation-rt035", type=Path, default=None)
    p.add_argument("--zeroshot-rpn", type=Path, default=None)
    p.add_argument("--zeroshot-oracle", type=Path, default=None)
    p.add_argument("--gt", type=Path,
                   default=Path("datasets/Omni3D/WildBox_val.json"))
    p.add_argument("--dataset-stats", type=Path,
                   default=Path("datasets/Omni3D/dataset_stats/dataset_stats.md"))
    p.add_argument("--classes", nargs="+", required=True,
                   help="Class names in display order.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output markdown file path. Parent dir created if needed.")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # GT facts
    gt_summary = ""
    if args.gt.exists():
        try:
            gt = json.load(open(args.gt))
            vids = {im.get("file_path", "/").split("/")[-3] for im in gt.get("images", [])}
            gt_summary = (f"**GT**: `{args.gt}` — "
                          f"{len(gt.get('images', []))} images, "
                          f"{len(gt.get('annotations', []))} annotations, "
                          f"{len(vids)} videos\n")
        except Exception:
            pass

    # Build the master document
    doc = []
    doc.append(f"# WildBox paper results — assembled {date.today().isoformat()}\n")
    doc.append(gt_summary)
    doc.append("Classes: " + ", ".join(args.classes))
    doc.append("\n")

    # ---- Dataset facts ----
    if args.dataset_stats.exists():
        doc.append("## 1. Dataset facts")
        doc.append("")
        doc.append(f"_Auto-generated by `tools/dataset_stats.py` — see `{args.dataset_stats}`._")
        doc.append("")
        body = args.dataset_stats.read_text(errors="ignore")
        # Strip top-level heading from embed
        body = re.sub(r"^# .*\n", "", body, count=1)
        doc.append(body)
        doc.append("\n---\n")

    # ---- Each run as a section ----
    doc.append("## 2. Per-run results\n")

    if args.zeroshot_rpn and args.zeroshot_rpn.exists():
        doc.append(render_run_section(
            "Zero-shot (RPN-transfer, closed-vocab)",
            parse_run(args.zeroshot_rpn), args.classes))

    if args.zeroshot_oracle and args.zeroshot_oracle.exists():
        doc.append(render_run_section(
            "Zero-shot (GDino oracle, paper protocol)",
            parse_run(args.zeroshot_oracle), args.classes))

    seed_runs = []
    if args.finetuned_seeds:
        for i, p_ in enumerate(args.finetuned_seeds):
            r = parse_run(p_)
            seed_runs.append(r)
            doc.append(render_run_section(
                f"Fine-tuned (REPEAT_THRESHOLD=0.5) seed {i}",
                r, args.classes))

    if args.ablation_rt035 and args.ablation_rt035.exists():
        doc.append(render_run_section(
            "Ablation: REPEAT_THRESHOLD=0.35 (1 seed)",
            parse_run(args.ablation_rt035), args.classes))

    # ---- Multi-seed mean±std summary ----
    if len(seed_runs) > 1:
        doc.append("## 3. Multi-seed aggregation\n")
        doc.append(render_multiseed_summary(
            "Fine-tuned (REPEAT_THRESHOLD=0.5)",
            seed_runs, args.classes))

    # ---- Reproducibility footer ----
    doc.append("## 4. Reproducibility\n")
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip())
        doc.append(f"- git commit: `{commit}`{' (working tree DIRTY)' if dirty else ''}")
    except Exception:
        pass
    doc.append("- env: Python 3.8.20, PyTorch 2.4.1+cu121")
    doc.append("- hardware: 1× NVIDIA A40")
    doc.append("")
    doc.append("Per-run config snapshots are at `<run_dir>/config.yaml` for each section above.")

    # Write
    out_text = "\n".join(doc)
    args.out.write_text(out_text)
    print(f"Wrote {args.out}  ({len(out_text)} chars, {out_text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()

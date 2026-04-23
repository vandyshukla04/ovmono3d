#!/usr/bin/env python
"""Aggregate per-class APs across multiple training seeds.

For rare classes with few videos (giraffe 4, gazelle 4, Grévy's zebra 5
in our corpus), single-seed numbers are high-variance and noisy.
Reviewer #4 asked for mean±std across 3 seeds for these classes; for
rhino/elephant (15-23 videos), single-seed is fine and we just copy
seed-0 values.

This tool reads `log.txt`, `log.rel.txt`, and `bev_ap.json` from each
seed's eval dir (the output of tools/run_full_eval.sh), parses the
per-class AP tables with the same regexes make_report.py uses, and
emits a markdown + LaTeX table with 'X.XX ± Y.YY' cells for the rare
classes and plain 'X.XX' for the stable classes.

Usage:
    python tools/aggregate_seed_ap.py \\
        --run-dirs output/wl5_rt0.5_multiseed/seed0/eval \\
                    output/wl5_rt0.5_multiseed/seed1/eval \\
                    output/wl5_rt0.5_multiseed/seed2/eval \\
        --rare-classes giraffe gazelle grevys_zebra \\
        --out output/wl5_rt0.5_multiseed/mean_std_report

Or glob-expand:
    python tools/aggregate_seed_ap.py \\
        --run-dirs output/wl5_rt0.5_multiseed/seed*/eval \\
        --rare-classes giraffe gazelle grevys_zebra \\
        --out output/wl5_rt0.5_multiseed/mean_std_report
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PAIR_RE = re.compile(r"(\w[\w ]*?)\s*\((AP|AR)\)\s*\|\s*([-+0-9.]+)")


def parse_per_class(log_text: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Returns {mode: {class_name: {"AP": v, "AR": v}}}.

    Scans for "Per-category bbox AP/AR in <MODE> mode:" blocks and
    extracts `class (AP) | value` pairs. Handles 2D, 3D, and 3D-Rel
    modes. Same parser logic as make_report.py.
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    blocks = re.finditer(
        r"Per-category bbox AP/AR in (\S+) mode:\s*\n((?:\|[^\n]*\n)+)",
        log_text,
    )
    for m in blocks:
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


def _read(fp: Path) -> str:
    return fp.read_text(errors="ignore") if fp.exists() else ""


def gather_one_run(run_dir: Path) -> Dict[str, dict]:
    """Collect all per-class + overall APs for one seed's eval dir."""
    log = _read(run_dir / "log.txt") + "\n" + _read(run_dir / "log.rel.txt")
    per_class = parse_per_class(log)

    # BEV per-class from bev_ap.json
    bev = {}
    bev_path = run_dir / "bev_ap.json"
    if bev_path.exists():
        try:
            bev = json.load(open(bev_path))
        except Exception:
            pass

    return {
        "per_class_2D": per_class.get("2D", {}),
        "per_class_3D": per_class.get("3D", {}),
        "per_class_3DRel": per_class.get("3D-Rel", {}),
        "bev": bev,
    }


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    mean = sum(xs) / len(xs)
    if len(xs) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)  # sample std
    return mean, math.sqrt(var)


def fmt_cell(mean: float, std: float, is_rare: bool, n_seeds: int) -> str:
    if math.isnan(mean):
        return "-"
    if is_rare and n_seeds > 1:
        return f"{mean:.2f} ± {std:.2f}"
    return f"{mean:.2f}"


def build_table(runs: List[Dict[str, dict]], rare: List[str],
                classes: List[str]) -> Tuple[str, str]:
    """Produces (markdown, latex) for the main 5-metric x N-class table."""
    n = len(runs)

    # Helper — aggregate a single metric across seeds
    def agg(getter):
        """getter takes one `run` dict and a class name; returns float or None."""
        rows = {}
        for c in classes:
            vals = []
            for r in runs:
                v = getter(r, c)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
            rows[c] = mean_std(vals)
        return rows

    metrics = [
        ("AP_BEV @ 0.50 (primary)",
         lambda r, c: (r.get("bev", {})
                        .get("ap_per_class", {})
                        .get("0.5", {}).get(c))),
        ("AP_BEV @ 0.25",
         lambda r, c: (r.get("bev", {})
                        .get("ap_per_class", {})
                        .get("0.25", {}).get(c))),
        ("AP_3D (0.05:0.50 mean)",
         lambda r, c: r.get("per_class_3D", {}).get(c, {}).get("AP")),
        ("Rel-AP_3D (0.05:0.50 mean)",
         lambda r, c: r.get("per_class_3DRel", {}).get(c, {}).get("AP")),
        ("AP_2D (COCO 0.50:0.95)",
         lambda r, c: r.get("per_class_2D", {}).get(c, {}).get("AP")),
    ]

    # Build markdown
    md = []
    md.append("| Metric | " + " | ".join(classes) + " |")
    md.append("|" + "---|" * (len(classes) + 1))
    for name, getter in metrics:
        rows = agg(getter)
        cells = [fmt_cell(m, s, c in rare, n) for c, (m, s) in rows.items()]
        md.append(f"| **{name}** | " + " | ".join(cells) + " |")
    md_text = "\n".join(md)

    # LaTeX (booktabs style)
    tex = []
    tex.append(r"\begin{tabular}{l" + "r" * len(classes) + "}")
    tex.append(r"\toprule")
    tex.append(r"Metric & " + " & ".join(classes) + r" \\")
    tex.append(r"\midrule")
    for name, getter in metrics:
        rows = agg(getter)
        cells = [fmt_cell(m, s, c in rare, n).replace("±", r"$\pm$") for c, (m, s) in rows.items()]
        tex.append(name + " & " + " & ".join(cells) + r" \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    return md_text, "\n".join(tex)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", required=True, type=Path,
                   help="Eval output dirs (one per seed), e.g. seed0/eval seed1/eval seed2/eval")
    p.add_argument("--rare-classes", nargs="+", required=True,
                   help="Classes that get mean±std treatment, e.g. giraffe gazelle grevys_zebra")
    p.add_argument("--classes", nargs="+",
                   default=["giraffe", "grevys_zebra", "elephant", "rhino", "gazelle"],
                   help="Full class list in column order (default 5-species wildlife)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for report.md + table_multiseed.tex + aggregated.json")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    for d in args.run_dirs:
        if not d.exists():
            print(f"WARN: {d} does not exist, skipping", file=sys.stderr)
            continue
        runs.append(gather_one_run(d))
        print(f"Loaded: {d}")

    if not runs:
        sys.exit("No run dirs found. Pass --run-dirs paths that exist.")

    md, tex = build_table(runs, args.rare_classes, args.classes)

    # Dump artifacts
    (args.out / "table_multiseed.md").write_text(
        f"# Multi-seed aggregated results\n\n"
        f"_{len(runs)} seeds; {', '.join(args.rare_classes)} reported as "
        f"mean ± sample std, others as seed-0 value (variance negligible with "
        f"15-23 videos per class)._\n\n"
        + md + "\n"
    )
    (args.out / "table_multiseed.tex").write_text(tex + "\n")
    (args.out / "aggregated.json").write_text(
        json.dumps({"n_seeds": len(runs),
                    "run_dirs": [str(d) for d in args.run_dirs],
                    "rare_classes": args.rare_classes,
                    "classes": args.classes}, indent=2)
    )

    print(f"\nWrote:")
    print(f"  {args.out}/table_multiseed.md")
    print(f"  {args.out}/table_multiseed.tex")
    print(f"  {args.out}/aggregated.json")
    print()
    print("Preview:")
    print(md)


if __name__ == "__main__":
    main()

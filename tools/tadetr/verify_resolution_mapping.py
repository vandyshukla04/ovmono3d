"""M1 GATE A: settle the 294x518 <-> 1920x1080 resolution mapping.  [CPU, ~5-10 min]

    python tools/tadetr/verify_resolution_mapping.py \
        --train /mnt/d/3DBOX/papersubdata/WildBox_train_paper.json \
        --val   /mnt/d/3DBOX/papersubdata/WildBox_val_paper.json \
        --out   datasets/tadetr/reports/resolution_mapping.json

Three tests, because "the mapping" is really three separate claims:

T-A1 INTERNAL: per segment, project tracking_summary 3D centers with the 518-space cameras.json
     K+extrinsic and compare to the (518-space) bbox_2d centers. Validates that depth/K/T/boxes are
     one self-consistent system. NOTE this proxy's residual includes the LEGITIMATE centroid-vs-
     box-center offset (measured project-wide at 9.6 full-res px ~ 2.6 px 518-space median), so the
     gate is calibrated against that floor: GATE: median < 5 px on >= 340/345; segments above 5 px
     are consistency-FLAGGED and must be checked individually in the terrain unit test (Gate B),
     which is the true arbiter. First run (2026-08-26): p50 0.77 / p90 1.95 / max 9.5 px; 3 flagged.

T-A2 LABEL<->518: match detector-json annotations to tracking observations on (track_id, image_name)
     and compare full-res bbox centers to bbox_2d centers under two hypotheses:
       uniform:  (u,v)_full = (u,v)_518 * (1920/518)
       per-axis: u_full = u_518 * (1920/518),  v_full = v_518 * (1080/294)
     The two differ by ~0.9% in v only; this picks the convention the LABELS were built with
     (what stamping/mask work must use).

T-A3 PAD ROWS: on a sample of segments, compare depth/conf statistics of bottom rows (291-293)
     to interior rows. Uniform scaling implies rows ~292+ are letterbox padding (1080*518/1920=291.4);
     per-axis implies all 294 rows are image. Decides whether the terrain builder must crop rows.

Writes a machine-readable report; tadetr/data/resolution.py holds the adopted constants.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tadetr.data.wildbox_paths import index_by_video_seg, parse_label_path  # noqa: E402

SX = 1920.0 / 518.0          # 3.7066
SY_UNIFORM = 1920.0 / 518.0
SY_PERAXIS = 1080.0 / 294.0  # 3.6735


def seg_internal_and_obs(sp):
    """Return (median internal proj err px, obs dict (track_id, image_name) -> bbox_2d center)."""
    cams = json.loads(sp.cameras_json.read_text())["cameras"]
    cam_by_idx = {c["frame_index"]: c for c in cams}
    ts = json.loads(sp.tracking_summary.read_text())
    errs, obs = [], {}
    for tid, tr in ts["tracks"].items():
        for f, c, b in zip(tr["frames"], tr["centers"], tr["bbox_2d"]):
            cam = cam_by_idx.get(int(f))
            if cam is None:
                continue
            E = np.array(cam["extrinsic"], float)
            K = np.array(cam["intrinsic"], float)
            p = E[:, :3] @ np.array(c, float) + E[:, 3]
            if p[2] <= 1e-9:
                continue
            uv = (K @ p)[:2] / p[2]
            bc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], float)
            errs.append(float(np.hypot(*(uv - bc))))
            try:
                obs[(int(tid.split("::")[-1]) if "::" in tid else int(tid),
                     cam["image_name"])] = bc
            except ValueError:
                obs[(tid, cam["image_name"])] = bc
    return (float(np.median(errs)) if errs else float("nan")), obs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("datasets/tadetr/reports/resolution_mapping.json"))
    ap.add_argument("--pad-sample", type=int, default=6, help="segments to open depth npz for T-A3")
    args = ap.parse_args()

    index = index_by_video_seg()
    print(f"dense segments indexed: {len(index)}")

    # detector-json annotations grouped by (video, seg): (track_id, image_name) -> full-res bbox center
    lab = defaultdict(dict)
    for path in (args.train, args.val):
        d = json.loads(path.read_text())
        ims = {i["id"]: i for i in d["images"]}
        for a in d["annotations"]:
            _, video, seg, image = parse_label_path(ims[a["image_id"]]["file_path"])
            b = a["bbox"]  # full-res XYWH
            lab[(video, seg)][(a.get("track_id"), image)] = np.array(
                [b[0] + b[2] / 2, b[1] + b[3] / 2], float)
        print(f"loaded {path.name}: {len(d['annotations']):,} annotations")

    report = {"segments": {}, "summary": {}}
    int_meds, uni_meds, axis_meds = [], [], []
    n_joined = 0
    for k, sp in sorted(index.items()):
        try:
            med_int, obs = seg_internal_and_obs(sp)
        except Exception as e:  # unreadable segment -> loud, listed, not fatal
            report["segments"][sp.name] = {"error": repr(e)}
            continue
        ju, ja = [], []
        for key, bc518 in obs.items():
            full = lab.get(sp.key, {}).get(key)
            if full is None:
                continue
            ju.append(float(np.hypot(*(bc518 - full / np.array([SX, SY_UNIFORM])))))
            ja.append(float(np.hypot(*(bc518 - full / np.array([SX, SY_PERAXIS])))))
        rec = {"internal_med_px": round(med_int, 3),
               "n_label_matches": len(ju),
               "uniform_med_px": round(float(np.median(ju)), 3) if ju else None,
               "peraxis_med_px": round(float(np.median(ja)), 3) if ja else None,
               "tracking_corrected": sp.tracking_is_corrected}
        report["segments"][sp.name] = rec
        if np.isfinite(med_int):
            int_meds.append(med_int)
        if ju:
            n_joined += 1
            uni_meds.append(np.median(ju))
            axis_meds.append(np.median(ja))

    # T-A3: pad-row forensics on a sample
    pad = []
    for k, sp in list(sorted(index.items()))[:: max(1, len(index) // args.pad_sample)][:args.pad_sample]:
        z = np.load(sp.depth_npz)
        conf = z["depth_conf"]
        pad.append({
            "seg": sp.name,
            "conf_row_289_291_mean": float(conf[:, 289:291, :].mean()),
            "conf_row_292_294_mean": float(conf[:, 292:294, :].mean()),
            "depth_row_292_294_zero_frac": float((z["depth"][:, 292:294, :] <= 0).mean()),
        })
    report["pad_rows_sample"] = pad

    int_meds, uni_meds, axis_meds = map(np.array, (int_meds, uni_meds, axis_meds))
    ok_internal = int(np.sum(int_meds < 5.0))
    flagged = sorted([(v["internal_med_px"], k) for k, v in report["segments"].items()
                      if v.get("internal_med_px", 0) >= 5.0], reverse=True)
    report["summary"] = {
        "n_segments": len(index),
        "n_with_internal": len(int_meds),
        "internal_med_of_medians_px": round(float(np.median(int_meds)), 3),
        "internal_ok_lt5px": ok_internal,
        "consistency_flagged": [k for _, k in flagged],
        "n_segments_label_joined": n_joined,
        "uniform_med_of_medians_px": round(float(np.median(uni_meds)), 3),
        "peraxis_med_of_medians_px": round(float(np.median(axis_meds)), 3),
        "winner": "peraxis" if np.median(axis_meds) < np.median(uni_meds) else "uniform",
        "gate_A_pass": bool(ok_internal >= 340),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    s = report["summary"]
    print(f"\nT-A1 internal: median-of-medians {s['internal_med_of_medians_px']} px; "
          f"{s['internal_ok_lt5px']}/{s['n_with_internal']} segments < 5 px; "
          f"flagged: {s['consistency_flagged']}")
    print(f"T-A2 label<->518 ({s['n_segments_label_joined']} segments joined): "
          f"uniform {s['uniform_med_of_medians_px']} px vs per-axis {s['peraxis_med_of_medians_px']} px "
          f"-> WINNER: {s['winner']}")
    for p in pad:
        print(f"T-A3 {p['seg']}: conf rows289-291 {p['conf_row_289_291_mean']:.2f} vs "
              f"rows292-294 {p['conf_row_292_294_mean']:.2f}; depth<=0 in 292-294: "
              f"{100 * p['depth_row_292_294_zero_frac']:.1f}%")
    print(f"\nGATE A: {'PASS' if s['gate_A_pass'] else 'FAIL'}  (report -> {args.out})")
    return 0 if s["gate_A_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

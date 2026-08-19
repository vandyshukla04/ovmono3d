"""Render detection + heading onto real frames, so a human can check the orientation by eye.

    # ground truth only (works today, before any training)
    python tools/aeroview/viz_heading.py \
        --gt /mnt/d/aeroview/labelled/WildBox_val_paper.json \
        --image-root /mnt/d/3DBOX/papersubdata \
        --video DJI_20250128082221_0004_V --n 12 --out /mnt/d/aeroview/viz

    # ground truth vs a trained model
    python tools/aeroview/viz_heading.py --gt ... --image-root ... \
        --preds /mnt/d/ovmono3d-lift/<run>/.../instances_predictions.pth --out ...

WHY THIS EXISTS
---------------
Every metric this repo computes on a cuboid -- NHD, 3D IoU, BEV AP -- is EXACTLY invariant to a 180 degree
flip about the box's vertical axis (verified: min BEV IoU 1.000 over 3000 random boxes). So a box pointing
exactly backwards is, to the entire evaluation stack, indistinguishable from a correct one. The only way to
catch a systematic head/tail inversion early is to LOOK at it. That is what this renders.

WHAT IS DRAWN, per animal
    * the 2D box (GT green / prediction magenta)
    * an ARROW along the animal's heading, correctly projected (see below) -- the arrowhead is the HEAD
    * a badge: flank L/R, the strength |sin alpha|, and whether the view is USABLE (|sin alpha| >= 0.35)
    * end-on animals (|sin alpha| < 0.35) are drawn dashed: their flank bit is near-degenerate and neither
      the model nor a human should be trusted there. Do not read a flank error off a dashed arrow.

THE PROJECTION (this is the part that is easy to get wrong)
-----------------------------------------------------------
alpha is ALLOCENTRIC: measured against the viewing ray, not the image axes. So it does NOT map to a fixed
image angle -- the same alpha points differently depending on where the animal sits in the frame. Recover the
3D direction first, then differentiate the projection:

    r = horizontal component of the ray to the animal (unit)      s = up x r
    d = cos(alpha) * r + sin(alpha) * s                           (the heading, in camera coords)
    image_dir  proportional to  ( fx * (d_x - (p_x/p_z) d_z),  fy * (d_y - (p_y/p_z) d_z) )

Treating alpha as a raw image angle is wrong everywhere except the principal point.

WHERE `up` COMES FROM (measured, not assumed)
---------------------------------------------
The Omni3D convention is dims = [W, H, L] with H on the box's local Y, so `R_cam[:, 1]` is the box's vertical
axis. Measured over 2,491 val frames, every box in a frame agrees on that axis to **0.00 degrees** -- it is a
per-segment constant. So `up` is recoverable EXACTLY from any single annotation, and this tool needs no
VGGT artefact, no cameras.json and no tracking_summary.json.

*** SIGN (this was a real bug, caught by eye before it was caught by any check) ***
`R_cam[:, 1]` is -up, NOT up: measured over all 7,460 labelled val annotations it points AWAY from the camera
in 100.0% of cases (mean dot with the animal->camera direction -0.286, zero positives). The pipeline's own
basis signs up TOWARDS the camera (`frame.sign_up_toward_camera`). Getting this backwards flips `s = up x r`,
which inverts the arrow's LEFT/RIGHT component while leaving the printed flank label -- computed from
sin(alpha) directly -- correct. So the picture disagrees with its own caption, and nothing else catches it.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

USABLE_SIN = 0.35  # below this the animal is near end-on and the flank bit is degenerate


# ---------------------------------------------------------------------------------------------------
def allocentric_dir(alpha: float, p_cam: np.ndarray, up_cam: np.ndarray) -> np.ndarray:
    """The 3D heading (camera coords) for an allocentric angle at a given position."""
    up = up_cam / max(np.linalg.norm(up_cam), 1e-12)
    r = p_cam - np.dot(p_cam, up) * up          # horizontal component of the ray to the animal
    n = np.linalg.norm(r)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    r = r / n
    s = np.cross(up, r)
    return math.cos(alpha) * r + math.sin(alpha) * s


def project_dir(d: np.ndarray, p_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """d/dt of the projection of p+td -- i.e. the image-space direction of a 3D direction at p."""
    pz = p_cam[2] if abs(p_cam[2]) > 1e-9 else 1e-9
    v = np.array([K[0, 0] * (d[0] - (p_cam[0] / pz) * d[2]),
                  K[1, 1] * (d[1] - (p_cam[1] / pz) * d[2])])
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([1.0, 0.0])


def flank_of(alpha: float) -> str:
    return "L" if math.sin(alpha) > 0 else "R"



def cuboid_edges(corners: np.ndarray):
    """The 12 edges of a cuboid from its 8 corners, without assuming a corner ordering.
    Every vertex of a box has exactly 3 neighbours (its 3 axis-aligned neighbours), so joining each
    corner to its 3 nearest peers yields exactly the 12 edges whatever order the corners arrive in."""
    D = np.linalg.norm(corners[:, None, :] - corners[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    E = set()
    for i in range(8):
        for j in np.argsort(D[i])[:3]:
            E.add((min(i, int(j)), max(i, int(j))))
    return sorted(E)


def draw_cuboid(ax, corners_cam, K, colour, lw=1.0, alpha=0.85):
    """Project the 8 camera-space corners and stroke the 12 edges."""
    c = np.asarray(corners_cam, float)
    if c.shape != (8, 3) or np.any(c[:, 2] <= 1e-6):
        return False
    uv = (K @ c.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    for i, j in cuboid_edges(c):
        ax.plot([uv[i, 0], uv[j, 0]], [uv[i, 1], uv[j, 1]],
                color=colour, lw=lw, alpha=alpha, solid_capstyle="round")
    return True


# ---------------------------------------------------------------------------------------------------
def draw(ax, box, alpha, p_cam, up_cam, K, colour, label, *, lw=2.0, corners=None, mode='both'):
    import matplotlib.patches as mp

    x1, y1, x2, y2 = box
    usable = abs(math.sin(alpha)) >= USABLE_SIN
    if mode in ("2d", "both"):
        ax.add_patch(mp.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                  edgecolor=colour, linewidth=lw,
                                  linestyle="-" if usable else (0, (4, 3))))
    if mode in ("3d", "both") and corners is not None:
        draw_cuboid(ax, corners, K, colour, lw=lw * 0.7)

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    d = allocentric_dir(alpha, np.asarray(p_cam, float), np.asarray(up_cam, float))
    u = project_dir(d, np.asarray(p_cam, float), K)
    L = 0.55 * max(x2 - x1, y2 - y1)
    ax.annotate("", xy=(cx + u[0] * L, cy + u[1] * L), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7",
                                color=colour, lw=lw,
                                linestyle="-" if usable else (0, (4, 3))))
    ax.text(x1, y1 - 4, f"{label} {flank_of(alpha)} |sin|={abs(math.sin(alpha)):.2f}"
                        f"{'' if usable else ' END-ON'}",
            color=colour, fontsize=7, weight="bold",
            bbox=dict(facecolor="black", alpha=0.55, pad=1, edgecolor="none"))


# ---------------------------------------------------------------------------------------------------
def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True, help="labelled WildBox json (build_heading_labels.py output)")
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--preds", type=Path, default=None, help="instances_predictions.pth (optional)")
    ap.add_argument("--video", default=None, help="restrict to one video")
    ap.add_argument("--n", type=int, default=12, help="how many frames to render")
    ap.add_argument("--score-thresh", type=float, default=0.25)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labelled-only", action="store_true", default=True,
                    help="only render frames that carry at least one heading label")
    ap.add_argument("--crops", action="store_true",
                    help="ALSO emit a zoomed per-animal contact sheet. Animals are ~142 px in a 1920x1080 "
                         "frame, so a full-frame render cannot show a heading -- this is the mode you "
                         "actually verify with.")
    ap.add_argument("--crop-pad", type=float, default=1.8, help="crop size as a multiple of the box")
    ap.add_argument("--crop-cols", type=int, default=6)
    ap.add_argument("--no-context", action="store_true",
                    help="do NOT draw the unlabelled GT animals. Only ~11%% of val annotations carry a "
                         "heading label, so without context boxes most animals in a frame are unmarked and "
                         "the eye lands on an unboxed one and reads the picture as misaligned.")
    ap.add_argument("--boxes", choices=["2d", "3d", "both"], default="both",
                    help="2D box, projected 3D cuboid, or both. The 3D cuboid is the detector's actual "
                         "output; the 2D box is what localises the animal for the badge. NOTE the cuboid's "
                         "own front/back is meaningless (PCA sign is arbitrary) -- the ARROW is the heading.")
    args = ap.parse_args()

    d = json.loads(args.gt.read_text())
    images = {im["id"]: im for im in d["images"]}
    anns = defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)

    cand = []
    for iid, al in anns.items():
        im = images[iid]
        if args.video and args.video not in im["file_path"]:
            continue
        if args.labelled_only and not any(x.get("heading_valid", 0) for x in al):
            continue
        cand.append(iid)
    cand.sort()
    if not cand:
        print("no frames matched (try dropping --video, or check --labelled-only)")
        return 1
    pick = [cand[i] for i in np.linspace(0, len(cand) - 1, min(args.n, len(cand))).astype(int)]
    print(f"{len(cand)} candidate frames; rendering {len(pick)}")

    preds = None
    if args.preds:
        import torch
        raw = torch.load(args.preds, map_location="cpu")
        preds = defaultdict(list)
        for r in raw:
            for inst in r["instances"]:
                preds[r["image_id"]].append(inst)
        print(f"loaded predictions for {len(preds)} images")

    args.out.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for iid in pick:
        im = images[iid]
        path = args.image_root / im["file_path"]
        if not path.is_file():
            print(f"  missing image: {path}")
            continue
        K = np.array(im["K"], dtype=float).reshape(3, 3)
        arr = np.asarray(Image.open(path).convert("RGB"))

        fig, ax = plt.subplots(figsize=(im["width"] / 140, im["height"] / 140), dpi=140)
        ax.imshow(arr)
        ax.set_axis_off()

        # context: every OTHER annotated animal, dim, so an unlabelled animal is never mistaken for
        # a misplaced box (only ~11% of val annotations carry a heading label)
        if not args.no_context:
            import matplotlib.patches as mp
            for a in anns[iid]:
                if a.get("heading_valid", 0):
                    continue
                x, y, w, h = a["bbox"]
                ax.add_patch(mp.Rectangle((x, y), w, h, fill=False,
                                          edgecolor="#7FB2FF", linewidth=0.9, alpha=0.55))
                ax.text(x, y - 3, "unlabelled", color="#7FB2FF", fontsize=5.5, alpha=0.85)

        for a in anns[iid]:
            if not a.get("heading_valid", 0):
                continue
            up = -np.array(a["R_cam"], dtype=float)[:, 1]     # NOTE the minus: R_cam[:,1] is -up
            x, y, w, h = a["bbox"]
            draw(ax, (x, y, x + w, y + h), float(a["heading_alpha"]),
                 a["center_cam"], up, K, "#39FF6A", "GT",
                 corners=a.get("bbox3D_cam"), mode=args.boxes)

        if preds is not None:
            up_ref = None
            for a in anns[iid]:
                up_ref = -np.array(a["R_cam"], dtype=float)[:, 1]
                break
            for p in preds.get(iid, []):
                if p.get("score", 0) < args.score_thresh or "alpha" not in p:
                    continue
                x, y, w, h = p["bbox"]
                up_p = -np.array(p["pose"], dtype=float)[:, 1] if "pose" in p else up_ref
                draw(ax, (x, y, x + w, y + h), float(p["alpha"]),
                     p["center_cam"], up_p, K, "#FF3DCB", f"P{p['score']:.2f}", lw=1.6,
                     corners=p.get("bbox3D"), mode=args.boxes)

        name = "__".join(im["file_path"].replace("\\", "/").split("/")[-3:]).replace(".jpg", "")
        fig.savefig(args.out / f"{name}.png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        n_ok += 1

    print(f"wrote {n_ok} frames -> {args.out}")

    # ---- zoomed per-animal contact sheet ---------------------------------------------------------
    if args.crops:
        cells = []
        for iid in pick:
            im = images[iid]
            path = args.image_root / im["file_path"]
            if not path.is_file():
                continue
            K = np.array(im["K"], dtype=float).reshape(3, 3)
            arr = np.asarray(Image.open(path).convert("RGB"))
            for a in anns[iid]:
                if not a.get("heading_valid", 0):
                    continue
                x, y, w, h = a["bbox"]
                cells.append((arr, (x, y, x + w, y + h), float(a["heading_alpha"]),
                              a["center_cam"], -np.array(a["R_cam"], float)[:, 1], K,
                              a.get("category_name", "?")))
        if cells:
            cols = args.crop_cols
            rows = int(np.ceil(len(cells) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), dpi=150)
            axes = np.atleast_1d(axes).ravel()
            for ax in axes:
                ax.set_axis_off()
            for ax, (arr, box, alpha, p, up, K, name) in zip(axes, cells):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                s = max(x2 - x1, y2 - y1) * args.crop_pad / 2
                X1, Y1 = int(max(0, cx - s)), int(max(0, cy - s))
                X2, Y2 = int(min(arr.shape[1], cx + s)), int(min(arr.shape[0], cy + s))
                ax.imshow(arr[Y1:Y2, X1:X2])
                d = allocentric_dir(alpha, np.asarray(p, float), up)
                u = project_dir(d, np.asarray(p, float), K)
                ccx, ccy = cx - X1, cy - Y1
                L = 0.42 * min(X2 - X1, Y2 - Y1)
                usable = abs(math.sin(alpha)) >= USABLE_SIN
                ax.annotate("", xy=(ccx + u[0] * L, ccy + u[1] * L), xytext=(ccx, ccy),
                            arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",
                                            color="#39FF6A", lw=2.2,
                                            linestyle="-" if usable else (0, (3, 2))))
                ax.set_title(f"{name[:9]} {flank_of(alpha)} {abs(math.sin(alpha)):.2f}"
                             f"{'' if usable else ' END'}", fontsize=6.5,
                             color="black" if usable else "darkred")
            sheet = args.out / "contact_sheet.png"
            fig.tight_layout()
            fig.savefig(sheet, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            print(f"wrote a {len(cells)}-animal contact sheet -> {sheet}")
    print("\nHOW TO READ IT:")
    print("  arrowhead = the animal's HEAD.  green = ground truth, magenta = prediction.")
    print("  dashed = |sin alpha| < 0.35, i.e. near end-on: the flank bit is degenerate there,")
    print("           so do NOT count a left/right disagreement on a dashed arrow as an error.")
    print("  a SYSTEMATIC 180-degree error shows up as arrows pointing at the tail across many animals --")
    print("  that is the failure no metric in this repo can see.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

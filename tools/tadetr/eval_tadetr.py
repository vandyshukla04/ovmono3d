"""TA-DETR inference -> instances_predictions.pth (the eval boundary).  [M3]

    python tools/tadetr/eval_tadetr.py \
        --config configs/tadetr/a1.yaml,configs/tadetr/local_paths.yaml \
        --ckpt <run_dir>/epoch_15.pth --out <run_dir>/preds/WildBox_val.pth \
        [--mode own|gt2d] [--score-thresh 0.05] [--limit-images N]

Modes:
  own   -- the model's own 300 queries; score = class prob; top detections above threshold.
  gt2d  -- box-conditioned oracle parity (one query per GT 2D box, reference at the box
           bottom-center; class/score/bbox from the oracle; 3D heads predicted) -- the swap-in
           protocol that isolates 3D-stage gains, mirroring the cubercnn oracle merge semantics.

Records match tools/ovmono3d_geo.py exactly (bbox3D corners via cubercnn get_cuboid_verts_faces;
`alpha` included -- the evaluator whitelist carries it). category_id written as DATASET ids
(1000..1005) for consistency with the A0 rows. After this, run (printed at the end):
  tools/eval_ovmono3d_geo.py (official NHD/AP)  + grade_detection_sane.py + bev_ap_eval.py
  + grade_orientation.py (the flip-sensitive heading grader; alpha is in the records)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cubercnn.util import get_cuboid_verts_faces  # noqa: E402
from tadetr.config import TADETRConfig  # noqa: E402
from tadetr.data.dataset import WildBoxTADETR, collate  # noqa: E402
from tadetr.geometry.heightfield import TerrainField  # noqa: E402
from tadetr.modeling.detector import TADETR  # noqa: E402
from tadetr.utils.rot import compose_pose, wrap  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["own", "gt2d"], default="own")
    ap.add_argument("--score-thresh", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit-images", type=int, default=0)
    args = ap.parse_args()

    cfg = TADETRConfig.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = WildBoxTADETR(cfg.data, "val", training=False)
    inv_cat = {v: k for k, v in ds.cat_map.items()}          # contiguous -> dataset id

    stats = json.loads((Path(__file__).resolve().parent / "data"
                        / "a0_class_stats.json").read_text())
    meta = json.loads(Path(cfg.data.category_meta).read_text())
    dims_median = torch.tensor([stats["classes"][n]["dims_median"]
                                for n in meta["thing_classes"]], dtype=torch.float32)
    model = TADETR(cfg, dims_median, n_classes=len(meta["thing_classes"])).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    state = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected[:5]
    assert all(k.startswith("backbone.vit.") for k in missing), missing[:5]
    model.eval()
    print(f"loaded {args.ckpt} (epoch {ck.get('epoch', '?')}, iter {ck.get('iter', '?')}); "
          f"mode={args.mode}")

    idxs = list(range(len(ds)))
    if args.limit_images:
        idxs = idxs[:args.limit_images]
    inst_by_img = defaultdict(list)
    W_in, H_in = cfg.data.input_w, cfg.data.input_h

    with torch.no_grad():
        for bstart in range(0, len(idxs), args.batch):
            chunk = [ds[i] for i in idxs[bstart:bstart + args.batch]]
            batch = collate(chunk)
            for k in ("image", "bridge", "K", "extrinsic", "cam_height", "telemetry",
                      "cam_feats"):
                batch[k] = batch[k].to(device)
            fields = {k: TerrainField(ds.cache(k), device=device, dtype=torch.float32)
                      for k in set(batch["seg_key"])}
            ref_boxes = None
            if args.mode == "gt2d":
                nmax = max(len(t["cls"]) for t in batch["targets"]) or 1
                ref_boxes = torch.zeros(len(chunk), nmax, 4, device=device)
                for i, t in enumerate(batch["targets"]):
                    if len(t["cls"]):
                        ref_boxes[i, :len(t["cls"])] = t["boxes"].to(device)
            outputs = model(batch, fields, stage=3, ref_boxes=ref_boxes)
            out = outputs[-1]

            prob = out["logits"].float().softmax(-1)[..., :-1]        # (B,Q,C)
            score, cls = prob.max(-1)
            psi = 0.5 * torch.atan2(out["axis"][..., 0].float(), out["axis"][..., 1].float())
            sflip = (out["sign_logit"].float().sigmoid() > 0.5).float()
            for i, s in enumerate(chunk):
                im_id = s["image_id"]
                br = batch["bridge"][i].cpu().numpy()                 # identity at eval, but honor
                K = batch["K"][i].cpu().numpy()
                if args.mode == "gt2d":
                    tg = batch["targets"][i]
                    qsel = torch.arange(len(tg["cls"]), device=device)
                    if len(qsel) == 0:
                        continue
                    scores_i = torch.ones(len(qsel))
                    cls_i = tg["cls"].to(device)
                else:
                    keep = score[i] >= args.score_thresh
                    qsel = keep.nonzero()[:, 0]
                    if len(qsel) > args.topk:
                        qsel = qsel[score[i][qsel].argsort(descending=True)[:args.topk]]
                    scores_i = score[i][qsel].cpu()
                    cls_i = cls[i][qsel]
                cc = out["center_cam"][i][qsel].float()
                dm = out["dims"][i][qsel].float()
                up = out["surface_up_cam"][i][qsel].float()
                ps = psi[i][qsel]
                if br[4] > 0.5:                                       # flipped view (not at eval)
                    ps = -ps
                R = compose_pose(up, ps, cc, sign_flip=sflip[i][qsel])
                alpha = wrap(ps + torch.pi * sflip[i][qsel])
                bx = out["boxes"][i][qsel].float().cpu().numpy()      # cxcywh view-normalized
                u1 = (bx[:, 0] - bx[:, 2] / 2) * W_in
                v1 = (bx[:, 1] - bx[:, 3] / 2) * H_in
                fu = br[2] + br[0] * u1
                fv = br[3] + br[1] * v1
                fw = bx[:, 2] * W_in * br[0]
                fh = bx[:, 3] * H_in * br[1]
                if args.mode == "gt2d":                               # oracle bbox verbatim
                    tgb = batch["targets"][i]["boxes"].cpu().numpy()
                    fu = (tgb[:, 0] - tgb[:, 2] / 2) * W_in * br[0] + br[2]
                    fv = (tgb[:, 1] - tgb[:, 3] / 2) * H_in * br[1] + br[3]
                    fw = tgb[:, 2] * W_in * br[0]
                    fh = tgb[:, 3] * H_in * br[1]
                for j in range(len(qsel)):
                    c = cc[j].cpu().numpy()
                    d = dm[j].cpu().numpy()
                    Rj = R[j].cpu().numpy()
                    verts, _ = get_cuboid_verts_faces(
                        box3d=[float(c[0]), float(c[1]), float(c[2]),
                               float(d[0]), float(d[1]), float(d[2])],
                        R=torch.tensor(Rj, dtype=torch.float32))
                    uvz = K @ c
                    cid = int(cls_i[j])
                    inst_by_img[im_id].append({
                        "image_id": im_id,
                        "category_id": inv_cat[cid],
                        "category_name": meta["thing_classes"][cid],
                        "bbox": [float(fu[j]), float(fv[j]), float(fw[j]), float(fh[j])],
                        "score": float(scores_i[j]),
                        "depth": float(c[2]),
                        "bbox3D": (verts.numpy() if hasattr(verts, "numpy")
                                   else np.asarray(verts)).tolist(),
                        "center_cam": c.tolist(),
                        "center_2D": [float(uvz[0] / uvz[2]), float(uvz[1] / uvz[2])],
                        "dimensions": d.tolist(),
                        "pose": Rj.tolist(),
                        "alpha": float(alpha[j]),
                    })
            if (bstart // args.batch) % 50 == 0:
                print(f"  {bstart + len(chunk)}/{len(idxs)} images", flush=True)

    dataset = []
    kept_ids = {ds.samples[i][0]["id"] for i in idxs}
    for im, _, _ in ds.samples:
        if im["id"] not in kept_ids:
            continue
        rec = dict(im)
        rec["image_id"] = im["id"]
        rec["instances"] = inst_by_img.get(im["id"], [])
        dataset.append(rec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.out)
    n = sum(len(v) for v in inst_by_img.values())
    print(f"\nwrote {n:,} detections over {len(dataset):,} images -> {args.out}")
    print("\nNext (the gate protocol):")
    print(f"  grade_detection_sane.py --val {cfg.data.val_json} --preds a1={args.out}")
    print(f"  eval_ovmono3d_geo.evaluate_predictions on {args.out} "
          f"(category_path=configs/wildbox/category_meta_wildlife6.json)")
    print(f"  bev_ap_eval.py --preds {args.out} --gt <val json>   (from the category-meta work dir)")
    print(f"  grade_orientation.py --preds {args.out} --val <val> --train <train>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

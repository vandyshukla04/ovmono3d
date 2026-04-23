#!/usr/bin/env python
"""Pre-compute GroundingDINO 2D detections for WildBox to match the
OVMono3D paper's zero-shot evaluation protocol.

At test time, OVMono3D's paper uses text-prompted GroundingDINO as an
open-vocabulary 2D detector, then lifts those boxes to 3D via the cube
head. `TEST.ORACLE2D=True` looks up precomputed detections from a
dataset-specific JSON at `datasets/Omni3D/gdino_<name>_oracle_2d.json`.

This script produces that JSON for any Omni3D-format dataset, given
a list of category text prompts.

The output format matches the one consumed by the evaluator (see
`cubercnn/data/build.py:_test_loader_from_config` `oracle2d=[...]`
argument, which is then wired through `load_omni3d_json`) — one entry
per image, with per-image `instances` in the same shape as prediction
instances produced by `train_net.py`.

Usage:
    python tools/precompute_gdino_oracle.py \\
        --gt           datasets/Omni3D/WildBox_val.json \\
        --out          datasets/Omni3D/gdino_WildBox_val_oracle_2d.json \\
        --species      rhino elephant zebra giraffe gazelle \\
        --checkpoint   checkpoints/groundingdino_swinb_cogcoor.pth \\
        --config       configs/GroundingDINO_SwinB_cfg.py \\
        --box-threshold 0.25 --text-threshold 0.20 \\
        --device cpu                          # cuda if GDino CUDA kernels work
        --limit 0                             # 0 = process all; >0 = subset for testing

Notes on the CPU path:
  Our cluster's GroundingDINO CUDA ops (ms_deform_attn) failed to build,
  so the library falls back to a slow CPU implementation. Expect ~5-15
  seconds per image. For 13k images this is 24-36 hours. Subsample via
  `--limit` for smoke tests before committing to a full run.

Output JSON schema (list, one entry per image):
    [
      {
        "image_id": <int>,
        "K": [[3x3 intrinsics]],            # copied from GT json
        "file_path": "...",                 # copied from GT json
        "height": H, "width": W,
        "instances": [
          {
            "bbox": [x, y, w, h],           # xywh in pixels
            "score": <float>,
            "category_id": <dataset_id>,    # e.g., 1004 for rhino
            "category_name": "rhino"
          },
          ...
        ]
      },
      ...
    ]
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("precompute_gdino_oracle")


def load_gdino(config_path: Path, ckpt_path: Path, device: str):
    """Load a GroundingDINO model. Imports are inside the function so the
    script can at least show --help on systems without GDino installed."""
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    # BERT text encoder lives inside groundingdino via transformers:
    # no separate init needed.

    args_cfg = SLConfig.fromfile(str(config_path))
    args_cfg.device = device
    # Disable gradient checkpointing for inference — saves nothing without backward
    # and roughly halves per-image latency when using the pure-Python
    # ms_deform_attn fallback (see also the torch.utils.checkpoint warning).
    if hasattr(args_cfg, "use_checkpoint"):
        args_cfg.use_checkpoint = False
    model = build_model(args_cfg)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    load_info = model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    logger.info(f"GDino checkpoint loaded: missing={len(load_info.missing_keys)} "
                f"unexpected={len(load_info.unexpected_keys)}")
    model = model.to(device).eval()

    # Belt-and-suspenders: the `use_checkpoint` cfg flag only guards some layers;
    # the BERT text encoder and transformer stages still flip the attr at runtime.
    # Walk the tree and force it off.
    for m in model.modules():
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = False
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = False
    return model


def preprocess_image(img_path: Path, device: str):
    """GroundingDINO expects a normalized 3xHxW tensor. Replicate their
    util.inference.load_image (which uses PIL + torchvision transforms),
    staying framework-agnostic."""
    from PIL import Image
    import torchvision.transforms.functional as F

    im = Image.open(img_path).convert("RGB")
    # GDino's default: normalize with ImageNet stats after resize to min-side 800
    # We keep native resolution to match what downstream uses; GDino internally
    # resizes to its expected range.
    t = F.pil_to_tensor(im).float() / 255.0
    t = F.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return t.to(device), im.size[::-1]  # (C,H,W), (H,W)


def run_one(model, image_tensor, text_prompt: str,
            box_threshold: float, text_threshold: float,
            device: str):
    """One forward pass producing (boxes [N,4] cxcywh-normalized,
    logits [N, num_tokens]). Ports groundingdino.util.inference.predict
    inline to avoid its file-only input path."""
    # GDino expects prompt to be lowercase ending with "."
    prompt = text_prompt.lower().strip()
    if not prompt.endswith("."):
        prompt += "."

    # inference_mode is stricter than no_grad: it also bypasses
    # torch.utils.checkpoint's recomputation path entirely.
    with torch.inference_mode():
        outputs = model(image_tensor.unsqueeze(0), captions=[prompt])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (N, n_tokens)
    boxes = outputs["pred_boxes"].cpu()[0]              # (N, 4) cxcywh norm

    # Filter by max-score across tokens
    mask = logits.max(dim=1).values > box_threshold
    logits_kept = logits[mask]
    boxes_kept = boxes[mask]

    return boxes_kept, logits_kept


def token_span_for_name(tokenizer, caption: str, name: str):
    """Return (start_token_idx, end_token_idx_exclusive) for `name` inside
    `caption` after BERT tokenization. Used to map logits (one per token)
    back to categories. Matches groundingdino.util.vl_utils."""
    # We build the span by finding character offsets then mapping to token
    # offsets via the tokenizer. This is robust to re-tokenization quirks.
    cap = caption.lower()
    nm = name.lower()
    char_start = cap.find(nm)
    if char_start < 0:
        return None
    char_end = char_start + len(nm)
    enc = tokenizer(caption, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    t_start = t_end = None
    for i, (a, b) in enumerate(offsets):
        if a <= char_start < b and t_start is None:
            t_start = i
        if a < char_end <= b:
            t_end = i + 1
            break
    if t_start is None or t_end is None:
        return None
    return (t_start, t_end)


def cxcywh_norm_to_xywh_pixels(boxes_n, img_w, img_h):
    """GDino returns cxcywh normalized to [0, 1]."""
    cx, cy, w, h = boxes_n.unbind(-1)
    cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
    x1 = cx - w / 2
    y1 = cy - h / 2
    return torch.stack([x1, y1, w, h], dim=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", type=Path, required=True,
                   help="Omni3D-format GT json, e.g. datasets/Omni3D/WildBox_val.json")
    p.add_argument("--out", type=Path, required=True,
                   help="Output oracle JSON path, e.g. datasets/Omni3D/gdino_WildBox_val_oracle_2d.json")
    p.add_argument("--species", nargs="+", required=True,
                   help="Text prompts per category; must match GT category names exactly")
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/groundingdino_swinb_cogcoor.pth"))
    p.add_argument("--config", type=Path,
                   default=Path("configs/GroundingDINO_SwinB_cfg.py"))
    p.add_argument("--box-threshold", type=float, default=0.15,
                   help="Min confidence for a detected box. Default 0.15 is tuned "
                        "for distant drone wildlife; GDino's official default of "
                        "0.25 suppresses most small animals at altitude.")
    p.add_argument("--text-threshold", type=float, default=0.10,
                   help="Min per-token confidence for class assignment. "
                        "Default 0.10 (vs paper 0.25) keeps ambiguous "
                        "species assignments that the 3D head can still refine.")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--fp16", action="store_true",
                   help="Run the model in fp16 on CUDA. ~2x faster on A40 with "
                        "no measurable recall loss in our tests.")
    p.add_argument("--limit", type=int, default=0,
                   help=">0 = precompute only the first N images (for testing)")
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Sanity
    if not args.checkpoint.exists():
        sys.exit(f"GDino checkpoint not found at {args.checkpoint} — "
                 f"uncomment the groundingdino wget in setup.sh and rerun")
    if not args.config.exists():
        sys.exit(f"GDino config not found at {args.config}")

    logger.info(f"Loading GT: {args.gt}")
    gt = json.load(open(args.gt))
    name_to_id = {c["name"]: c["id"] for c in gt["categories"]}
    for s in args.species:
        if s not in name_to_id:
            sys.exit(f"--species '{s}' not in GT categories "
                     f"{list(name_to_id.keys())}")

    caption = ". ".join(args.species) + "."
    logger.info(f"Prompt: {caption!r}")

    # Load model
    logger.info(f"Loading GroundingDINO on device={args.device} fp16={args.fp16}")
    model = load_gdino(args.config, args.checkpoint, args.device)
    if args.fp16:
        if args.device != "cuda":
            sys.exit("--fp16 requires --device cuda")
        model = model.half()

    # Build the tokenizer we need for per-category token spans
    from groundingdino.util.get_tokenlizer import get_tokenlizer
    tokenizer = get_tokenlizer("bert-base-uncased")

    # Pre-compute token spans per species
    spans = {}
    for s in args.species:
        sp = token_span_for_name(tokenizer, caption, s)
        if sp is None:
            sys.exit(f"Couldn't find token span for '{s}' in caption {caption!r}")
        spans[s] = sp
    logger.info(f"Token spans: {spans}")

    images = gt["images"]
    if args.limit > 0:
        images = images[: args.limit]
    logger.info(f"Processing {len(images)} images")

    out_rows = []
    t0 = time.time()
    kept_total = 0
    # Log the first few images individually so slow startup is visible; the
    # --log-every cadence kicks in after that.
    first_few_logged = 0

    for i, img in enumerate(images):
        img_path = Path(img["file_path"])
        if not img_path.is_absolute():
            img_path = Path("datasets") / img_path
        if not img_path.exists():
            logger.warning(f"Skipping missing image: {img_path}")
            out_rows.append({
                "image_id": img["id"],
                "K": img["K"], "file_path": str(img["file_path"]),
                "height": img["height"], "width": img["width"],
                "instances": [],
            })
            continue

        try:
            image_tensor, (H0, W0) = preprocess_image(img_path, args.device)
            if args.fp16:
                image_tensor = image_tensor.half()
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            out_rows.append({
                "image_id": img["id"],
                "K": img["K"], "file_path": str(img["file_path"]),
                "height": img["height"], "width": img["width"],
                "instances": [],
            })
            continue

        boxes_n, logits = run_one(model, image_tensor, caption,
                                  box_threshold=args.box_threshold,
                                  text_threshold=args.text_threshold,
                                  device=args.device)

        instances = []
        if boxes_n.numel() > 0:
            boxes_xywh = cxcywh_norm_to_xywh_pixels(
                boxes_n, img_w=W0, img_h=H0).numpy()
            for bi in range(boxes_n.shape[0]):
                # Per-token logits: (n_tokens,). Species score = max logit in
                # its token span.
                per_species = {
                    s: float(logits[bi, ts[0]:ts[1]].max().item())
                    for s, ts in spans.items()
                }
                # Assign box to top-scoring species if above text threshold
                best_sp, best_score = max(per_species.items(),
                                          key=lambda x: x[1])
                if best_score < args.text_threshold:
                    continue
                x, y, w, h = boxes_xywh[bi]
                if w <= 1 or h <= 1:
                    continue
                instances.append({
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "score": best_score,
                    "category_id": int(name_to_id[best_sp]),
                    "category_name": best_sp,
                })
                kept_total += 1

        out_rows.append({
            "image_id": img["id"],
            "K": img["K"],
            "file_path": str(img["file_path"]),
            "height": img["height"],
            "width": img["width"],
            "instances": instances,
        })

        if first_few_logged < 3:
            elapsed = time.time() - t0
            logger.info(f"[{i+1}/{len(images)}] kept={kept_total} "
                        f"per-img={(elapsed/(i+1)):.2f}s "
                        f"(warmup — expect first image to be slower)")
            first_few_logged += 1

        if (i + 1) % args.log_every == 0 or i == len(images) - 1:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta = rate * (len(images) - i - 1)
            logger.info(f"[{i+1}/{len(images)}] "
                        f"kept={kept_total} "
                        f"rate={rate:.2f}s/img "
                        f"elapsed={elapsed/60:.1f}m "
                        f"eta={eta/60:.1f}m")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_rows, f, indent=2)
    logger.info(f"Wrote {args.out}  ({len(out_rows)} images, "
                f"{kept_total} boxes total)")


if __name__ == "__main__":
    main()

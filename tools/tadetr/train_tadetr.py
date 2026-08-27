"""TA-DETR training loop.  [M3]

    python tools/tadetr/train_tadetr.py --config configs/tadetr/a1.yaml \
        [--resume] [override.dotted=value ...]

Plain PyTorch (no detectron2). AMP: bf16 on A40 with the composition running in an fp32 island
(inside the detector); train.amp=off => full fp32 (the V100 recipe -- NEVER fp16 on Volta).
Resume-safe: checkpoints carry model/opt/sched/epoch/iter; --resume picks up last.pth.
Curriculum stages (spec 4.5) switch loss groups by epoch via criterion(stage=...).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tadetr.config import TADETRConfig  # noqa: E402
from tadetr.data.dataset import WildBoxTADETR, collate  # noqa: E402
from tadetr.data.samplers import SegmentBatchSampler  # noqa: E402
from tadetr.geometry.heightfield import TerrainField  # noqa: E402
from tadetr.modeling.criterion import TADETRCriterion  # noqa: E402
from tadetr.modeling.detector import TADETR  # noqa: E402


def build_fields(dataset, keys, device, cache={}):
    for k in keys:
        if k not in cache:
            cache[k] = TerrainField(dataset.cache(k), device=device, dtype=torch.float32)
    return cache


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = TADETRConfig.load(args.config, args.overrides)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(cfg.dump())
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = WildBoxTADETR(cfg.data, "train", training=True)
    sampler = SegmentBatchSampler(ds.by_segment, cfg.data.segments_per_batch,
                                  cfg.data.frames_per_segment, seed=cfg.train.seed)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                                         num_workers=cfg.data.num_workers, pin_memory=True)

    import json as _json
    stats = _json.loads(Path("tools/tadetr/data/a0_class_stats.json").read_text())
    meta = _json.loads(Path(cfg.data.category_meta).read_text())
    dims_median = torch.tensor([stats["classes"][name]["dims_median"]
                                for name in meta["thing_classes"]], dtype=torch.float32)
    model = TADETR(cfg, dims_median, n_classes=len(meta["thing_classes"])).to(device)
    criterion = TADETRCriterion(cfg.loss, n_classes=len(meta["thing_classes"]))

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_train/1e6:.1f}M (backbone frozen)")
    opt = torch.optim.AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    iters_per_epoch = len(sampler)
    # the schedule steps once per OPTIMIZER step (= loader iters / accum), so the horizon must be
    # counted in optimizer steps or cosine never completes (found in the pre-launch audit)
    total_opt_steps = max(cfg.train.epochs * iters_per_epoch // cfg.train.accum_steps, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / cfg.train.warmup_iters)
        * 0.5 * (1 + math.cos(math.pi * min(it / total_opt_steps, 1.0))))

    def save_ckpt(path, epoch, it):
        # frozen backbone excluded: it is 1.2 GB of redundant weights per file; resume rebuilds it
        # from train.backbone_weights and loads the rest with strict=False
        slim = {k: v for k, v in model.state_dict().items()
                if not k.startswith("backbone.vit.")}
        torch.save({"model": slim, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "iter": it, "cfg": cfg.dump()}, path)

    start_epoch = git = 0
    ckpt_path = out_dir / "last.pth"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        assert not unexpected, f"unexpected ckpt keys: {unexpected[:5]}"
        assert all(k.startswith("backbone.vit.") for k in missing), \
            f"missing non-backbone keys: {[k for k in missing if not k.startswith('backbone.vit.')][:5]}"
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, git = ck["epoch"], ck["iter"]
        print(f"resumed from {ckpt_path} at epoch {start_epoch}, iter {git}")

    use_amp = cfg.train.amp == "bf16" and device == "cuda"
    print(f"amp={'bf16' if use_amp else 'off'}; {iters_per_epoch} iters/epoch x "
          f"{cfg.train.epochs} epochs; accum {cfg.train.accum_steps}")

    model.train()
    t0 = time.time()
    for epoch in range(start_epoch, cfg.train.epochs):
        sampler.set_epoch(epoch)
        stage = 1 if epoch < cfg.train.stage2_epoch else \
            (2 if epoch < cfg.train.stage3_epoch else 3)
        for bi, batch in enumerate(loader):
            for k in ("image", "bridge", "K", "extrinsic", "cam_height", "telemetry",
                      "cam_feats"):
                batch[k] = batch[k].to(device, non_blocking=True)
            batch["targets"] = [{k: (v.to(device) if torch.is_tensor(v) else v)
                                 for k, v in t.items()} for t in batch["targets"]]
            fields = build_fields(ds, set(batch["seg_key"]), device)
            for k in set(batch["seg_key"]):     # terrain dropout: degrade to the plane (edge tier)
                fields[k].plane_only = bool(torch.rand(1).item() < cfg.data.terrain_dropout_p)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(batch, fields, stage=stage)
            losses = criterion(outputs, batch["targets"], stage=stage)
            loss = sum(losses.values())
            (loss / cfg.train.accum_steps).backward()
            if (bi + 1) % cfg.train.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.train.clip_grad)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
            git += 1
            if git % cfg.train.log_every == 0:
                main_losses = {k: round(float(v), 4) for k, v in losses.items()
                               if k.startswith("loss_")}   # final layer; aux are l<i>_loss_*
                print(f"e{epoch} it{git} stage{stage} loss {float(loss):.3f} "
                      f"{main_losses} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time()-t0)/max(git,1):.2f}s/it", flush=True)
            if git % cfg.train.ckpt_every_iters == 0:
                save_ckpt(ckpt_path, epoch, git)
        save_ckpt(ckpt_path, epoch + 1, git)
        save_ckpt(out_dir / f"epoch_{epoch+1}.pth", epoch + 1, git)
        print(f"epoch {epoch+1} done -> {ckpt_path}", flush=True)
    print("training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

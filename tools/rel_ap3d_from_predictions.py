"""Run only the Rel-AP3D scale search on cached instances_predictions.pth.
Skips the model inference pass entirely — works on CPU in ~15 min per run.

Mirrors the `EVAL_REL_AP3D=True` code path in train_net.py but consumes
predictions from disk via Omni3DEvaluationHelper.add_predictions().

Usage (from repo root):
    python tools/rel_ap3d_from_predictions.py \\
        --run-dir output/wl6_zeroshot_oracle2d \\
        --config  output/wl6_zeroshot_oracle2d/config.yaml \\
        --gt      datasets/Omni3D/WildBox_val.json
"""
import argparse, json, os, sys
from pathlib import Path
import torch

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from cubercnn.config import get_cfg_defaults
from cubercnn.data import get_filter_settings_from_cfg
from cubercnn.evaluation import Omni3DEvaluationHelper
from cubercnn import util


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Run dir containing inference/iter_final/<dset>/instances_predictions.pth")
    ap.add_argument("--config",  type=Path, required=True,
                    help="The run's config.yaml (used for filter_settings + REL_AP3D_SEARCH).")
    ap.add_argument("--gt",      type=Path, required=True,
                    help="Omni3D-format GT JSON for the val dataset.")
    ap.add_argument("--dataset-name", default="WildBox_val")
    ap.add_argument("--out", type=Path, default=None,
                    help="Where to write log.rel.txt and updated paper_report. "
                         "Defaults to <run-dir>.")
    args = ap.parse_args()
    out = args.out or args.run_dir
    out.mkdir(parents=True, exist_ok=True)

    setup_logger(output=str(out), name="cubercnn")

    # Build cfg from the run's config.yaml — picks up REL_AP3D_SEARCH range
    cfg = get_cfg(); get_cfg_defaults(cfg)
    cfg.merge_from_file(str(args.config))
    cfg.MODEL.DEVICE = "cpu"
    cfg.TEST.EVAL_REL_AP3D = True

    # Setup category metadata so the evaluator knows the WildBox 6-species
    # mapping (must match what was active when predictions were generated).
    meta_path = args.run_dir / "category_meta.json"
    if not meta_path.exists():
        # fall back to repo-level
        meta_path = Path("configs/category_meta.json")
    meta = util.load_json(str(meta_path))
    thing_classes = meta["thing_classes"]
    id_map = {int(k): v for k, v in meta["thing_dataset_id_to_contiguous_id"].items()}
    MetadataCatalog.get("omni3d_model").thing_classes = thing_classes
    MetadataCatalog.get("omni3d_model").thing_dataset_id_to_contiguous_id = id_map
    print(f"[rel-ap3d] using {len(thing_classes)} classes: {thing_classes}", flush=True)

    # Load predictions
    pred_path = args.run_dir / "inference/iter_final" / args.dataset_name / "instances_predictions.pth"
    if not pred_path.exists():
        sys.exit(f"missing predictions at {pred_path}")
    print(f"[rel-ap3d] loading predictions from {pred_path}", flush=True)
    predictions = torch.load(str(pred_path), weights_only=False, map_location="cpu")

    filter_settings = get_filter_settings_from_cfg(cfg)
    filter_settings["category_names"] = thing_classes

    eval_helper = Omni3DEvaluationHelper(
        dataset_names=[args.dataset_name],
        filter_settings=filter_settings,
        output_folder=str(out),
        iter_label="final",
        only_2d=False,
        eval_categories=thing_classes,
    )

    # Materialise the dataset (registers metadata used by the evaluator)
    DatasetCatalog.get(args.dataset_name)

    eval_helper.add_predictions(args.dataset_name, predictions)
    print(f"[rel-ap3d] running Omni3D evaluator with EVAL_REL_AP3D=True ...", flush=True)
    eval_helper.evaluate(args.dataset_name)
    eval_helper.summarize_all()
    print(f"\n[rel-ap3d] DONE — results written under {out}/", flush=True)


if __name__ == "__main__":
    main()

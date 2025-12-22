#!/usr/bin/env python3
"""
Simple inference script for rhino detection model
"""

import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis
from pycocotools.coco import COCO
from tqdm import tqdm

def run_inference(args, cfg, model):
    """Run inference on a folder of images"""
    
    # Get list of images
    list_of_ims = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
        list_of_ims.extend(util.list_files(args.input_folder, ext))
    
    print(f"Found {len(list_of_ims)} images to process")
    
    model.eval()
    
    # Camera parameters
    focal_length = args.focal_length
    principal_point = args.principal_point
    threshold = args.threshold
    
    # Output setup
    output_dir = cfg.OUTPUT_DIR
    util.mkdir_if_missing(output_dir)
    
    # Image preprocessing
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])
    
    # Process each image
    for img_path in tqdm(list_of_ims):
        im_name = util.file_parts(img_path)[1]
        im = util.imread(img_path)
        
        if im is None:
            print(f"Could not read {img_path}")
            continue
        
        h, w = im.shape[:2]
        
        # Set up camera intrinsics
        if focal_length == 0:
            # Default: assume 90 degree FOV
            focal_length_ndc = 4.0
            focal_length = focal_length_ndc * h / 2
        
        if len(principal_point) == 0:
            px, py = w/2, h/2
        else:
            px, py = principal_point
        
        K = np.array([
            [focal_length, 0.0, px],
            [0.0, focal_length, py],
            [0.0, 0.0, 1.0]
        ])
        
        # Preprocess image
        aug_input = T.AugInput(im)
        _ = augmentations(aug_input)
        image = aug_input.image
        
        # Prepare batch (no category_list needed)
        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(),
            'height': h,
            'width': w,
            'K': K
        }]
        
        # Run inference
        with torch.no_grad():
            outputs = model(batched)
            instances = outputs[0]['instances']
        
        # Process detections
        meshes = []
        meshes_text = []
        n_det = len(instances)
        n_valid = 0
        
        if n_det > 0:
            for idx in range(n_det):
                score = instances.scores[idx].item()
                
                # Skip low confidence detections
                if score < threshold:
                    continue
                
                n_valid += 1
                
                # Extract 3D predictions
                center_cam = instances.pred_center_cam[idx].cpu().numpy()
                dimensions = instances.pred_dimensions[idx].cpu().numpy()
                pose = instances.pred_pose[idx].cpu().numpy()
                
                # Create 3D box mesh for visualization
                bbox3D = center_cam.tolist() + dimensions.tolist()
                meshes_text.append(f'rhino {score:.2f}')
                
                # Color for this detection
                color = [c/255.0 for c in util.get_color(idx)]
                box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                meshes.append(box_mesh)
        
        print(f'{im_name}: {n_det} detections, {n_valid} above threshold')
        
        # Visualize and save
        if len(meshes) > 0:
            # Draw 3D boxes on image and create top-down view
            im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(
                im, K, meshes, text=meshes_text, 
                scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85
            )
            
            # Combine views
            im_concat = np.concatenate((im_drawn_rgb, im_topdown), axis=1)
            
            # Display if requested
            if args.display:
                vis.imshow(im_concat)
            
            # Save outputs
            util.imwrite(im_concat, os.path.join(output_dir, f'{im_name}_result.jpg'))
            util.imwrite(im_drawn_rgb, os.path.join(output_dir, f'{im_name}_3dboxes.jpg'))
            util.imwrite(im_topdown, os.path.join(output_dir, f'{im_name}_topdown.jpg'))
            
            # Optionally save detection info
            if args.save_detections:
                det_info = {
                    'image': im_name,
                    'detections': []
                }
                for idx in range(len(meshes)):
                    det_info['detections'].append({
                        'score': float(meshes_text[idx].split()[-1]),
                        'center_cam': center_cam.tolist(),
                        'dimensions': dimensions.tolist(),
                        'pose': pose.tolist()
                    })
                util.save_json(os.path.join(output_dir, f'{im_name}_detections.json'), det_info)
        else:
            # No detections - save original image
            util.imwrite(im, os.path.join(output_dir, f'{im_name}_no_detections.jpg'))


def setup(args):
    """Create configs and perform basic setups"""
    cfg = get_cfg()
    get_cfg_defaults(cfg)
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Simple rhino 3D detection inference")
    
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument("--input-folder", required=True, help="folder containing images")
    parser.add_argument("--weights", required=True, help="path to model weights")
    parser.add_argument("--output-dir", default="output/inference", help="output directory")
    parser.add_argument("--threshold", type=float, default=0.3, help="detection threshold")
    parser.add_argument("--focal-length", type=float, default=0, help="focal length in pixels (0=auto)")
    parser.add_argument("--principal-point", type=float, nargs=2, default=[], help="principal point in pixels")
    parser.add_argument("--display", action="store_true", help="display results with matplotlib")
    parser.add_argument("--save-detections", action="store_true", help="save detection details as JSON")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                       help="modify config options using command line")
    
    args = parser.parse_args()
    
    # Setup config
    cfg = setup(args)
    cfg.defrost()
    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()
    
    # Build model
    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    
    # Run inference
    model.eval()
    with torch.no_grad():
        run_inference(args, cfg, model)
    
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
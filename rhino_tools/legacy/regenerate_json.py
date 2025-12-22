#!/usr/bin/env python3
"""
Regenerate RHINO JSON files from existing images with all necessary fixes
"""

import os
import json
import numpy as np
import glob
from tqdm import tqdm

def find_matched_pairs():
    """Find matched video-result directories"""
    video_base = "/home/shuklva/CUT3R/examples/wd_data/rhinos_cami"
    results_base = "/home/shuklva/CUT3R/results"
    
    video_dirs = sorted(glob.glob(os.path.join(video_base, "rhin-*")))
    video_mapping = {}
    for vdir in video_dirs:
        basename = os.path.basename(vdir)
        video_id = basename.replace("rhin-", "")
        video_mapping[video_id] = vdir
    
    result_dirs = sorted(glob.glob(os.path.join(results_base, "tmp-rhin-*")))
    result_mapping = {}
    for rdir in result_dirs:
        basename = os.path.basename(rdir)
        parts = basename.replace("tmp-rhin-", "").split("-revisit-")
        result_id = parts[0]
        revisit_num = int(parts[1])
        if result_id not in result_mapping or revisit_num == 1:
            result_mapping[result_id] = rdir
    
    matched_pairs = []
    for vid_id in video_mapping:
        if vid_id in result_mapping:
            matched_pairs.append((vid_id, video_mapping[vid_id], result_mapping[vid_id]))
    
    return matched_pairs

def compute_3d_corners(center, dimensions, rotation_matrix):
    """Compute 8 corners of 3D bounding box"""
    w, h, l = dimensions
    corners_local = np.array([
        [-w/2, -h/2, -l/2],
        [ w/2, -h/2, -l/2],
        [ w/2,  h/2, -l/2],
        [-w/2,  h/2, -l/2],
        [-w/2, -h/2,  l/2],
        [ w/2, -h/2,  l/2],
        [ w/2,  h/2,  l/2],
        [-w/2,  h/2,  l/2]
    ])
    R = np.array(rotation_matrix)
    corners_cam = (R @ corners_local.T).T + np.array(center)
    return corners_cam.tolist()

def project_to_2d(corners_3d, K):
    """Project 3D corners to 2D image plane"""
    corners_3d = np.array(corners_3d)
    corners_2d = (K @ corners_3d.T).T
    corners_2d[:, 0] /= corners_2d[:, 2]
    corners_2d[:, 1] /= corners_2d[:, 2]
    
    x_min = float(corners_2d[:, 0].min())
    y_min = float(corners_2d[:, 1].min())
    x_max = float(corners_2d[:, 0].max())
    y_max = float(corners_2d[:, 1].max())
    
    return [x_min, y_min, x_max, y_max]

def regenerate_json_files():
    """Regenerate JSON files from existing image data"""
    
    print("="*70)
    print("REGENERATING RHINO JSON FILES")
    print("="*70)
    
    # Find matched pairs
    matched_pairs = find_matched_pairs()
    print(f"Found {len(matched_pairs)} matched video-result pairs")
    
    # Output directory
    output_dir = "/home/shuklva/ovmono3d/datasets/Omni3D"
    os.makedirs(output_dir, exist_ok=True)
    
    # Split videos (same as before)
    video_ids = [vid_id for vid_id, _, _ in matched_pairs]
    # Using the same split as before for consistency
    train_videos = ['30_4', '32_1', '94_1', '35_1', '36_1', '35_2', '57_1', '35_3']
    val_videos = ['90_1', '105_1']
    test_videos = ['30_3', '57_2']
    
    print(f"\nData split:")
    print(f"  Train: {train_videos}")
    print(f"  Val: {val_videos}")
    print(f"  Test: {test_videos}")
    
    splits = {
        'RHINO_train': train_videos,
        'RHINO_val': val_videos,
        'RHINO_test': test_videos
    }
    
    # Use category ID 98 (rhino in stats.json)
    category_id = 98
    categories = [{
        "id": category_id,
        "name": "rhino",
        "supercategory": "animal"
    }]
    
    for split_name, video_list in splits.items():
        print(f"\n{'='*50}")
        print(f"Processing {split_name}")
        print('='*50)
        
        # Use lowercase dataset_id to match what the code expects
        dataset_id_value = split_name.lower()
        
        dataset = {
            "info": {
                "id": dataset_id_value,
                "name": split_name,
                "source": "rhino_wildlife",
                "known_category_ids": [category_id]
            },
            "categories": categories,
            "images": [],
            "annotations": []
        }
        
        image_id_counter = 1
        anno_id_counter = 1
        
        for vid_id in tqdm(video_list, desc=f"Videos in {split_name}"):
            
            # Find directories
            video_dir = None
            result_dir = None
            for v_id, v_dir, r_dir in matched_pairs:
                if v_id == vid_id:
                    video_dir = v_dir
                    result_dir = r_dir
                    break
            
            if not video_dir:
                continue
            
            # Directories
            bbox_dir = os.path.join(result_dir, "bounding_boxes")
            camera_dir = os.path.join(result_dir, "camera")
            grounded_sam_dir = os.path.join(video_dir, "grounded-sam")
            
            # Check if image already exists in output
            image_output_dir = f"/home/shuklva/ovmono3d/datasets/rhino/{vid_id}"
            
            bbox_files = sorted(glob.glob(os.path.join(bbox_dir, "*.json")))
            
            for bbox_file in bbox_files:
                frame_name = os.path.splitext(os.path.basename(bbox_file))[0]
                
                # Check if image exists in output location
                output_image_path = os.path.join(image_output_dir, f"{frame_name}.jpg")
                if not os.path.exists(output_image_path):
                    continue
                
                camera_path = os.path.join(camera_dir, f"{frame_name}.npz")
                sam_path = os.path.join(grounded_sam_dir, f"{frame_name}_results.json")
                
                if not os.path.exists(camera_path):
                    continue
                
                # Load camera intrinsics
                cam_data = np.load(camera_path)
                K = cam_data['intrinsics'].astype(float)
                cam_data.close()
                
                # Get image dimensions
                img_width, img_height = 768, 432
                if os.path.exists(sam_path):
                    with open(sam_path, 'r') as f:
                        sam_data = json.load(f)
                    img_width = sam_data.get('img_width', 768)
                    img_height = sam_data.get('img_height', 432)
                
                # Convert K to nested list format (3x3)
                K_nested = [
                    [float(K[0,0]), float(K[0,1]), float(K[0,2])],
                    [float(K[1,0]), float(K[1,1]), float(K[1,2])],
                    [float(K[2,0]), float(K[2,1]), float(K[2,2])]
                ]
                
                # Create image entry
                image_entry = {
                    "id": image_id_counter,
                    "file_path": f"rhino/{vid_id}/{frame_name}.jpg",
                    "dataset_id": dataset_id_value,
                    "height": img_height,
                    "width": img_width,
                    "K": K_nested  # Properly formatted as 3x3
                }
                dataset["images"].append(image_entry)
                
                # Load 3D boxes
                with open(bbox_file, 'r') as f:
                    boxes_3d = json.load(f)
                
                for box in boxes_3d:
                    if box.get('class_name') != 'rhino':
                        continue
                    
                    center = box['center']
                    dims = box['dimensions']
                    R = box['rotation_matrix']
                    
                    if center[2] <= 0:
                        continue
                    
                    corners_3d = compute_3d_corners(center, dims, R)
                    bbox_2d_proj = project_to_2d(corners_3d, K)
                    
                    bbox_2d_tight = bbox_2d_proj.copy()
                    if os.path.exists(sam_path):
                        with open(sam_path, 'r') as f:
                            sam_data = json.load(f)
                        for sam_anno in sam_data.get('annotations', []):
                            if sam_anno.get('class_name') == 'rhino':
                                bbox_2d_tight = sam_anno['bbox']
                                break
                    
                    x1, y1, x2, y2 = bbox_2d_tight
                    bbox_xywh = [x1, y1, x2-x1, y2-y1]
                    
                    annotation = {
                        "id": anno_id_counter,
                        "image_id": image_id_counter,
                        "category_id": category_id,
                        "category_name": "rhino",
                        "dataset_id": dataset_id_value,  # IMPORTANT: Add this field
                        
                        "bbox": bbox_xywh,
                        "bbox2D_proj": bbox_2d_proj,
                        "bbox2D_tight": bbox_2d_tight,
                        "bbox2D_trunc": bbox_2d_proj,
                        
                        "center_cam": center,
                        "dimensions": dims,
                        "R_cam": R,
                        "pose": R,
                        "bbox3D_cam": corners_3d,
                        
                        "truncation": 0.0,
                        "visibility": 1.0,
                        "behind_camera": False,
                        "valid3D": True,
                        "lidar_pts": 100,
                        "segmentation_pts": 100,
                        "depth_error": 0.0,
                        
                        "area": bbox_xywh[2] * bbox_xywh[3],
                        "iscrowd": False,
                        "ignore": False,
                        "ignore2D": False,
                        "ignore3D": False
                    }
                    
                    dataset["annotations"].append(annotation)
                    anno_id_counter += 1
                
                image_id_counter += 1
        
        # Save JSON
        output_path = os.path.join(output_dir, f"{split_name}.json")
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"  Images: {len(dataset['images'])}")
        print(f"  Annotations: {len(dataset['annotations'])}")
        print(f"  Saved to: {output_path}")
    
    print("\n" + "="*70)
    print("JSON REGENERATION COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    regenerate_json_files()
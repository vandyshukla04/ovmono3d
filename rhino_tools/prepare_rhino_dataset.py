#!/usr/bin/env python3
"""
Complete RHINO Dataset Preparation Pipeline

This script handles the entire workflow for preparing RHINO dataset from CUT3R output.
Run this script once and it will:
1. Find matched video-result pairs from CUT3R output
2. Generate properly formatted JSON annotations for train/val/test splits
3. Register the rhino category in Omni3D stats.json
4. Validate the generated dataset

Usage:
    python rhino_tools/prepare_rhino_dataset.py --cutr_videos /path/to/CUT3R/videos --cutr_results /path/to/CUT3R/results
"""

import os
import json
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path


# Configuration
RHINO_CATEGORY_ID = 98
RHINO_CATEGORY = {
    "id": RHINO_CATEGORY_ID,
    "name": "rhino",
    "supercategory": "animal"
}

# Default data splits (can be customized)
DEFAULT_SPLITS = {
    'RHINO_train': ['30_4', '32_1', '94_1', '35_1', '36_1', '35_2', '57_1', '35_3'],
    'RHINO_val': ['90_1', '105_1'],
    'RHINO_test': ['30_3', '57_2']
}


class RhinoDatasetPreparer:
    """Handles complete RHINO dataset preparation from CUT3R output"""

    def __init__(self, cutr_videos_dir, cutr_results_dir, output_base_dir, image_output_dir):
        self.cutr_videos_dir = Path(cutr_videos_dir)
        self.cutr_results_dir = Path(cutr_results_dir)
        self.output_base_dir = Path(output_base_dir)
        self.image_output_dir = Path(image_output_dir)

        # Ensure output directories exist
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

    def find_matched_pairs(self):
        """Find matched video-result directory pairs from CUT3R output"""
        print("\n" + "="*70)
        print("STEP 1: Finding matched video-result pairs")
        print("="*70)

        # Find all video directories
        video_dirs = sorted(self.cutr_videos_dir.glob("rhin-*"))
        video_mapping = {}
        for vdir in video_dirs:
            video_id = vdir.name.replace("rhin-", "")
            video_mapping[video_id] = vdir

        print(f"Found {len(video_mapping)} video directories")

        # Find all result directories (prefer revisit-1)
        result_dirs = sorted(self.cutr_results_dir.glob("tmp-rhin-*"))
        result_mapping = {}
        for rdir in result_dirs:
            basename = rdir.name
            parts = basename.replace("tmp-rhin-", "").split("-revisit-")
            result_id = parts[0]
            revisit_num = int(parts[1]) if len(parts) > 1 else 0

            # Prefer revisit-1, or use if not already mapped
            if result_id not in result_mapping or revisit_num == 1:
                result_mapping[result_id] = rdir

        print(f"Found {len(result_mapping)} result directories")

        # Match pairs
        matched_pairs = []
        for vid_id in sorted(video_mapping.keys()):
            if vid_id in result_mapping:
                matched_pairs.append((vid_id, video_mapping[vid_id], result_mapping[vid_id]))
                print(f"  ✓ Matched: {vid_id}")
            else:
                print(f"  ✗ No results for: {vid_id}")

        print(f"\nTotal matched pairs: {len(matched_pairs)}")
        return matched_pairs

    def compute_3d_corners(self, center, dimensions, rotation_matrix):
        """Compute 8 corners of 3D bounding box in camera coordinates"""
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

    def project_to_2d(self, corners_3d, K):
        """Project 3D corners to 2D and compute bounding box"""
        corners_3d = np.array(corners_3d)
        corners_2d = (K @ corners_3d.T).T
        corners_2d[:, 0] /= corners_2d[:, 2]
        corners_2d[:, 1] /= corners_2d[:, 2]

        x_min = float(corners_2d[:, 0].min())
        y_min = float(corners_2d[:, 1].min())
        x_max = float(corners_2d[:, 0].max())
        y_max = float(corners_2d[:, 1].max())

        return [x_min, y_min, x_max, y_max]

    def process_video_pair(self, video_id, video_dir, result_dir):
        """Process a single video-result pair and extract annotations"""
        annotations = []

        bbox_dir = result_dir / "bounding_boxes"
        camera_dir = result_dir / "camera"
        grounded_sam_dir = video_dir / "grounded-sam"

        video_output_dir = self.image_output_dir / video_id

        if not bbox_dir.exists():
            print(f"    Warning: bbox directory not found for {video_id}")
            return annotations

        bbox_files = sorted(bbox_dir.glob("*.json"))

        for bbox_file in bbox_files:
            frame_name = bbox_file.stem

            # Check if image exists in output location
            output_image_path = video_output_dir / f"{frame_name}.jpg"
            if not output_image_path.exists():
                continue

            camera_path = camera_dir / f"{frame_name}.npz"
            sam_path = grounded_sam_dir / f"{frame_name}_results.json"

            if not camera_path.exists():
                continue

            # Load camera intrinsics
            cam_data = np.load(camera_path)
            K = cam_data['intrinsics'].astype(float)
            cam_data.close()

            # Get image dimensions
            img_width, img_height = 768, 432
            if sam_path.exists():
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

            # Load 3D boxes
            with open(bbox_file, 'r') as f:
                boxes_3d = json.load(f)

            frame_annotations = []
            for box in boxes_3d:
                if box.get('class_name') != 'rhino':
                    continue

                center = box['center']
                dims = box['dimensions']
                R = box['rotation_matrix']

                # Skip boxes behind camera
                if center[2] <= 0:
                    continue

                # Compute 3D corners and project to 2D
                corners_3d = self.compute_3d_corners(center, dims, R)
                bbox_2d_proj = self.project_to_2d(corners_3d, K)

                # Use SAM tight bbox if available
                bbox_2d_tight = bbox_2d_proj.copy()
                if sam_path.exists():
                    with open(sam_path, 'r') as f:
                        sam_data = json.load(f)
                    for sam_anno in sam_data.get('annotations', []):
                        if sam_anno.get('class_name') == 'rhino':
                            bbox_2d_tight = sam_anno['bbox']
                            break

                # Convert to XYWH format
                x1, y1, x2, y2 = bbox_2d_tight
                bbox_xywh = [x1, y1, x2-x1, y2-y1]

                frame_annotations.append({
                    'bbox': bbox_xywh,
                    'bbox2D_proj': bbox_2d_proj,
                    'bbox2D_tight': bbox_2d_tight,
                    'bbox2D_trunc': bbox_2d_proj,
                    'center_cam': center,
                    'dimensions': dims,
                    'R_cam': R,
                    'pose': R,
                    'bbox3D_cam': corners_3d,
                    'area': bbox_xywh[2] * bbox_xywh[3]
                })

            if frame_annotations:
                annotations.append({
                    'frame_name': frame_name,
                    'video_id': video_id,
                    'K': K_nested,
                    'width': img_width,
                    'height': img_height,
                    'boxes': frame_annotations
                })

        return annotations

    def generate_json_splits(self, matched_pairs, splits=None):
        """Generate JSON files for train/val/test splits"""
        print("\n" + "="*70)
        print("STEP 2: Generating JSON annotation files")
        print("="*70)

        if splits is None:
            splits = DEFAULT_SPLITS

        print(f"\nData splits:")
        for split_name, video_list in splits.items():
            print(f"  {split_name}: {video_list}")

        generated_files = []

        for split_name, video_list in splits.items():
            print(f"\n{'='*50}")
            print(f"Processing {split_name}")
            print('='*50)

            # Use lowercase dataset_id (required by dataset_mapper.py)
            dataset_id_value = split_name.lower()

            dataset = {
                "info": {
                    "id": dataset_id_value,
                    "name": split_name,
                    "source": "rhino_wildlife",
                    "known_category_ids": [RHINO_CATEGORY_ID]
                },
                "categories": [RHINO_CATEGORY],
                "images": [],
                "annotations": []
            }

            image_id_counter = 1
            anno_id_counter = 1

            for vid_id in tqdm(video_list, desc=f"Videos in {split_name}"):
                # Find directories for this video
                video_dir = None
                result_dir = None
                for v_id, v_dir, r_dir in matched_pairs:
                    if v_id == vid_id:
                        video_dir = v_dir
                        result_dir = r_dir
                        break

                if not video_dir:
                    print(f"  Warning: No data found for video {vid_id}")
                    continue

                # Process this video
                frame_annotations = self.process_video_pair(vid_id, video_dir, result_dir)

                for frame_data in frame_annotations:
                    # Create image entry
                    image_entry = {
                        "id": image_id_counter,
                        "file_path": f"rhino/{frame_data['video_id']}/{frame_data['frame_name']}.jpg",
                        "dataset_id": dataset_id_value,
                        "height": frame_data['height'],
                        "width": frame_data['width'],
                        "K": frame_data['K']
                    }
                    dataset["images"].append(image_entry)

                    # Create annotation entries for each box
                    for box in frame_data['boxes']:
                        annotation = {
                            "id": anno_id_counter,
                            "image_id": image_id_counter,
                            "category_id": RHINO_CATEGORY_ID,
                            "category_name": "rhino",
                            "dataset_id": dataset_id_value,

                            # 2D bounding boxes
                            "bbox": box['bbox'],
                            "bbox_mode": 1,  # XYWH_ABS
                            "bbox2D_proj": box['bbox2D_proj'],
                            "bbox2D_tight": box['bbox2D_tight'],
                            "bbox2D_trunc": box['bbox2D_trunc'],

                            # 3D information
                            "center_cam": box['center_cam'],
                            "dimensions": box['dimensions'],
                            "R_cam": box['R_cam'],
                            "pose": box['pose'],
                            "bbox3D_cam": box['bbox3D_cam'],

                            # Metadata
                            "truncation": 0.0,
                            "visibility": 1.0,
                            "behind_camera": False,
                            "valid3D": True,
                            "lidar_pts": 100,
                            "segmentation_pts": 100,
                            "depth_error": 0.0,

                            # Standard COCO fields
                            "area": box['area'],
                            "iscrowd": False,
                            "ignore": False,
                            "ignore2D": False,
                            "ignore3D": False
                        }

                        dataset["annotations"].append(annotation)
                        anno_id_counter += 1

                    image_id_counter += 1

            # Save JSON file
            output_path = self.output_base_dir / f"{split_name}.json"
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)

            print(f"  Images: {len(dataset['images'])}")
            print(f"  Annotations: {len(dataset['annotations'])}")
            print(f"  Saved to: {output_path}")

            generated_files.append(output_path)

        return generated_files

    def register_in_stats(self):
        """Register rhino category in Omni3D stats.json"""
        print("\n" + "="*70)
        print("STEP 3: Registering rhino in Omni3D stats.json")
        print("="*70)

        stats_path = self.output_base_dir / "stats.json"

        if not stats_path.exists():
            print(f"  Warning: stats.json not found at {stats_path}")
            print(f"  Skipping stats registration")
            return

        # Create backup
        backup_path = stats_path.with_suffix('.json.backup')
        if not backup_path.exists():
            shutil.copy2(stats_path, backup_path)
            print(f"  Created backup: {backup_path}")

        # Load existing stats
        with open(stats_path, 'r') as f:
            stats = json.load(f)

        # Check if rhino already exists
        if 'rhino' in stats.get('category_names', []):
            print("  'rhino' already registered in stats.json")
            return

        # Add rhino
        if 'category_names' not in stats:
            stats['category_names'] = []
        stats['category_names'].append('rhino')

        if 'categories' not in stats:
            stats['categories'] = []
        stats['categories'].append(RHINO_CATEGORY)

        # Save updated stats
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"  ✓ Successfully registered rhino (category_id: {RHINO_CATEGORY_ID})")
        print(f"  Total categories: {len(stats['categories'])}")

    def validate_dataset(self, json_files):
        """Validate generated JSON files"""
        print("\n" + "="*70)
        print("STEP 4: Validating generated dataset")
        print("="*70)

        all_valid = True

        for json_file in json_files:
            print(f"\nValidating {json_file.name}...")

            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check required fields
            checks = [
                ('info.id', data.get('info', {}).get('id') is not None),
                ('categories', len(data.get('categories', [])) > 0),
                ('images', len(data.get('images', [])) > 0),
                ('annotations', len(data.get('annotations', [])) > 0),
            ]

            for check_name, passed in checks:
                status = "✓" if passed else "✗"
                print(f"  {status} {check_name}")
                if not passed:
                    all_valid = False

            # Validate image entries
            if data.get('images'):
                img = data['images'][0]
                img_checks = [
                    ('K matrix (3x3)', isinstance(img.get('K'), list) and len(img['K']) == 3 and len(img['K'][0]) == 3),
                    ('dataset_id', img.get('dataset_id') is not None),
                    ('file_path', img.get('file_path') is not None),
                ]

                for check_name, passed in img_checks:
                    status = "✓" if passed else "✗"
                    print(f"  {status} Image: {check_name}")
                    if not passed:
                        all_valid = False

            # Validate annotation entries
            if data.get('annotations'):
                anno = data['annotations'][0]
                anno_checks = [
                    ('category_id', anno.get('category_id') == RHINO_CATEGORY_ID),
                    ('dataset_id', anno.get('dataset_id') is not None),
                    ('bbox', len(anno.get('bbox', [])) == 4),
                    ('center_cam', len(anno.get('center_cam', [])) == 3),
                    ('dimensions', len(anno.get('dimensions', [])) == 3),
                    ('bbox3D_cam', len(anno.get('bbox3D_cam', [])) == 8),
                ]

                for check_name, passed in anno_checks:
                    status = "✓" if passed else "✗"
                    print(f"  {status} Annotation: {check_name}")
                    if not passed:
                        all_valid = False

            # Statistics
            print(f"\n  Statistics:")
            print(f"    Images: {len(data['images'])}")
            print(f"    Annotations: {len(data['annotations'])}")
            if data['images']:
                avg_annos = len(data['annotations']) / len(data['images'])
                print(f"    Avg annotations/image: {avg_annos:.2f}")

        print("\n" + "="*70)
        if all_valid:
            print("✓ VALIDATION PASSED - Dataset is ready for training!")
        else:
            print("✗ VALIDATION FAILED - Please fix the issues above")
        print("="*70)

        return all_valid


def main():
    parser = argparse.ArgumentParser(description="Prepare RHINO dataset from CUT3R output")
    parser.add_argument(
        '--cutr_videos',
        type=str,
        default='/home/shuklva/CUT3R/examples/wd_data/rhinos_cami',
        help='Path to CUT3R video directory'
    )
    parser.add_argument(
        '--cutr_results',
        type=str,
        default='/home/shuklva/CUT3R/results',
        help='Path to CUT3R results directory'
    )
    parser.add_argument(
        '--output_json_dir',
        type=str,
        default='/home/shuklva/ovmono3d/datasets/Omni3D',
        help='Output directory for JSON annotation files'
    )
    parser.add_argument(
        '--output_image_dir',
        type=str,
        default='/home/shuklva/ovmono3d/datasets/rhino',
        help='Directory containing rhino images (organized by video ID)'
    )
    parser.add_argument(
        '--skip_validation',
        action='store_true',
        help='Skip dataset validation step'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("RHINO DATASET PREPARATION PIPELINE")
    print("="*70)
    print(f"CUT3R videos: {args.cutr_videos}")
    print(f"CUT3R results: {args.cutr_results}")
    print(f"Output JSON dir: {args.output_json_dir}")
    print(f"Output image dir: {args.output_image_dir}")

    # Initialize preparer
    preparer = RhinoDatasetPreparer(
        cutr_videos_dir=args.cutr_videos,
        cutr_results_dir=args.cutr_results,
        output_base_dir=args.output_json_dir,
        image_output_dir=args.output_image_dir
    )

    # Step 1: Find matched pairs
    matched_pairs = preparer.find_matched_pairs()

    if not matched_pairs:
        print("\n✗ ERROR: No matched video-result pairs found!")
        print("Please check your CUT3R output directories")
        return 1

    # Step 2: Generate JSON files
    json_files = preparer.generate_json_splits(matched_pairs)

    # Step 3: Register in stats.json
    preparer.register_in_stats()

    # Step 4: Validate
    if not args.skip_validation:
        valid = preparer.validate_dataset(json_files)
        if not valid:
            return 1

    print("\n" + "="*70)
    print("✓ DATASET PREPARATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review the generated JSON files")
    print("  2. Run training with: python tools/train_net.py --config-file configs/RHINO_train.yaml")
    print("  3. Monitor training in: output/rhino_cubercnn_b4_ovmono_ckpt/")

    return 0


if __name__ == "__main__":
    exit(main())

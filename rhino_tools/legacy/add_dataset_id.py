#!/usr/bin/env python3
"""
Add dataset_id field to all annotations in RHINO JSON files
"""

import json
import os

def add_dataset_id_to_annotations():
    base_dir = "/home/shuklva/ovmono3d/datasets/Omni3D"
    
    for filename in ['RHINO_train.json', 'RHINO_val.json', 'RHINO_test.json']:
        filepath = os.path.join(base_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Skipping {filename} - not found")
            continue
        
        print(f"Processing {filename}...")
        
        # Load JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Get the dataset_id from info
        dataset_id = data['info']['id']
        print(f"  Dataset ID: {dataset_id}")
        
        # Create image_id to dataset_id mapping
        image_to_dataset = {}
        for img in data['images']:
            image_to_dataset[img['id']] = img['dataset_id']
        
        # Add dataset_id to each annotation
        updated_count = 0
        for anno in data['annotations']:
            # Get dataset_id from the corresponding image
            img_id = anno['image_id']
            if img_id in image_to_dataset:
                anno['dataset_id'] = image_to_dataset[img_id]
                updated_count += 1
            else:
                # Fallback to info id
                anno['dataset_id'] = dataset_id
                updated_count += 1
        
        # Save updated JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Updated {updated_count} annotations with dataset_id")
    
    print("\nAll files updated successfully!")

if __name__ == "__main__":
    add_dataset_id_to_annotations()
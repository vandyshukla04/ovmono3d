#!/usr/bin/env python3
"""
Fix category IDs in already generated RHINO JSON files
"""

import json
import os

def fix_category_ids():
    base_dir = "/home/shuklva/ovmono3d/datasets/Omni3D"
    
    # Process each RHINO JSON file
    for filename in ['RHINO_train.json', 'RHINO_val.json', 'RHINO_test.json']:
        filepath = os.path.join(base_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Skipping {filename} - file not found")
            continue
            
        print(f"Processing {filename}...")
        
        # Load JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Fix category ID to 98 (rhino's ID in stats.json)
        data['categories'] = [{
            "id": 98,
            "name": "rhino", 
            "supercategory": "animal"
        }]
        
        # Update info section
        data['info']['known_category_ids'] = [98]
        
        # Fix all annotations
        for anno in data['annotations']:
            if anno['category_name'] == 'rhino':
                anno['category_id'] = 98
        
        # Save updated JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  - Updated {len(data['annotations'])} annotations")
        print(f"  - Set category_id to 98")
    
    # Also update category_meta_rhino.json
    category_meta_path = os.path.join(base_dir, "category_meta_rhino.json")
    if os.path.exists(category_meta_path):
        print("\nUpdating category_meta_rhino.json...")
        meta = {
            "thing_classes": ["rhino"],
            "thing_dataset_id_to_contiguous_id": {
                "98": 0  # Map category ID 98 to class 0
            }
        }
        with open(category_meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print("  - Updated category mapping")
    
    print("\nAll files updated successfully!")

if __name__ == "__main__":
    fix_category_ids()
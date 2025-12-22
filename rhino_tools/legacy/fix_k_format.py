#!/usr/bin/env python3
"""
Fix K matrix format from flat list to 3x3 nested list
"""

import json
import os
import numpy as np

def fix_k_matrix_format():
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
        
        # Fix K matrix for each image
        fixed_count = 0
        for img in data['images']:
            if 'K' in img and isinstance(img['K'], list):
                if len(img['K']) == 9:
                    # Convert flat list to 3x3 nested list
                    K_flat = img['K']
                    K_matrix = [
                        [K_flat[0], K_flat[1], K_flat[2]],
                        [K_flat[3], K_flat[4], K_flat[5]],
                        [K_flat[6], K_flat[7], K_flat[8]]
                    ]
                    img['K'] = K_matrix
                    fixed_count += 1
                elif len(img['K']) == 3 and len(img['K'][0]) == 3:
                    # Already in correct format
                    pass
                else:
                    print(f"  WARNING: Unexpected K format for image {img['id']}")
        
        # Save updated JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Fixed {fixed_count} K matrices")
        
        # Verify the fix
        with open(filepath, 'r') as f:
            verify_data = json.load(f)
        
        if verify_data['images']:
            K = verify_data['images'][0]['K']
            print(f"  Sample K matrix after fix:")
            print(f"    [[{K[0][0]:.2f}, {K[0][1]:.2f}, {K[0][2]:.2f}],")
            print(f"     [{K[1][0]:.2f}, {K[1][1]:.2f}, {K[1][2]:.2f}],")
            print(f"     [{K[2][0]:.2f}, {K[2][1]:.2f}, {K[2][2]:.2f}]]")
            print(f"  Can access K[1][1]: {K[1][1]}")
    
    print("\nAll K matrices fixed!")

if __name__ == "__main__":
    fix_k_matrix_format()
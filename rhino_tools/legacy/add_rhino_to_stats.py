#!/usr/bin/env python3
"""
Script to add rhino category to Omni3D stats.json
"""

import json
import os
import shutil

def add_rhino_to_stats():
    stats_path = "/home/shuklva/ovmono3d/datasets/Omni3D/stats.json"
    backup_path = "/home/shuklva/ovmono3d/datasets/Omni3D/stats_backup.json"
    
    # Create backup
    print("Creating backup...")
    shutil.copy2(stats_path, backup_path)
    print(f"Backup saved to: {backup_path}")
    
    # Load existing stats
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    print(f"Current number of categories: {len(stats['categories'])}")
    print(f"Last category ID: {stats['categories'][-1]['id']}")
    
    # Check if rhino already exists
    if 'rhino' in stats['category_names']:
        print("'rhino' already exists in stats.json!")
        return
    
    # Add rhino to category_names
    stats['category_names'].append('rhino')
    
    # Add rhino category with next available ID (98)
    rhino_category = {
        "supercategory": "animal",
        "id": 98,
        "name": "rhino"
    }
    stats['categories'].append(rhino_category)
    
    # Save updated stats
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    
    print("\nSuccessfully added rhino category:")
    print(f"  - Category ID: 98")
    print(f"  - Category name: rhino")
    print(f"  - Supercategory: animal")
    print(f"Total categories now: {len(stats['categories'])}")
    
    # Verify by reloading
    with open(stats_path, 'r') as f:
        verify_stats = json.load(f)
    
    if 'rhino' in verify_stats['category_names']:
        print("\nVerification successful: 'rhino' is now in stats.json")
    else:
        print("\nERROR: Verification failed!")

if __name__ == "__main__":
    add_rhino_to_stats()
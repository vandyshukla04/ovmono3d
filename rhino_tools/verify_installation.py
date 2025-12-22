#!/usr/bin/env python3
"""
Verification script to check RHINO dataset setup

Checks:
- Dataset JSON files exist and are valid
- Rhino category registered in stats.json
- Image directories exist
- Config files are present
"""

import os
import json
import sys
from pathlib import Path

def check_file(path, description):
    """Check if a file exists"""
    if os.path.exists(path):
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ✗ {description} - NOT FOUND")
        return False

def check_json_valid(path, description):
    """Check if a JSON file exists and is valid"""
    if not os.path.exists(path):
        print(f"  ✗ {description} - NOT FOUND")
        return False

    try:
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"  ✓ {description} ({len(data.get('images', []))} images, {len(data.get('annotations', []))} annos)")
        return True
    except Exception as e:
        print(f"  ✗ {description} - INVALID JSON: {e}")
        return False

def main():
    print("="*70)
    print("RHINO DATASET VERIFICATION")
    print("="*70)

    base_dir = Path(__file__).parent.parent
    all_checks_passed = True

    # Check dataset JSONs
    print("\n1. Dataset JSON Files:")
    json_dir = base_dir / "datasets" / "Omni3D"
    all_checks_passed &= check_json_valid(json_dir / "RHINO_train.json", "RHINO_train.json")
    all_checks_passed &= check_json_valid(json_dir / "RHINO_val.json", "RHINO_val.json")
    all_checks_passed &= check_json_valid(json_dir / "RHINO_test.json", "RHINO_test.json")

    # Check stats.json for rhino
    print("\n2. Category Registration:")
    stats_path = json_dir / "stats.json"
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        if 'rhino' in stats.get('category_names', []):
            print(f"  ✓ Rhino registered in stats.json (category_id: 98)")
        else:
            print(f"  ✗ Rhino NOT registered in stats.json")
            all_checks_passed = False
    else:
        print(f"  ✗ stats.json not found")
        all_checks_passed = False

    # Check image directories
    print("\n3. Image Directories:")
    rhino_dir = base_dir / "datasets" / "rhino"
    if rhino_dir.exists():
        video_dirs = sorted([d for d in rhino_dir.iterdir() if d.is_dir()])
        print(f"  ✓ Rhino image directory exists ({len(video_dirs)} video folders)")
        for vdir in video_dirs[:5]:  # Show first 5
            image_count = len(list(vdir.glob("*.jpg")))
            print(f"    - {vdir.name}: {image_count} images")
        if len(video_dirs) > 5:
            print(f"    ... and {len(video_dirs) - 5} more")
    else:
        print(f"  ✗ Rhino image directory not found")
        all_checks_passed = False

    # Check config files
    print("\n4. Configuration Files:")
    all_checks_passed &= check_file(base_dir / "configs" / "RHINO_train.yaml", "RHINO_train.yaml")

    # Check tools
    print("\n5. RHINO Tools:")
    tools_dir = base_dir / "rhino_tools"
    all_checks_passed &= check_file(tools_dir / "prepare_rhino_dataset.py", "prepare_rhino_dataset.py")
    all_checks_passed &= check_file(tools_dir / "demo_rhino.py", "demo_rhino.py")
    all_checks_passed &= check_file(tools_dir / "README.md", "rhino_tools/README.md")

    # Check documentation
    print("\n6. Documentation:")
    all_checks_passed &= check_file(base_dir / "RHINO_TRAINING_GUIDE.md", "RHINO_TRAINING_GUIDE.md")
    all_checks_passed &= check_file(base_dir / "CLEANUP_SUMMARY.md", "CLEANUP_SUMMARY.md")

    # Check checkpoints
    print("\n7. Model Checkpoints:")
    checkpoint_dir = base_dir / "checkpoints"
    if checkpoint_dir.exists():
        pth_files = list(checkpoint_dir.glob("*.pth"))
        if pth_files:
            print(f"  ✓ Checkpoints directory ({len(pth_files)} .pth files)")
            for pth in pth_files[:3]:
                size_mb = pth.stat().st_size / (1024*1024)
                print(f"    - {pth.name} ({size_mb:.1f} MB)")
        else:
            print(f"  ! Checkpoints directory exists but no .pth files found")
            print(f"    Download ovmono3d_lift.pth for better training results")
    else:
        print(f"  ! Checkpoints directory not found")
        print(f"    Training will start from scratch (slower convergence)")

    # Final summary
    print("\n" + "="*70)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - Ready for training!")
        print("="*70)
        print("\nNext steps:")
        print("  1. python rhino_tools/prepare_rhino_dataset.py  # If dataset needs regeneration")
        print("  2. python tools/train_net.py --config-file configs/RHINO_train.yaml --num-gpus 1")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please review the issues above")
        print("="*70)
        print("\nTo fix:")
        print("  1. Run: python rhino_tools/prepare_rhino_dataset.py")
        print("  2. Check paths in the script match your CUT3R output")
        return 1

if __name__ == "__main__":
    sys.exit(main())

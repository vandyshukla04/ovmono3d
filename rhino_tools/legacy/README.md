# Legacy Scripts (Archived)

These scripts are **no longer needed** and have been replaced by the unified `prepare_rhino_dataset.py` pipeline.

They are kept here for reference only.

## Replaced Scripts

### regenerate_json.py
**Old purpose**: Generate RHINO JSON files from CUT3R output
**New solution**: Use `rhino_tools/prepare_rhino_dataset.py`

### fix_category.py
**Old purpose**: Fix category IDs to 98
**Issue**: Shouldn't be needed if data is generated correctly
**New solution**: `prepare_rhino_dataset.py` sets category_id=98 from the start

### fix_category_.py
**Old purpose**: Minimal version of fix_category.py
**Issue**: Duplicate script
**New solution**: Not needed

### fix_k_format.py
**Old purpose**: Convert K matrix from flat list to 3x3 nested list
**Issue**: Data should be generated in correct format
**New solution**: `prepare_rhino_dataset.py` formats K as 3x3 from the start

### add_dataset_id.py
**Old purpose**: Add dataset_id field to annotations
**Issue**: Should be included during generation
**New solution**: `prepare_rhino_dataset.py` adds dataset_id automatically

### add_rhino_to_stats.py
**Old purpose**: Register rhino category in stats.json
**New solution**: `prepare_rhino_dataset.py` handles registration (Step 3)

## Why These Were Needed (and why they're not anymore)

### The Old Workflow Problem

The old workflow was a multi-step "fix-and-patch" approach:

1. Run `regenerate_json.py` → Generate JSONs (but with issues)
2. Run `fix_k_format.py` → Fix K matrix format
3. Run `fix_category.py` → Fix category IDs
4. Run `add_dataset_id.py` → Add missing dataset_id
5. Run `add_rhino_to_stats.py` → Register category
6. Hope everything works!

**Problems:**
- Had to run scripts in correct order
- Easy to forget a step
- Each fix script assumed certain state
- No validation
- "Two steps forward, one step back"

### The New Workflow Solution

The new workflow is a single comprehensive pipeline:

1. Run `prepare_rhino_dataset.py` → Done!

**Benefits:**
- ✓ Single command
- ✓ Correct format from the start
- ✓ Automatic validation
- ✓ Clear error messages
- ✓ No manual fixes needed

## Migration Guide

If you have old data prepared with legacy scripts:

### Option 1: Regenerate (Recommended)
```bash
python rhino_tools/prepare_rhino_dataset.py
```

### Option 2: Keep existing (if it works)
If your current `RHINO_*.json` files work for training, you can keep them.
Just don't use the legacy scripts for new data.

## Deleting These Scripts

These scripts can be safely deleted once you've verified the new pipeline works:

```bash
rm -rf rhino_tools/legacy/
```

They are only kept for reference during transition.

import json
import numpy as np

data = json.load(open('datasets/Omni3D/RHINO_train.json'))

# Analyze dimension statistics
all_dims = []
all_depths = []
for anno in data['annotations']:
    dims = anno['dimensions']  # W, H, L
    depth = anno['center_cam'][2]  # Z distance
    all_dims.append(dims)
    all_depths.append(depth)

dims_array = np.array(all_dims)
depths_array = np.array(all_depths)

print(f"Dimension stats (W, H, L):")
print(f"  Mean: {dims_array.mean(axis=0)}")
print(f"  Min:  {dims_array.min(axis=0)}")
print(f"  Max:  {dims_array.max(axis=0)}")
print(f"\nDepth stats:")
print(f"  Mean: {depths_array.mean():.2f}")
print(f"  Range: {depths_array.min():.2f} - {depths_array.max():.2f}")

# Check a few examples
for i in range(min(5, len(data['annotations']))):
    anno = data['annotations'][i]
    dims = anno['dimensions']
    depth = anno['center_cam'][2]
    print(f"\nExample {i}: Depth={depth:.2f}, Dims (WxHxL)={dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f}")
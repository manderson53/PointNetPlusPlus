import os
import numpy as np
from plyfile import PlyData

# --- Configuration ---
PLY_FILES = [
    r"C:\Users\RemoteCollabHoloLens\Desktop\PointNet++\Benchmark\Benchmark\training_10_classes\Lille2.ply",
    r"C:\Users\RemoteCollabHoloLens\Desktop\PointNet++\Benchmark\Benchmark\training_10_classes\Lille1_2.ply",
    r"C:\Users\RemoteCollabHoloLens\Desktop\PointNet++\Benchmark\Benchmark\training_10_classes\paris.ply"
]
OUTPUT_DIR = r"preprocessed_training_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_xyz_range(ply_file):
    """
    Computes min/max for x,y,z from a PLY file efficiently.
    """
    plydata = PlyData.read(ply_file)
    vertex = plydata['vertex']
    
    # Convert to numpy arrays (x, y, z)
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    
    xyz_min = np.array([x.min(), y.min(), z.min()])
    xyz_max = np.array([x.max(), y.max(), z.max()])
    return xyz_min, xyz_max

def main():
    xyz_ranges = {}
    for ply_path in PLY_FILES:
        block_name = os.path.splitext(os.path.basename(ply_path))[0]
        print(f"Processing {block_name}...")
        xyz_min, xyz_max = compute_xyz_range(ply_path)
        xyz_ranges[block_name] = {'min': xyz_min, 'max': xyz_max}
        print(f"  min: {xyz_min}, max: {xyz_max}")
    
    output_file = os.path.join(OUTPUT_DIR, 'block_xyz_ranges.npy')
    np.save(output_file, xyz_ranges)
    print(f"\nSaved block XYZ ranges to {output_file}")

if __name__ == "__main__":
    main()

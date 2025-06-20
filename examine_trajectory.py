#!/usr/bin/env python3
"""
Script to examine trajectory data
"""
import numpy as np

# Load trajectory data
data = np.load('data/trajectories/trajectory_20250620_120525.npz')

print("Trajectory Data Analysis:")
print("=" * 50)
print(f"Keys: {list(data.keys())}")
print()

for key in data.keys():
    value = data[key]
    if hasattr(value, 'shape'):
        print(f"{key}: {value.shape} - {value.dtype}")
        if len(value.shape) <= 2 and value.size <= 20:
            print(f"  Content: {value}")
    else:
        print(f"{key}: {value}")
    print()

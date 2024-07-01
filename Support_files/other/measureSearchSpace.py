import numpy as np
from scipy.stats import iqr

def measure_search_space(metrics, resolution=None):
    if resolution is not None:
        # Adjust metrics based on the resolution to minimize or grow the voxel size
        metrics = np.round(metrics / resolution) * resolution

    # Unique and repeated behavior analysis
    unique_behaviors, counts = np.unique(metrics, axis=0, return_counts=True)
    voxels_occupied = np.sum(counts == 1)
    num_repeated_behaviors = unique_behaviors[counts > 1]

    # Calculate the 'quality' as the number of unique behaviors plus the number of unique repeated behaviors
    quality = voxels_occupied + len(num_repeated_behaviors)

    # Collect additional statistics for each metric dimension
    stats = np.array([iqr(metrics, axis=0),
                      np.median(np.abs(metrics - np.median(metrics, axis=0)), axis=0),  # MAD
                      np.ptp(metrics, axis=0),  # Range
                      np.std(metrics, axis=0),  # Standard Deviation
                      np.var(metrics, axis=0)])  # Variance

    return quality, stats.T

# Example usage
metrics = np.random.rand(100, 5) * 100  # Example metrics array with random data
quality, stats = measure_search_space(metrics, resolution=10)
print("Quality:", quality)
print("Stats:\n", stats)

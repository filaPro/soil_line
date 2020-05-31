import numpy as np


# filter rare (red, nir) pairs
def filter(mask, images, n_bins=100, threshold=.0):
    x = np.ravel(images['nir'])
    y = np.ravel(images['red'])
    histogram, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
    histogram = np.ravel(histogram)
    cumulative_sum = np.cumsum(np.sort(histogram))
    sum_threshold = cumulative_sum[-1] * threshold
    index_threshold = np.searchsorted(cumulative_sum, sum_threshold)
    absolute_threshold = np.sort(histogram)[index_threshold]
    histogram_mask = histogram <= absolute_threshold
    x = np.clip(((x - x_edges[0]) / (x_edges[1] - x_edges[0])).astype(np.int), 0, n_bins - 1)
    y = np.clip(((y - y_edges[0]) / (y_edges[1] - y_edges[0])).astype(np.int), 0, n_bins - 1)
    xy = x * n_bins + y
    rare_mask = histogram_mask[xy]
    mask[np.reshape(rare_mask, mask.shape)] = .0
    return mask

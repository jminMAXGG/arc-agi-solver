"""
Grid utility functions for ARC puzzle solving.
"""

import numpy as np
from scipy import ndimage


def get_background_color(grid):
    """Detect background color as most frequent value."""
    return int(np.bincount(grid.flatten()).argmax())


def get_unique_colors(grid, exclude_bg=True):
    """Get set of unique colors in grid."""
    colors = set(np.unique(grid))
    if exclude_bg:
        bg = get_background_color(grid)
        colors.discard(bg)
    return colors


def find_connected_regions(grid, background=None):
    """
    Find connected non-background regions.

    Returns list of (positions, values, mask) tuples.
    """
    if background is None:
        background = get_background_color(grid)

    mask = (grid != background).astype(int)
    labeled, num_features = ndimage.label(mask)

    regions = []
    for i in range(1, num_features + 1):
        region_mask = (labeled == i)
        positions = np.argwhere(region_mask)
        values = grid[region_mask]
        regions.append((positions, values, region_mask))

    return regions


def get_bounding_box(positions):
    """Get bounding box of positions as (min_r, min_c, max_r, max_c)."""
    if len(positions) == 0:
        return None
    arr = np.array(positions)
    return (
        int(arr[:, 0].min()),
        int(arr[:, 1].min()),
        int(arr[:, 0].max()),
        int(arr[:, 1].max())
    )


def extract_subgrid(grid, bbox):
    """Extract subgrid from bounding box."""
    min_r, min_c, max_r, max_c = bbox
    return grid[min_r:max_r+1, min_c:max_c+1].copy()


def place_subgrid(target, subgrid, position, overwrite_bg=True, bg=0):
    """Place subgrid at position in target grid."""
    r, c = position
    h, w = subgrid.shape
    th, tw = target.shape

    for dr in range(h):
        for dc in range(w):
            nr, nc = r + dr, c + dc
            if 0 <= nr < th and 0 <= nc < tw:
                val = subgrid[dr, dc]
                if overwrite_bg or val != bg:
                    target[nr, nc] = val

    return target


def get_shape_signature(positions):
    """
    Get translation-invariant shape signature.

    Returns sorted tuple of relative positions.
    """
    if len(positions) == 0:
        return tuple()

    arr = np.array(positions)
    min_r = arr[:, 0].min()
    min_c = arr[:, 1].min()

    relative = [(int(r - min_r), int(c - min_c)) for r, c in positions]
    return tuple(sorted(relative))


def shapes_match(positions1, positions2):
    """Check if two sets of positions have the same shape."""
    return get_shape_signature(positions1) == get_shape_signature(positions2)


__all__ = [
    'get_background_color',
    'get_unique_colors',
    'find_connected_regions',
    'get_bounding_box',
    'extract_subgrid',
    'place_subgrid',
    'get_shape_signature',
    'shapes_match'
]

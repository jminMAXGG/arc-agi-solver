"""
Basic grid operations for ARC puzzles.
"""

import numpy as np
from ..utils.grid_utils import get_background_color


def crop_to_content(grid, background=None):
    """Crop grid to bounding box of non-background content."""
    if background is None:
        background = get_background_color(grid)

    rows = np.any(grid != background, axis=1)
    cols = np.any(grid != background, axis=0)

    if not rows.any() or not cols.any():
        return grid.copy()

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return grid[rmin:rmax+1, cmin:cmax+1].copy()


def pad_grid(grid, padding, value=0):
    """Add padding around grid."""
    return np.pad(grid, padding, mode='constant', constant_values=value)


def resize_grid(grid, new_shape, fill_value=0):
    """Resize grid to new shape, filling with value."""
    h, w = new_shape
    result = np.full((h, w), fill_value, dtype=grid.dtype)

    oh, ow = grid.shape
    copy_h = min(h, oh)
    copy_w = min(w, ow)

    result[:copy_h, :copy_w] = grid[:copy_h, :copy_w]
    return result


def split_grid_horizontal(grid):
    """Split grid into left and right halves."""
    mid = grid.shape[1] // 2
    return grid[:, :mid].copy(), grid[:, mid:].copy()


def split_grid_vertical(grid):
    """Split grid into top and bottom halves."""
    mid = grid.shape[0] // 2
    return grid[:mid, :].copy(), grid[mid:, :].copy()


__all__ = [
    'crop_to_content',
    'pad_grid',
    'resize_grid',
    'split_grid_horizontal',
    'split_grid_vertical'
]

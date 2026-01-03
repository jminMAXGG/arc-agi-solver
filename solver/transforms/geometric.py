"""
Geometric transforms: rotation, reflection, scaling.
"""

import numpy as np


def rotate_90(grid):
    """Rotate grid 90 degrees clockwise."""
    return np.rot90(grid, k=-1)


def rotate_180(grid):
    """Rotate grid 180 degrees."""
    return np.rot90(grid, k=2)


def rotate_270(grid):
    """Rotate grid 270 degrees clockwise (90 counter-clockwise)."""
    return np.rot90(grid, k=1)


def flip_horizontal(grid):
    """Flip grid horizontally (left-right)."""
    return np.fliplr(grid)


def flip_vertical(grid):
    """Flip grid vertically (top-bottom)."""
    return np.flipud(grid)


def flip_diagonal(grid):
    """Flip grid along main diagonal (transpose)."""
    return grid.T


def flip_antidiagonal(grid):
    """Flip grid along anti-diagonal."""
    return np.rot90(np.fliplr(grid))


def scale_grid(grid, factor):
    """Scale grid by integer factor."""
    h, w = grid.shape
    result = np.zeros((h * factor, w * factor), dtype=grid.dtype)

    for r in range(h):
        for c in range(w):
            result[r*factor:(r+1)*factor, c*factor:(c+1)*factor] = grid[r, c]

    return result


def tile_grid(grid, rows, cols):
    """Tile grid into rows x cols copies."""
    return np.tile(grid, (rows, cols))


__all__ = [
    'rotate_90',
    'rotate_180',
    'rotate_270',
    'flip_horizontal',
    'flip_vertical',
    'flip_diagonal',
    'flip_antidiagonal',
    'scale_grid',
    'tile_grid'
]

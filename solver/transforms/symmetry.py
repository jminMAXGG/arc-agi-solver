"""
Symmetry detection and application transforms.
"""

import numpy as np
from .geometric import (
    flip_horizontal, flip_vertical,
    flip_diagonal, flip_antidiagonal,
    rotate_90, rotate_180, rotate_270
)


def has_horizontal_symmetry(grid):
    """Check if grid is horizontally symmetric."""
    return np.array_equal(grid, flip_horizontal(grid))


def has_vertical_symmetry(grid):
    """Check if grid is vertically symmetric."""
    return np.array_equal(grid, flip_vertical(grid))


def has_rotational_symmetry(grid, degrees=180):
    """Check if grid has rotational symmetry."""
    if degrees == 90:
        return np.array_equal(grid, rotate_90(grid))
    elif degrees == 180:
        return np.array_equal(grid, rotate_180(grid))
    elif degrees == 270:
        return np.array_equal(grid, rotate_270(grid))
    return False


def complete_horizontal_symmetry(grid, direction='right'):
    """Complete grid to achieve horizontal symmetry."""
    h, w = grid.shape
    if direction == 'right':
        result = np.zeros((h, w * 2), dtype=grid.dtype)
        result[:, :w] = grid
        result[:, w:] = flip_horizontal(grid)
    else:
        result = np.zeros((h, w * 2), dtype=grid.dtype)
        result[:, w:] = grid
        result[:, :w] = flip_horizontal(grid)
    return result


def complete_vertical_symmetry(grid, direction='down'):
    """Complete grid to achieve vertical symmetry."""
    h, w = grid.shape
    if direction == 'down':
        result = np.zeros((h * 2, w), dtype=grid.dtype)
        result[:h, :] = grid
        result[h:, :] = flip_vertical(grid)
    else:
        result = np.zeros((h * 2, w), dtype=grid.dtype)
        result[h:, :] = grid
        result[:h, :] = flip_vertical(grid)
    return result


def detect_symmetry_type(grid):
    """
    Detect type of symmetry in grid.

    Returns list of detected symmetries.
    """
    symmetries = []

    if has_horizontal_symmetry(grid):
        symmetries.append('horizontal')
    if has_vertical_symmetry(grid):
        symmetries.append('vertical')
    if has_rotational_symmetry(grid, 90):
        symmetries.append('rotate_90')
    if has_rotational_symmetry(grid, 180):
        symmetries.append('rotate_180')

    return symmetries


__all__ = [
    'has_horizontal_symmetry',
    'has_vertical_symmetry',
    'has_rotational_symmetry',
    'complete_horizontal_symmetry',
    'complete_vertical_symmetry',
    'detect_symmetry_type'
]

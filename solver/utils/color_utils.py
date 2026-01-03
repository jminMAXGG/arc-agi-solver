"""
Color analysis utilities for ARC puzzle solving.
"""

import numpy as np
from collections import Counter


def count_colors(grid):
    """Count occurrences of each color."""
    return Counter(grid.flatten())


def get_minority_color(grid, exclude=None):
    """Get least frequent non-background color."""
    exclude = exclude or set()
    counts = count_colors(grid)

    # Remove excluded colors
    for c in exclude:
        counts.pop(c, None)

    if not counts:
        return None

    return min(counts, key=counts.get)


def get_majority_color(grid, exclude=None):
    """Get most frequent color (excluding specified)."""
    exclude = exclude or set()
    counts = count_colors(grid)

    for c in exclude:
        counts.pop(c, None)

    if not counts:
        return None

    return max(counts, key=counts.get)


def find_color_positions(grid, color):
    """Get all positions of a specific color."""
    return np.argwhere(grid == color)


def get_shared_colors(grid1, grid2):
    """Get colors present in both grids."""
    return set(np.unique(grid1)) & set(np.unique(grid2))


def get_unique_to_grid(grid1, grid2):
    """Get colors in grid1 but not in grid2."""
    return set(np.unique(grid1)) - set(np.unique(grid2))


def replace_color(grid, old_color, new_color):
    """Replace all occurrences of old_color with new_color."""
    result = grid.copy()
    result[result == old_color] = new_color
    return result


def color_histogram(grid):
    """Get color histogram as dict."""
    unique, counts = np.unique(grid, return_counts=True)
    return dict(zip(map(int, unique), map(int, counts)))


__all__ = [
    'count_colors',
    'get_minority_color',
    'get_majority_color',
    'find_color_positions',
    'get_shared_colors',
    'get_unique_to_grid',
    'replace_color',
    'color_histogram'
]

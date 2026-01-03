"""
Fill and region operations.
"""

import numpy as np
from scipy import ndimage
from ..utils.grid_utils import get_background_color


def flood_fill(grid, start_pos, new_color, connectivity=4):
    """
    Flood fill from start position with new color.

    Args:
        grid: Input grid
        start_pos: (row, col) starting position
        new_color: Color to fill with
        connectivity: 4 or 8 for neighbor connectivity

    Returns:
        Grid with filled region
    """
    result = grid.copy()
    r, c = start_pos
    target_color = result[r, c]

    if target_color == new_color:
        return result

    h, w = result.shape
    stack = [(r, c)]
    visited = set()

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1), (0, 1),
                     (1, -1), (1, 0), (1, 1)]

    while stack:
        cr, cc = stack.pop()
        if (cr, cc) in visited:
            continue
        if not (0 <= cr < h and 0 <= cc < w):
            continue
        if result[cr, cc] != target_color:
            continue

        result[cr, cc] = new_color
        visited.add((cr, cc))

        for dr, dc in neighbors:
            stack.append((cr + dr, cc + dc))

    return result


def fill_enclosed_regions(grid, fill_color, background=None):
    """Fill regions completely enclosed by non-background."""
    if background is None:
        background = get_background_color(grid)

    result = grid.copy()
    h, w = result.shape

    # Create mask of background regions
    bg_mask = (result == background)

    # Label connected background regions
    labeled, num = ndimage.label(bg_mask)

    # Find which regions touch the border
    border_labels = set()
    border_labels.update(labeled[0, :])
    border_labels.update(labeled[-1, :])
    border_labels.update(labeled[:, 0])
    border_labels.update(labeled[:, -1])
    border_labels.discard(0)

    # Fill enclosed regions
    for i in range(1, num + 1):
        if i not in border_labels:
            result[labeled == i] = fill_color

    return result


def gravity_fill(grid, direction='down', background=0):
    """Apply gravity to non-background cells."""
    result = grid.copy()
    h, w = result.shape

    if direction == 'down':
        for c in range(w):
            col = result[:, c]
            non_bg = col[col != background]
            result[:, c] = background
            result[h-len(non_bg):, c] = non_bg

    elif direction == 'up':
        for c in range(w):
            col = result[:, c]
            non_bg = col[col != background]
            result[:, c] = background
            result[:len(non_bg), c] = non_bg

    elif direction == 'left':
        for r in range(h):
            row = result[r, :]
            non_bg = row[row != background]
            result[r, :] = background
            result[r, :len(non_bg)] = non_bg

    elif direction == 'right':
        for r in range(h):
            row = result[r, :]
            non_bg = row[row != background]
            result[r, :] = background
            result[r, w-len(non_bg):] = non_bg

    return result


__all__ = [
    'flood_fill',
    'fill_enclosed_regions',
    'gravity_fill'
]

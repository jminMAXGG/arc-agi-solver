"""
Overlay and composition operations.
"""

import numpy as np
from ..utils.grid_utils import get_background_color


def overlay_grids(base, overlay, background=None):
    """
    Overlay one grid on another, ignoring background in overlay.

    Non-background pixels from overlay replace base pixels.
    """
    if background is None:
        background = get_background_color(overlay)

    result = base.copy()
    mask = overlay != background
    result[mask] = overlay[mask]
    return result


def merge_grids(grid1, grid2, priority='first', background=0):
    """
    Merge two grids of same shape.

    priority: 'first' or 'second' determines which grid wins on conflict
    """
    if grid1.shape != grid2.shape:
        raise ValueError("Grids must have same shape")

    result = np.full_like(grid1, background)

    if priority == 'first':
        # Second grid first, then first grid overwrites
        mask2 = grid2 != background
        result[mask2] = grid2[mask2]
        mask1 = grid1 != background
        result[mask1] = grid1[mask1]
    else:
        # First grid first, then second grid overwrites
        mask1 = grid1 != background
        result[mask1] = grid1[mask1]
        mask2 = grid2 != background
        result[mask2] = grid2[mask2]

    return result


def xor_grids(grid1, grid2, background=0):
    """Return XOR of two grids (cells different in each)."""
    result = np.full_like(grid1, background)
    diff_mask = grid1 != grid2
    result[diff_mask] = grid1[diff_mask]
    return result


def and_grids(grid1, grid2, background=0):
    """Return AND of two grids (cells same and non-background in both)."""
    result = np.full_like(grid1, background)
    same_mask = (grid1 == grid2) & (grid1 != background)
    result[same_mask] = grid1[same_mask]
    return result


def stack_layers(layers, background=0):
    """
    Stack multiple grid layers, later layers on top.

    Args:
        layers: List of grids
        background: Background color to treat as transparent

    Returns:
        Composed grid
    """
    if not layers:
        return None

    result = layers[0].copy()
    for layer in layers[1:]:
        mask = layer != background
        result[mask] = layer[mask]

    return result


__all__ = [
    'overlay_grids',
    'merge_grids',
    'xor_grids',
    'and_grids',
    'stack_layers'
]

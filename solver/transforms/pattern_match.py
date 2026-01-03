"""
Pattern matching and shape recognition.
"""

import numpy as np
from itertools import combinations
from ..utils.grid_utils import get_shape_signature, find_connected_regions


def find_pattern_matches(grid, pattern, background=None):
    """
    Find all positions where pattern matches in grid.

    Returns list of (row, col) positions of top-left corner.
    """
    if background is None:
        background = 0

    gh, gw = grid.shape
    ph, pw = pattern.shape
    matches = []

    for r in range(gh - ph + 1):
        for c in range(gw - pw + 1):
            subgrid = grid[r:r+ph, c:c+pw]
            if np.array_equal(subgrid, pattern):
                matches.append((r, c))

    return matches


def find_partial_matches(grid, pattern, background=0):
    """
    Find positions where pattern matches ignoring background.

    Pattern background cells can match anything.
    """
    gh, gw = grid.shape
    ph, pw = pattern.shape
    matches = []

    for r in range(gh - ph + 1):
        for c in range(gw - pw + 1):
            subgrid = grid[r:r+ph, c:c+pw]
            pattern_mask = pattern != background
            if np.array_equal(subgrid[pattern_mask], pattern[pattern_mask]):
                matches.append((r, c))

    return matches


def find_matching_subset(target_shape, positions):
    """
    Find a subset of positions that forms target shape.

    Used for partitioned marker matching.
    """
    n = len(target_shape)
    pos_list = [tuple(p) for p in positions]

    if len(pos_list) < n:
        return None

    for combo in combinations(pos_list, n):
        combo_shape = get_shape_signature(list(combo))
        if combo_shape == target_shape:
            return combo

    return None


def match_regions_by_shape(pattern_regions, marker_positions):
    """
    Match pattern regions to marker groups by shape.

    Each marker can only be used once (partitioning).

    Args:
        pattern_regions: List of (shape_signature, region_data) tuples
        marker_positions: Dict mapping color -> set of positions

    Returns:
        List of (region, matched_markers, shift) tuples
    """
    used_markers = {c: set() for c in marker_positions}
    matches = []

    # Sort by size (larger shapes first - harder to match)
    sorted_regions = sorted(pattern_regions, key=lambda x: -len(x[0]))

    for shape, region_data in sorted_regions:
        color = region_data.get('anchor_color')
        if color not in marker_positions:
            continue

        available = marker_positions[color] - used_markers[color]
        if len(available) < len(shape):
            continue

        match = find_matching_subset(shape, list(available))
        if match is None:
            continue

        # Mark as used
        for m in match:
            used_markers[color].add(m)

        matches.append((region_data, match))

    return matches


def extract_templates(grid, background=None):
    """
    Extract unique template patterns from grid.

    Returns list of (template_grid, positions) tuples.
    """
    if background is None:
        from ..utils.grid_utils import get_background_color
        background = get_background_color(grid)

    regions = find_connected_regions(grid, background)
    templates = []

    for positions, values, mask in regions:
        if len(positions) == 0:
            continue

        # Get bounding box
        min_r = positions[:, 0].min()
        min_c = positions[:, 1].min()
        max_r = positions[:, 0].max()
        max_c = positions[:, 1].max()

        h = max_r - min_r + 1
        w = max_c - min_c + 1

        template = np.full((h, w), background, dtype=grid.dtype)
        for pos, val in zip(positions, values):
            template[pos[0] - min_r, pos[1] - min_c] = val

        templates.append((template, (min_r, min_c)))

    return templates


__all__ = [
    'find_pattern_matches',
    'find_partial_matches',
    'find_matching_subset',
    'match_regions_by_shape',
    'extract_templates'
]

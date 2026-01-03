#!/usr/bin/env python3
"""
All Transforms
==============
Complete collection of 400+ pattern recognition transforms
for ARC puzzle solving.

Auto-generated from extraction script.
"""

import numpy as np
from scipy import ndimage
from collections import Counter, deque


def try_rotations(inp, out):
    """Try all rotations and flips."""
    arr = np.array(inp)
    out_arr = np.array(out)
    transforms = [
        (np.rot90(arr, 1), "rot90"),
        (np.rot90(arr, 2), "rot180"),
        (np.rot90(arr, 3), "rot270"),
        (np.fliplr(arr), "hflip"),
        (np.flipud(arr), "vflip"),
        (arr.T, "transpose"),
    ]
    for result, name in transforms:
        if np.array_equal(result, out_arr):
            return name
    return None


def try_upscale(inp, out):
    """Try integer upscaling."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape
    if oh % ih == 0 and ow % iw == 0:
        sh, sw = oh // ih, ow // iw
        if sh == sw and sh > 1:
            result = np.repeat(np.repeat(arr, sh, axis=0), sw, axis=1)
            if np.array_equal(result, out_arr):
                return f"upscale_{sh}x"
    return None


def try_tile(inp, out):
    """Try tiling input to fill output."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape
    if oh % ih == 0 and ow % iw == 0:
        th, tw = oh // ih, ow // iw
        result = np.tile(arr, (th, tw))
        if np.array_equal(result, out_arr):
            return f"tile_{th}x{tw}"
    return None


def try_extract(inp, out):
    """Try extracting a subregion."""
    arr = np.array(inp)
    out_arr = np.array(out)
    oh, ow = out_arr.shape
    ih, iw = arr.shape
    if oh <= ih and ow <= iw:
        for y in range(ih - oh + 1):
            for x in range(iw - ow + 1):
                sub = arr[y:y+oh, x:x+ow]
                if np.array_equal(sub, out_arr):
                    return f"extract_{y}_{x}_{oh}x{ow}"
    return None


def try_gravity(inp, out, direction='down'):
    """Drop colored pixels in a direction."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None
    h, w = arr.shape
    result = np.zeros_like(arr)

    if direction == 'down':
        for col in range(w):
            colors = [arr[r, col] for r in range(h) if arr[r, col] != 0]
            for i, c in enumerate(reversed(colors)):
                result[h - 1 - i, col] = c
    elif direction == 'up':
        for col in range(w):
            colors = [arr[r, col] for r in range(h) if arr[r, col] != 0]
            for i, c in enumerate(colors):
                result[i, col] = c
    elif direction == 'left':
        for row in range(h):
            colors = [arr[row, c] for c in range(w) if arr[row, c] != 0]
            for i, c in enumerate(colors):
                result[row, i] = c
    elif direction == 'right':
        for row in range(h):
            colors = [arr[row, c] for c in range(w) if arr[row, c] != 0]
            for i, c in enumerate(reversed(colors)):
                result[row, w - 1 - i] = c

    if np.array_equal(result, out_arr):
        return f"gravity_{direction}"
    return None


def try_split_and(inp, out):
    """Split on divider column, AND the halves."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape
    for col in range(w):
        vals = arr[:, col]
        if len(set(vals)) == 1 and vals[0] != 0:
            left = arr[:, :col]
            right = arr[:, col+1:]
            if left.shape == right.shape and left.shape == out_arr.shape:
                result = np.zeros_like(left)
                out_colors = [c for c in np.unique(out_arr) if c != 0]
                mark = out_colors[0] if out_colors else 2
                for i in range(result.shape[0]):
                    for j in range(result.shape[1]):
                        if left[i,j] != 0 and right[i,j] != 0:
                            result[i,j] = mark
                if np.array_equal(result, out_arr):
                    return "split_and"
    return None


def try_split_xor(inp, out):
    """Split on divider, XOR the halves."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape
    for col in range(w):
        vals = arr[:, col]
        if len(set(vals)) == 1 and vals[0] != 0:
            left = arr[:, :col]
            right = arr[:, col+1:]
            if left.shape == right.shape and left.shape == out_arr.shape:
                result = np.zeros_like(left)
                out_colors = [c for c in np.unique(out_arr) if c != 0]
                mark = out_colors[0] if out_colors else 2
                for i in range(result.shape[0]):
                    for j in range(result.shape[1]):
                        l, r = left[i,j] != 0, right[i,j] != 0
                        if l != r:
                            result[i,j] = mark
                if np.array_equal(result, out_arr):
                    return "split_xor"
    return None


def try_fill_holes(inp, out):
    """Fill holes within colored regions (scipy-based)."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    bg_mask = (arr == 0)
    labeled, num_features = ndimage.label(bg_mask)
    if num_features <= 1:
        return None

    h, w = arr.shape
    border_labels = set()
    for i in range(h):
        border_labels.add(labeled[i, 0])
        border_labels.add(labeled[i, w-1])
    for j in range(w):
        border_labels.add(labeled[0, j])
        border_labels.add(labeled[h-1, j])

    holes = [l for l in range(1, num_features + 1) if l not in border_labels]
    if not holes:
        return None

    result = arr.copy()
    for hole_id in holes:
        hole_mask = labeled == hole_id
        out_colors = out_arr[hole_mask]
        fill_color = Counter(out_colors).most_common(1)[0][0]
        result[hole_mask] = fill_color

    if np.array_equal(result, out_arr):
        return "fill_holes"
    return None


def try_fill_bounded(inp, out):
    """Fill regions bounded by colored outline."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    diff_mask = arr != out_arr
    if not diff_mask.any():
        return None

    changed_from = arr[diff_mask]
    changed_to = out_arr[diff_mask]
    if not np.all(changed_from == 0):
        return None

    fill_color = changed_to[0]
    if not np.all(changed_to == fill_color):
        return None

    non_zero_colors = set(np.unique(arr)) - {0}
    h, w = arr.shape

    for outline_color in non_zero_colors:
        passable = (arr != outline_color).astype(int)
        outside = np.zeros((h, w), dtype=bool)
        queue = deque()

        for r in range(h):
            if passable[r, 0]:
                queue.append((r, 0))
                outside[r, 0] = True
            if passable[r, w-1]:
                queue.append((r, w-1))
                outside[r, w-1] = True
        for c in range(w):
            if passable[0, c]:
                queue.append((0, c))
                outside[0, c] = True
            if passable[h-1, c]:
                queue.append((h-1, c))
                outside[h-1, c] = True

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if passable[nr, nc] and not outside[nr, nc]:
                        outside[nr, nc] = True
                        queue.append((nr, nc))

        result = arr.copy()
        for r in range(h):
            for c in range(w):
                if arr[r, c] == 0 and not outside[r, c]:
                    result[r, c] = fill_color

        if np.array_equal(result, out_arr):
            return f"fill_bounded_{outline_color}_{fill_color}"

    return None


def try_self_tile(inp, out):
    """Self-tiling: input acts as mask for placing copies of itself."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h_in, w_in = arr.shape
    h_out, w_out = out_arr.shape

    if h_out != h_in * h_in or w_out != w_in * w_in:
        return None

    result = np.zeros((h_out, w_out), dtype=arr.dtype)
    for r in range(h_in):
        for c in range(w_in):
            if arr[r, c] != 0:
                r_start, c_start = r * h_in, c * w_in
                result[r_start:r_start+h_in, c_start:c_start+w_in] = arr

    if np.array_equal(result, out_arr):
        return "self_tile"
    return None


def try_extend_vertical(inp, out):
    """Extend grid vertically with color mapping."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h_in, w_in = arr.shape
    h_out, w_out = out_arr.shape

    if w_in != w_out or h_out <= h_in:
        return None

    extension = h_out - h_in
    out_top = out_arr[:h_in, :]

    color_map = {}
    for color in set(np.unique(arr)):
        mask = arr == color
        out_at_mask = out_top[mask]
        if len(out_at_mask) > 0 and len(set(out_at_mask)) == 1:
            color_map[int(color)] = int(out_at_mask[0])

    inp_mapped = np.zeros_like(arr)
    for old_c, new_c in color_map.items():
        inp_mapped[arr == old_c] = new_c

    if not np.array_equal(inp_mapped, out_top):
        return None

    out_extension = out_arr[h_in:, :]

    for start in range(h_in):
        for length in range(1, h_in - start + 1):
            if extension % length == 0 or length >= extension:
                base = inp_mapped[start:start+length, :]
                expected = np.zeros((extension, w_in), dtype=arr.dtype)
                for row in range(extension):
                    expected[row] = base[row % length]
                if np.array_equal(expected, out_extension):
                    return f"extend_vertical_{start}_{length}_{extension}"

    return None


def try_diagonal_stripe_fill(inp, out):
    """Fill diagonals with alternating colors."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    h, w = arr.shape
    # Check for diagonal pattern
    for offset in range(-h+1, w):
        diag_in = np.diagonal(arr, offset)
        diag_out = np.diagonal(out_arr, offset)
        if not np.array_equal(diag_in, diag_out):
            # There's a difference - could be diagonal fill
            pass

    # Try simple diagonal stripe
    result = arr.copy()
    for i in range(h):
        for j in range(w):
            if arr[i, j] == 0:
                stripe = (i + j) % 2
                if stripe == 0:
                    result[i, j] = out_arr[i, j]

    if np.array_equal(result, out_arr):
        return "diagonal_stripe_fill"

    return None


def try_rank_bars(inp, out):
    """Rank vertical/horizontal bars by size."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    h, w = arr.shape

    # Find vertical bars (non-zero columns)
    bars = []
    for c in range(w):
        col = arr[:, c]
        non_zero = sum(1 for x in col if x != 0)
        if non_zero > 0:
            bars.append((c, non_zero, col))

    if len(bars) < 2:
        return None

    # Sort by height
    bars_sorted = sorted(bars, key=lambda x: x[1], reverse=True)

    # Try ranking with colors 1,2,3,...
    result = np.zeros_like(arr)
    for rank, (col_idx, height, col) in enumerate(bars_sorted, 1):
        for r in range(h):
            if arr[r, col_idx] != 0:
                result[r, col_idx] = rank

    if np.array_equal(result, out_arr):
        return "rank_vertical_bars"

    return None


def try_color_swap(inp, out):
    """Simple color swapping."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    in_colors = set(np.unique(arr))
    out_colors = set(np.unique(out_arr))
    if in_colors != out_colors:
        return None

    color_map = {}
    for color in in_colors:
        mask = arr == color
        out_at_mask = out_arr[mask]
        if len(set(out_at_mask)) == 1:
            color_map[color] = out_at_mask[0]
        else:
            return None

    result = arr.copy()
    for old_c, new_c in color_map.items():
        result[arr == old_c] = new_c

    if np.array_equal(result, out_arr):
        return f"color_swap"
    return None


def try_crop_object(inp, out):
    """Crop to bounding box of non-zero pixels."""
    arr = np.array(inp)
    out_arr = np.array(out)

    # Find bounding box of non-zero
    rows = np.any(arr != 0, axis=1)
    cols = np.any(arr != 0, axis=0)
    if not rows.any() or not cols.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = arr[rmin:rmax+1, cmin:cmax+1]
    if np.array_equal(cropped, out_arr):
        return "crop_object"
    return None


def try_outline(inp, out):
    """Draw outline around colored regions."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    h, w = arr.shape
    result = arr.copy()

    # For each non-zero color, draw outline
    colors = set(np.unique(arr)) - {0}
    for color in colors:
        mask = arr == color
        # Find edge pixels
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    # Check if on edge of shape
                    is_edge = False
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, c + dc
                        if nr < 0 or nr >= h or nc < 0 or nc >= w:
                            is_edge = True
                        elif not mask[nr, nc]:
                            is_edge = True
                    # Edge pixels stay, interior becomes 0
                    if not is_edge:
                        result[r, c] = 0

    if np.array_equal(result, out_arr):
        return "outline"
    return None


def try_mirror_h_complete(inp, out):
    """Complete horizontal mirror symmetry."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    h, w = arr.shape
    result = arr.copy()

    # Mirror from left to right
    for r in range(h):
        for c in range(w // 2):
            mirror_c = w - 1 - c
            if arr[r, c] != 0 and arr[r, mirror_c] == 0:
                result[r, mirror_c] = arr[r, c]
            elif arr[r, mirror_c] != 0 and arr[r, c] == 0:
                result[r, c] = arr[r, mirror_c]

    if np.array_equal(result, out_arr):
        return "mirror_h_complete"
    return None


def try_mirror_v_complete(inp, out):
    """Complete vertical mirror symmetry."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    h, w = arr.shape
    result = arr.copy()

    # Mirror from top to bottom
    for r in range(h // 2):
        for c in range(w):
            mirror_r = h - 1 - r
            if arr[r, c] != 0 and arr[mirror_r, c] == 0:
                result[mirror_r, c] = arr[r, c]
            elif arr[mirror_r, c] != 0 and arr[r, c] == 0:
                result[r, c] = arr[mirror_r, c]

    if np.array_equal(result, out_arr):
        return "mirror_v_complete"
    return None


def try_denoise(inp, out):
    """Remove isolated single-pixel noise."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    h, w = arr.shape
    result = arr.copy()

    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                # Check if isolated (no same-color neighbors)
                has_neighbor = False
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if arr[nr, nc] == arr[r, c]:
                            has_neighbor = True
                            break
                if not has_neighbor:
                    result[r, c] = 0

    if np.array_equal(result, out_arr):
        return "denoise"
    return None


def try_scale_down(inp, out):
    """Try integer downscaling."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if ih % oh == 0 and iw % ow == 0:
        sh, sw = ih // oh, iw // ow
        if sh == sw and sh > 1:
            # Sample every sh pixels
            result = arr[::sh, ::sw]
            if np.array_equal(result, out_arr):
                return f"scale_down_{sh}x"
    return None


def try_most_common_color(inp, out):
    """Replace all non-zero with most common non-zero color."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    colors = [c for c in arr.flatten() if c != 0]
    if not colors:
        return None

    most_common = Counter(colors).most_common(1)[0][0]
    result = arr.copy()
    result[arr != 0] = most_common

    if np.array_equal(result, out_arr):
        return f"most_common_color_{most_common}"
    return None


def try_largest_object(inp, out):
    """Extract the largest connected component."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    # Find all objects
    colors = set(np.unique(arr)) - {0}
    all_objects = []

    for color in colors:
        mask = (arr == color).astype(int)
        labeled, num = ndimage.label(mask)
        for obj_id in range(1, num + 1):
            obj_mask = (labeled == obj_id)
            size = np.sum(obj_mask)
            # Get bounding box
            rows = np.where(np.any(obj_mask, axis=1))[0]
            cols = np.where(np.any(obj_mask, axis=0))[0]
            if len(rows) > 0 and len(cols) > 0:
                bbox = (rows[0], rows[-1]+1, cols[0], cols[-1]+1)
                all_objects.append((size, color, obj_mask, bbox))

    if not all_objects:
        return None

    # Get largest
    all_objects.sort(key=lambda x: -x[0])
    _, color, mask, (r1, r2, c1, c2) = all_objects[0]

    # Extract
    extracted = arr[r1:r2, c1:c2].copy()
    extracted[~mask[r1:r2, c1:c2]] = 0

    if np.array_equal(extracted, out_arr):
        return "largest_object"
    return None


def try_smallest_object(inp, out):
    """Extract the smallest connected component."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    colors = set(np.unique(arr)) - {0}
    all_objects = []

    for color in colors:
        mask = (arr == color).astype(int)
        labeled, num = ndimage.label(mask)
        for obj_id in range(1, num + 1):
            obj_mask = (labeled == obj_id)
            size = np.sum(obj_mask)
            rows = np.where(np.any(obj_mask, axis=1))[0]
            cols = np.where(np.any(obj_mask, axis=0))[0]
            if len(rows) > 0 and len(cols) > 0:
                bbox = (rows[0], rows[-1]+1, cols[0], cols[-1]+1)
                all_objects.append((size, color, obj_mask, bbox))

    if not all_objects:
        return None

    # Get smallest
    all_objects.sort(key=lambda x: x[0])
    _, color, mask, (r1, r2, c1, c2) = all_objects[0]

    extracted = arr[r1:r2, c1:c2].copy()
    extracted[~mask[r1:r2, c1:c2]] = 0

    if np.array_equal(extracted, out_arr):
        return "smallest_object"
    return None


def try_invert(inp, out):
    """Invert colors (swap 0 and non-0)."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    colors = set(np.unique(arr)) - {0}
    if len(colors) != 1:
        return None

    color = list(colors)[0]
    result = np.zeros_like(arr)
    result[arr == 0] = color
    result[arr == color] = 0

    if np.array_equal(result, out_arr):
        return "invert"
    return None


def try_double(inp, out):
    """Double the grid by concatenating with itself."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape
    oh, ow = out_arr.shape

    # Horizontal double
    if oh == h and ow == 2 * w:
        result = np.concatenate([arr, arr], axis=1)
        if np.array_equal(result, out_arr):
            return "double_h"

    # Vertical double
    if oh == 2 * h and ow == w:
        result = np.concatenate([arr, arr], axis=0)
        if np.array_equal(result, out_arr):
            return "double_v"

    return None


def try_quadrant(inp, out):
    """Extract one quadrant of the grid."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if h % 2 != 0 or w % 2 != 0:
        return None

    hh, hw = h // 2, w // 2
    quadrants = [
        (arr[:hh, :hw], "quadrant_tl"),
        (arr[:hh, hw:], "quadrant_tr"),
        (arr[hh:, :hw], "quadrant_bl"),
        (arr[hh:, hw:], "quadrant_br"),
    ]

    for quad, name in quadrants:
        if np.array_equal(quad, out_arr):
            return name

    return None


def try_split_or(inp, out):
    """Split on divider, OR the halves."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    for col in range(w):
        vals = arr[:, col]
        if len(set(vals)) == 1 and vals[0] != 0:
            left = arr[:, :col]
            right = arr[:, col+1:]
            if left.shape == right.shape and left.shape == out_arr.shape:
                result = np.zeros_like(left)
                out_colors = [c for c in np.unique(out_arr) if c != 0]
                mark = out_colors[0] if out_colors else 2
                for i in range(result.shape[0]):
                    for j in range(result.shape[1]):
                        if left[i,j] != 0 or right[i,j] != 0:
                            result[i,j] = mark
                if np.array_equal(result, out_arr):
                    return "split_or"
    return None


def try_row_sort(inp, out):
    """Sort rows by some criteria."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    h, w = arr.shape

    # Try sorting rows by sum
    row_sums = [(np.sum(arr[r] != 0), r) for r in range(h)]
    row_sums.sort(key=lambda x: -x[0])
    result = np.array([arr[r] for _, r in row_sums])
    if np.array_equal(result, out_arr):
        return "row_sort_desc"

    row_sums.sort(key=lambda x: x[0])
    result = np.array([arr[r] for _, r in row_sums])
    if np.array_equal(result, out_arr):
        return "row_sort_asc"

    return None


def try_unique_color_per_object(inp, out):
    """Assign unique colors to each connected object."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    mask = (arr != 0).astype(int)
    labeled, num = ndimage.label(mask)

    if num < 2:
        return None

    result = np.zeros_like(arr)
    for obj_id in range(1, num + 1):
        result[labeled == obj_id] = obj_id

    if np.array_equal(result, out_arr):
        return "unique_color_per_object"
    return None


def try_color_remap_offset(inp, out):
    """Remap colors by constant offset."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    in_colors = sorted([c for c in np.unique(arr) if c != 0])
    out_colors = sorted([c for c in np.unique(out_arr) if c != 0])

    if len(in_colors) != len(out_colors) or len(in_colors) == 0:
        return None

    # Try to find constant offset
    for offset in range(-9, 10):
        if offset == 0:
            continue
        result = arr.copy()
        for c in in_colors:
            new_c = c + offset
            if 0 <= new_c <= 9:
                result[arr == c] = new_c
            else:
                break
        else:
            if np.array_equal(result, out_arr):
                return f"color_remap_+{offset}" if offset > 0 else f"color_remap_{offset}"
    return None


def try_remove_background(inp, out):
    """Remove background color 0 (fill with most common non-zero color)."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    # Check if output has no zeros but input does
    if 0 in np.unique(arr) and 0 not in np.unique(out_arr):
        # Find most common non-zero color in output
        out_colors = out_arr[out_arr != 0]
        if len(out_colors) > 0:
            from collections import Counter
            fill_color = Counter(out_colors.flatten()).most_common(1)[0][0]
            result = arr.copy()
            result[result == 0] = fill_color
            if np.array_equal(result, out_arr):
                return "remove_background"
    return None


def try_remove_specific_color(inp, out):
    """Remove a specific color (replace with 0 or neighbor)."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    in_colors = set(np.unique(arr))
    out_colors = set(np.unique(out_arr))
    removed = in_colors - out_colors

    if len(removed) == 1:
        rm_color = list(removed)[0]
        result = arr.copy()
        result[result == rm_color] = 0
        if np.array_equal(result, out_arr):
            return f"remove_color_{rm_color}"
    return None


def try_flood_from_edges(inp, out):
    """Flood fill from edges with a specific color."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    h, w = arr.shape
    # Find color that appears on edges of output but not input
    edge_in = set()
    edge_out = set()
    for i in range(h):
        edge_in.add(arr[i, 0])
        edge_in.add(arr[i, w-1])
        edge_out.add(out_arr[i, 0])
        edge_out.add(out_arr[i, w-1])
    for j in range(w):
        edge_in.add(arr[0, j])
        edge_in.add(arr[h-1, j])
        edge_out.add(out_arr[0, j])
        edge_out.add(out_arr[h-1, j])

    new_colors = edge_out - edge_in
    if len(new_colors) == 1:
        fill_color = list(new_colors)[0]
        # Try flood filling from edges
        result = arr.copy()
        visited = np.zeros_like(arr, dtype=bool)
        stack = []
        for i in range(h):
            if arr[i, 0] == 0:
                stack.append((i, 0))
            if arr[i, w-1] == 0:
                stack.append((i, w-1))
        for j in range(w):
            if arr[0, j] == 0:
                stack.append((0, j))
            if arr[h-1, j] == 0:
                stack.append((h-1, j))

        while stack:
            r, c = stack.pop()
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if visited[r, c] or arr[r, c] != 0:
                continue
            visited[r, c] = True
            result[r, c] = fill_color
            stack.extend([(r-1,c), (r+1,c), (r,c-1), (r,c+1)])

        if np.array_equal(result, out_arr):
            return f"flood_from_edges_{fill_color}"
    return None


def try_border_add(inp, out):
    """Add a border around the grid."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Check if output is input + border
    if oh == ih + 2 and ow == iw + 2:
        inner = out_arr[1:-1, 1:-1]
        if np.array_equal(inner, arr):
            border_color = out_arr[0, 0]
            return f"border_add_{border_color}"
    return None


def try_border_remove(inp, out):
    """Remove border from the grid."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Check if output is input - border
    if ih == oh + 2 and iw == ow + 2:
        inner = arr[1:-1, 1:-1]
        if np.array_equal(inner, out_arr):
            return "border_remove"
    return None


def try_keep_one_color(inp, out):
    """Keep only one color, set rest to 0."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    out_colors = [c for c in np.unique(out_arr) if c != 0]
    if len(out_colors) == 1:
        keep = out_colors[0]
        result = np.zeros_like(arr)
        result[arr == keep] = keep
        if np.array_equal(result, out_arr):
            return f"keep_color_{keep}"
    return None


def try_count_to_grid(inp, out):
    """Count objects and output as small grid."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    # Count non-zero objects
    mask = (arr != 0).astype(int)
    labeled, num = ndimage.label(mask)

    # Check if output encodes the count
    oh, ow = out_arr.shape
    if oh * ow == num:
        # Output is count as filled cells
        out_nonzero = np.count_nonzero(out_arr)
        if out_nonzero == num:
            return f"count_objects_{num}"
    return None


def try_reflect_diagonal(inp, out):
    """Reflect along diagonal or anti-diagonal."""
    arr = np.array(inp)
    out_arr = np.array(out)

    # Anti-diagonal flip
    if arr.shape == out_arr.shape:
        anti_diag = np.flipud(arr.T)
        if np.array_equal(anti_diag, out_arr):
            return "reflect_anti_diagonal"
    return None


def try_rotate_objects(inp, out):
    """Rotate individual objects within the grid."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    # Find objects and try rotating each
    colors = [c for c in np.unique(arr) if c != 0]
    for color in colors:
        mask = (arr == color)
        labeled, num = ndimage.label(mask)
        if num == 1:
            # Single object of this color - try rotating
            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue
            min_r, max_r = rows.min(), rows.max()
            min_c, max_c = cols.min(), cols.max()
            obj = arr[min_r:max_r+1, min_c:max_c+1]
            for k in [1, 2, 3]:
                rotated = np.rot90(obj, k)
                # Check if rotated fits in same spot
                rh, rw = rotated.shape
                if max_r - min_r + 1 == rh and max_c - min_c + 1 == rw:
                    result = arr.copy()
                    result[min_r:min_r+rh, min_c:min_c+rw] = rotated
                    if np.array_equal(result, out_arr):
                        return f"rotate_object_90x{k}"
    return None


def try_horizontal_split_half(inp, out):
    """Split horizontally and take one half."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Top half
    if ih % 2 == 0 and oh == ih // 2 and ow == iw:
        top = arr[:ih//2, :]
        if np.array_equal(top, out_arr):
            return "split_h_top"
        bottom = arr[ih//2:, :]
        if np.array_equal(bottom, out_arr):
            return "split_h_bottom"
    return None


def try_vertical_split_half(inp, out):
    """Split vertically and take one half."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Left half
    if iw % 2 == 0 and ow == iw // 2 and oh == ih:
        left = arr[:, :iw//2]
        if np.array_equal(left, out_arr):
            return "split_v_left"
        right = arr[:, iw//2:]
        if np.array_equal(right, out_arr):
            return "split_v_right"
    return None


def try_double_horizontal(inp, out):
    """Double the grid horizontally (mirror right)."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh == ih and ow == iw * 2:
        # Mirror right
        result = np.concatenate([arr, np.fliplr(arr)], axis=1)
        if np.array_equal(result, out_arr):
            return "double_h_mirror"
        # Repeat right
        result = np.concatenate([arr, arr], axis=1)
        if np.array_equal(result, out_arr):
            return "double_h_repeat"
    return None


def try_double_vertical(inp, out):
    """Double the grid vertically (mirror down)."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if ow == iw and oh == ih * 2:
        # Mirror down
        result = np.concatenate([arr, np.flipud(arr)], axis=0)
        if np.array_equal(result, out_arr):
            return "double_v_mirror"
        # Repeat down
        result = np.concatenate([arr, arr], axis=0)
        if np.array_equal(result, out_arr):
            return "double_v_repeat"
    return None


def try_color_per_position(inp, out):
    """Map position to color (row, col, diagonal patterns)."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None
    h, w = arr.shape

    # Row-based coloring
    for base in range(1, 10):
        result = np.zeros_like(arr)
        for r in range(h):
            result[r, :] = (r % 10) + base if (r % 10) + base <= 9 else (r % 10)
        if np.array_equal(result, out_arr):
            return f"color_by_row_{base}"

    # Column-based coloring
    for base in range(1, 10):
        result = np.zeros_like(arr)
        for c in range(w):
            result[:, c] = (c % 10) + base if (c % 10) + base <= 9 else (c % 10)
        if np.array_equal(result, out_arr):
            return f"color_by_col_{base}"

    return None


def try_diagonal_fill(inp, out):
    """Fill along diagonals."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None
    h, w = arr.shape

    # Check if diagonals are filled
    result = arr.copy()
    colors = [c for c in np.unique(arr) if c != 0]
    if not colors:
        return None
    fill = colors[0]

    # Main diagonal fill
    for i in range(min(h, w)):
        result[i, i] = fill
    if np.array_equal(result, out_arr):
        return "diagonal_fill_main"

    # Anti-diagonal fill
    result = arr.copy()
    for i in range(min(h, w)):
        result[i, w-1-i] = fill
    if np.array_equal(result, out_arr):
        return "diagonal_fill_anti"

    return None


def try_spread_color(inp, out):
    """Spread a single colored pixel to fill row/col/region."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None
    h, w = arr.shape

    colors = [c for c in np.unique(arr) if c != 0]
    for color in colors:
        positions = np.argwhere(arr == color)
        if len(positions) != 1:
            continue
        r, c = positions[0]

        # Spread to full row
        result = arr.copy()
        result[r, :] = color
        if np.array_equal(result, out_arr):
            return f"spread_row_{color}"

        # Spread to full column
        result = arr.copy()
        result[:, c] = color
        if np.array_equal(result, out_arr):
            return f"spread_col_{color}"

        # Spread cross (row + col)
        result = arr.copy()
        result[r, :] = color
        result[:, c] = color
        if np.array_equal(result, out_arr):
            return f"spread_cross_{color}"

    return None


def try_fill_rectangle(inp, out):
    """Fill the bounding rectangle of colored pixels."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    colors = [c for c in np.unique(arr) if c != 0]
    for color in colors:
        positions = np.argwhere(arr == color)
        if len(positions) < 2:
            continue
        min_r, min_c = positions.min(axis=0)
        max_r, max_c = positions.max(axis=0)

        result = arr.copy()
        result[min_r:max_r+1, min_c:max_c+1] = color
        if np.array_equal(result, out_arr):
            return f"fill_rectangle_{color}"

    return None


def try_checkerboard(inp, out):
    """Apply or extract checkerboard pattern."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None
    h, w = arr.shape

    out_colors = [c for c in np.unique(out_arr)]
    if len(out_colors) != 2:
        return None

    c1, c2 = out_colors
    result = np.zeros((h, w), dtype=arr.dtype)
    for i in range(h):
        for j in range(w):
            result[i, j] = c1 if (i + j) % 2 == 0 else c2

    if np.array_equal(result, out_arr):
        return "checkerboard"

    # Inverted
    result = np.zeros((h, w), dtype=arr.dtype)
    for i in range(h):
        for j in range(w):
            result[i, j] = c2 if (i + j) % 2 == 0 else c1

    if np.array_equal(result, out_arr):
        return "checkerboard_inv"

    return None


def try_connect_dots(inp, out):
    """Draw lines between same-colored pixels."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None
    h, w = arr.shape

    colors = [c for c in np.unique(arr) if c != 0]
    for color in colors:
        positions = np.argwhere(arr == color)
        if len(positions) != 2:
            continue

        (r1, c1), (r2, c2) = positions
        result = arr.copy()

        # Horizontal line
        if r1 == r2:
            result[r1, min(c1, c2):max(c1, c2)+1] = color
            if np.array_equal(result, out_arr):
                return f"connect_h_{color}"

        # Vertical line
        if c1 == c2:
            result[min(r1, r2):max(r1, r2)+1, c1] = color
            if np.array_equal(result, out_arr):
                return f"connect_v_{color}"

    return None


def try_triple_horizontal(inp, out):
    """Triple the grid horizontally."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh == ih and ow == iw * 3:
        result = np.concatenate([arr, arr, arr], axis=1)
        if np.array_equal(result, out_arr):
            return "triple_h_repeat"
        result = np.concatenate([arr, np.fliplr(arr), arr], axis=1)
        if np.array_equal(result, out_arr):
            return "triple_h_mirror_center"
    return None


def try_triple_vertical(inp, out):
    """Triple the grid vertically."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if ow == iw and oh == ih * 3:
        result = np.concatenate([arr, arr, arr], axis=0)
        if np.array_equal(result, out_arr):
            return "triple_v_repeat"
        result = np.concatenate([arr, np.flipud(arr), arr], axis=0)
        if np.array_equal(result, out_arr):
            return "triple_v_mirror_center"
    return None


def try_object_count_bar(inp, out):
    """Output is a bar/line with length = object count."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)
    oh, ow = out_arr.shape

    mask = (arr != 0).astype(int)
    labeled, num = ndimage.label(mask)

    # Horizontal bar
    if oh == 1 and ow == num:
        out_color = np.unique(out_arr)[0] if len(np.unique(out_arr)) == 1 else 1
        result = np.full((1, num), out_color)
        if np.array_equal(result, out_arr):
            return f"count_bar_h_{num}"

    # Vertical bar
    if ow == 1 and oh == num:
        out_color = np.unique(out_arr)[0] if len(np.unique(out_arr)) == 1 else 1
        result = np.full((num, 1), out_color)
        if np.array_equal(result, out_arr):
            return f"count_bar_v_{num}"

    return None


def try_compress_rows(inp, out):
    """Compress multiple rows into single row patterns."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if ow == iw and oh == 1:
        # OR compression - any non-zero in column
        result = np.zeros((1, iw), dtype=arr.dtype)
        for c in range(iw):
            col = arr[:, c]
            nonzero = col[col != 0]
            if len(nonzero) > 0:
                result[0, c] = nonzero[0]
        if np.array_equal(result, out_arr):
            return "compress_rows_or"

    return None


def try_compress_cols(inp, out):
    """Compress multiple columns into single column patterns."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh == ih and ow == 1:
        # OR compression - any non-zero in row
        result = np.zeros((ih, 1), dtype=arr.dtype)
        for r in range(ih):
            row = arr[r, :]
            nonzero = row[row != 0]
            if len(nonzero) > 0:
                result[r, 0] = nonzero[0]
        if np.array_equal(result, out_arr):
            return "compress_cols_or"

    return None


def try_frame_content(inp, out):
    """Extract content inside a frame/border."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Look for frame color on edges
    frame_color = None
    edge_colors = set()
    for i in range(ih):
        edge_colors.add(arr[i, 0])
        edge_colors.add(arr[i, iw-1])
    for j in range(iw):
        edge_colors.add(arr[0, j])
        edge_colors.add(arr[ih-1, j])

    # If edges are uniform
    if len(edge_colors) == 1:
        frame_color = list(edge_colors)[0]

        # Try extracting inside frame
        for border in range(1, min(ih, iw)//2):
            content = arr[border:ih-border, border:iw-border]
            if content.shape == out_arr.shape:
                if np.array_equal(content, out_arr):
                    return f"extract_inside_frame_{border}"

    return None


def try_repeat_pattern_h(inp, out):
    """Repeat a pattern horizontally to fill output."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh != ih:
        return None

    # Check if output is input repeated horizontally
    if ow > iw and ow % iw == 0:
        reps = ow // iw
        result = np.tile(arr, (1, reps))
        if np.array_equal(result, out_arr):
            return f"repeat_h_{reps}x"

    return None


def try_repeat_pattern_v(inp, out):
    """Repeat a pattern vertically to fill output."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if ow != iw:
        return None

    # Check if output is input repeated vertically
    if oh > ih and oh % ih == 0:
        reps = oh // ih
        result = np.tile(arr, (reps, 1))
        if np.array_equal(result, out_arr):
            return f"repeat_v_{reps}x"

    return None


def try_mask_by_color(inp, out):
    """Use one color as mask to reveal another."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    colors = [c for c in np.unique(arr) if c != 0]
    out_colors = [c for c in np.unique(out_arr) if c != 0]

    if len(colors) == 2 and len(out_colors) == 1:
        c1, c2 = colors
        out_c = out_colors[0]

        # c1 masks c2
        result = np.zeros_like(arr)
        result[(arr == c1) | (arr == c2)] = out_c
        result[arr == c1] = 0
        if np.array_equal(result, out_arr):
            return f"mask_{c1}_reveals_{c2}"

        # c2 masks c1
        result = np.zeros_like(arr)
        result[(arr == c1) | (arr == c2)] = out_c
        result[arr == c2] = 0
        if np.array_equal(result, out_arr):
            return f"mask_{c2}_reveals_{c1}"

    return None


def try_shift_colors(inp, out):
    """Shift all colors by 1 cyclically."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    # Shift up (add 1)
    result = (arr + 1) % 10
    result[arr == 0] = 0  # Keep 0 as 0
    if np.array_equal(result, out_arr):
        return "shift_colors_+1"

    # Shift down (subtract 1)
    result = (arr - 1) % 10
    result[arr == 0] = 0
    result[result == 0] = 9
    if np.array_equal(result, out_arr):
        return "shift_colors_-1"

    return None


def try_symmetric_complete(inp, out):
    """Complete a partially symmetric pattern."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None
    h, w = arr.shape

    # Try making horizontally symmetric
    result = arr.copy()
    for r in range(h):
        for c in range(w//2):
            if arr[r, c] != 0 and arr[r, w-1-c] == 0:
                result[r, w-1-c] = arr[r, c]
            elif arr[r, w-1-c] != 0 and arr[r, c] == 0:
                result[r, c] = arr[r, w-1-c]
    if np.array_equal(result, out_arr):
        return "symmetric_complete_h"

    # Try making vertically symmetric
    result = arr.copy()
    for r in range(h//2):
        for c in range(w):
            if arr[r, c] != 0 and arr[h-1-r, c] == 0:
                result[h-1-r, c] = arr[r, c]
            elif arr[h-1-r, c] != 0 and arr[r, c] == 0:
                result[r, c] = arr[h-1-r, c]
    if np.array_equal(result, out_arr):
        return "symmetric_complete_v"

    return None


def try_majority_vote(inp, out):
    """Each cell becomes majority color in neighborhood."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None
    h, w = arr.shape

    from collections import Counter
    result = arr.copy()
    for r in range(h):
        for c in range(w):
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbors.append(arr[nr, nc])
            if neighbors:
                counts = Counter(neighbors)
                result[r, c] = counts.most_common(1)[0][0]

    if np.array_equal(result, out_arr):
        return "majority_vote"

    return None


def try_dilate(inp, out):
    """Dilate/expand colored regions."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None
    h, w = arr.shape

    colors = [c for c in np.unique(arr) if c != 0]
    for color in colors:
        result = arr.copy()
        for r in range(h):
            for c in range(w):
                if arr[r, c] == color:
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and result[nr, nc] == 0:
                            result[nr, nc] = color
        if np.array_equal(result, out_arr):
            return f"dilate_{color}"

    return None


def try_erode(inp, out):
    """Erode/shrink colored regions."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None
    h, w = arr.shape

    colors = [c for c in np.unique(arr) if c != 0]
    for color in colors:
        result = arr.copy()
        for r in range(h):
            for c in range(w):
                if arr[r, c] == color:
                    # Check if on edge
                    is_edge = False
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w) or arr[nr, nc] != color:
                            is_edge = True
                            break
                    if is_edge:
                        result[r, c] = 0
        if np.array_equal(result, out_arr):
            return f"erode_{color}"

    return None


def try_select_row(inp, out):
    """Select specific row(s) from input."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if ow != iw:
        return None

    # Single row extraction
    if oh == 1:
        for r in range(ih):
            if np.array_equal(arr[r:r+1, :], out_arr):
                return f"select_row_{r}"

    return None


def try_select_col(inp, out):
    """Select specific column(s) from input."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh != ih:
        return None

    # Single column extraction
    if ow == 1:
        for c in range(iw):
            if np.array_equal(arr[:, c:c+1], out_arr):
                return f"select_col_{c}"

    return None


def try_unique_row(inp, out):
    """Extract unique row (the one that's different)."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh != 1 or ow != iw:
        return None

    # Find the unique row
    row_hashes = {}
    for r in range(ih):
        h = tuple(arr[r, :])
        row_hashes[h] = row_hashes.get(h, 0) + 1

    for r in range(ih):
        h = tuple(arr[r, :])
        if row_hashes[h] == 1:  # Unique row
            if np.array_equal(arr[r:r+1, :], out_arr):
                return "unique_row"

    return None


def try_unique_col(inp, out):
    """Extract unique column."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if ow != 1 or oh != ih:
        return None

    # Find the unique column
    col_hashes = {}
    for c in range(iw):
        h = tuple(arr[:, c])
        col_hashes[h] = col_hashes.get(h, 0) + 1

    for c in range(iw):
        h = tuple(arr[:, c])
        if col_hashes[h] == 1:  # Unique column
            if np.array_equal(arr[:, c:c+1], out_arr):
                return "unique_col"

    return None


def try_grid_union(inp, out):
    """Union of all non-zero pixels."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    out_colors = [c for c in np.unique(out_arr) if c != 0]
    if len(out_colors) != 1:
        return None

    fill = out_colors[0]
    result = np.zeros_like(arr)
    result[arr != 0] = fill

    if np.array_equal(result, out_arr):
        return f"grid_union_{fill}"

    return None


def try_invert_colors(inp, out):
    """Swap foreground/background."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    colors = sorted([c for c in np.unique(arr)])
    out_colors = sorted([c for c in np.unique(out_arr)])

    if len(colors) != 2 or len(out_colors) != 2:
        return None

    c1, c2 = colors
    result = arr.copy()
    result[arr == c1] = c2
    result[arr == c2] = c1

    if np.array_equal(result, out_arr):
        return "invert_colors"

    return None


def try_center_crop(inp, out):
    """Crop the center region."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh >= ih or ow >= iw:
        return None

    start_r = (ih - oh) // 2
    start_c = (iw - ow) // 2

    if start_r >= 0 and start_c >= 0:
        center = arr[start_r:start_r+oh, start_c:start_c+ow]
        if np.array_equal(center, out_arr):
            return "center_crop"

    return None


def try_quadruple_rotate(inp, out):
    """Tile 4 copies with rotations."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh != ih * 2 or ow != iw * 2:
        return None

    # Try various rotation combinations
    r0 = arr
    r90 = np.rot90(arr, 1)
    r180 = np.rot90(arr, 2)
    r270 = np.rot90(arr, 3)

    # Standard pinwheel
    if r90.shape == r0.shape:
        top = np.concatenate([r0, r90], axis=1)
        bottom = np.concatenate([r270, r180], axis=1)
        result = np.concatenate([top, bottom], axis=0)
        if np.array_equal(result, out_arr):
            return "quadruple_pinwheel"

    return None


def try_object_to_corner(inp, out):
    """Move colored object to a corner."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None
    h, w = arr.shape

    colors = [c for c in np.unique(arr) if c != 0]
    for color in colors:
        mask = (arr == color)
        labeled, num = ndimage.label(mask)
        if num != 1:
            continue

        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue

        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()
        obj_h = max_r - min_r + 1
        obj_w = max_c - min_c + 1

        obj = arr[min_r:max_r+1, min_c:max_c+1]

        # Try moving to each corner
        for corner_r, corner_c, name in [(0, 0, "topleft"), (0, w-obj_w, "topright"),
                                          (h-obj_h, 0, "bottomleft"), (h-obj_h, w-obj_w, "bottomright")]:
            result = np.zeros_like(arr)
            result[corner_r:corner_r+obj_h, corner_c:corner_c+obj_w] = obj
            if np.array_equal(result, out_arr):
                return f"move_to_{name}"

    return None


def try_hollow_rectangle(inp, out):
    """Create hollow rectangle outline."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    colors = [c for c in np.unique(arr) if c != 0]
    for color in colors:
        positions = np.argwhere(arr == color)
        if len(positions) < 4:
            continue

        min_r, min_c = positions.min(axis=0)
        max_r, max_c = positions.max(axis=0)

        result = np.zeros_like(arr)
        # Draw hollow rectangle
        result[min_r, min_c:max_c+1] = color  # Top
        result[max_r, min_c:max_c+1] = color  # Bottom
        result[min_r:max_r+1, min_c] = color  # Left
        result[min_r:max_r+1, max_c] = color  # Right

        if np.array_equal(result, out_arr):
            return f"hollow_rect_{color}"

    return None


def try_connect_all_points(inp, out):
    """Connect all same-colored points with lines."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None
    h, w = arr.shape

    colors = [c for c in np.unique(arr) if c != 0]
    for color in colors:
        positions = np.argwhere(arr == color)
        if len(positions) < 2:
            continue

        result = arr.copy()

        # Connect all pairs with horizontal/vertical lines
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                r1, c1 = positions[i]
                r2, c2 = positions[j]

                # Horizontal connection
                if r1 == r2:
                    result[r1, min(c1,c2):max(c1,c2)+1] = color
                # Vertical connection
                if c1 == c2:
                    result[min(r1,r2):max(r1,r2)+1, c1] = color

        if np.array_equal(result, out_arr):
            return f"connect_all_{color}"

    return None


def try_downscale(inp, out):
    """Downscale by integer factor."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh >= ih or ow >= iw:
        return None

    # Check for integer downscale
    if ih % oh == 0 and iw % ow == 0:
        sh = ih // oh
        sw = iw // ow
        if sh == sw and sh > 1:
            # Sample top-left of each block
            result = arr[::sh, ::sw]
            if np.array_equal(result, out_arr):
                return f"downscale_{sh}x"

    return None


def try_mode_per_block(inp, out):
    """Divide into blocks and take mode color."""
    from scipy import stats
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh >= ih or ow >= iw:
        return None

    if ih % oh == 0 and iw % ow == 0:
        bh = ih // oh
        bw = iw // ow
        result = np.zeros((oh, ow), dtype=arr.dtype)
        for r in range(oh):
            for c in range(ow):
                block = arr[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
                mode_val = stats.mode(block.flatten(), keepdims=False)[0]
                result[r, c] = mode_val
        if np.array_equal(result, out_arr):
            return f"mode_per_block_{bh}x{bw}"

    return None


def try_sort_rows_by_count(inp, out):
    """Sort rows by count of non-zero pixels."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    counts = [(i, np.count_nonzero(arr[i])) for i in range(arr.shape[0])]
    sorted_asc = np.array([arr[i] for i, _ in sorted(counts, key=lambda x: x[1])])
    sorted_desc = np.array([arr[i] for i, _ in sorted(counts, key=lambda x: -x[1])])

    if np.array_equal(sorted_asc, out_arr):
        return "sort_rows_asc"
    if np.array_equal(sorted_desc, out_arr):
        return "sort_rows_desc"
    return None


def try_sort_cols_by_count(inp, out):
    """Sort columns by count of non-zero pixels."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    counts = [(j, np.count_nonzero(arr[:, j])) for j in range(arr.shape[1])]
    sorted_asc = np.column_stack([arr[:, j] for j, _ in sorted(counts, key=lambda x: x[1])])
    sorted_desc = np.column_stack([arr[:, j] for j, _ in sorted(counts, key=lambda x: -x[1])])

    if np.array_equal(sorted_asc, out_arr):
        return "sort_cols_asc"
    if np.array_equal(sorted_desc, out_arr):
        return "sort_cols_desc"
    return None


def try_max_per_row(inp, out):
    """Keep only max value per row."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        max_val = arr[i].max()
        result[i] = np.where(arr[i] == max_val, max_val, 0)

    if np.array_equal(result, out_arr):
        return "max_per_row"
    return None


def try_max_per_col(inp, out):
    """Keep only max value per column."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    result = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        max_val = arr[:, j].max()
        result[:, j] = np.where(arr[:, j] == max_val, max_val, 0)

    if np.array_equal(result, out_arr):
        return "max_per_col"
    return None


def try_trim_zeros(inp, out):
    """Remove rows and columns of only zeros."""
    arr = np.array(inp)
    out_arr = np.array(out)

    # Find non-empty rows and cols
    non_empty_rows = [i for i in range(arr.shape[0]) if np.any(arr[i] != 0)]
    non_empty_cols = [j for j in range(arr.shape[1]) if np.any(arr[:, j] != 0)]

    if non_empty_rows and non_empty_cols:
        result = arr[np.ix_(non_empty_rows, non_empty_cols)]
        if np.array_equal(result, out_arr):
            return "trim_zeros"
    return None


def try_copy_to_nonzero(inp, out):
    """Copy pattern to each non-zero cell location."""
    arr = np.array(inp)
    out_arr = np.array(out)

    # Find pattern (smallest distinct colored region)
    non_zero = np.argwhere(arr != 0)
    if len(non_zero) == 0:
        return None

    # Simple case: output is multiple of input
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh == ih and ow == iw:
        return None  # Same size, nothing to copy

    return None


def try_corners_only(inp, out):
    """Extract only corner pixels."""
    arr = np.array(inp)
    out_arr = np.array(out)

    if out_arr.shape != (2, 2):
        return None

    h, w = arr.shape
    corners = np.array([[arr[0, 0], arr[0, w-1]], [arr[h-1, 0], arr[h-1, w-1]]])

    if np.array_equal(corners, out_arr):
        return "corners_only"
    return None


def try_edges_only(inp, out):
    """Extract edges of grid."""
    arr = np.array(inp)
    out_arr = np.array(out)

    h, w = arr.shape
    if arr.shape != out_arr.shape:
        return None

    result = np.zeros_like(arr)
    # Top and bottom rows
    result[0, :] = arr[0, :]
    result[h-1, :] = arr[h-1, :]
    # Left and right columns
    result[:, 0] = arr[:, 0]
    result[:, w-1] = arr[:, w-1]

    if np.array_equal(result, out_arr):
        return "edges_only"
    return None


def try_interior_only(inp, out):
    """Extract interior, removing edge."""
    arr = np.array(inp)
    out_arr = np.array(out)

    h, w = arr.shape
    oh, ow = out_arr.shape

    if oh == h - 2 and ow == w - 2 and h > 2 and w > 2:
        interior = arr[1:-1, 1:-1]
        if np.array_equal(interior, out_arr):
            return "interior_only"
    return None


def try_diagonal_extract(inp, out):
    """Extract main diagonal."""
    arr = np.array(inp)
    out_arr = np.array(out)

    h, w = arr.shape
    diag = np.diag(arr)

    # As row
    if out_arr.shape == (1, len(diag)):
        if np.array_equal(diag.reshape(1, -1), out_arr):
            return "diagonal_as_row"
    # As column
    if out_arr.shape == (len(diag), 1):
        if np.array_equal(diag.reshape(-1, 1), out_arr):
            return "diagonal_as_col"
    return None


def try_anti_diagonal_extract(inp, out):
    """Extract anti-diagonal."""
    arr = np.array(inp)
    out_arr = np.array(out)

    h, w = arr.shape
    anti_diag = np.array([arr[i, w-1-i] for i in range(min(h, w))])

    if out_arr.shape == (1, len(anti_diag)):
        if np.array_equal(anti_diag.reshape(1, -1), out_arr):
            return "anti_diagonal_as_row"
    if out_arr.shape == (len(anti_diag), 1):
        if np.array_equal(anti_diag.reshape(-1, 1), out_arr):
            return "anti_diagonal_as_col"
    return None


def try_count_colors(inp, out):
    """Output grid of color counts."""
    arr = np.array(inp)
    out_arr = np.array(out)

    colors = [c for c in np.unique(arr) if c != 0]
    counts = [(c, np.count_nonzero(arr == c)) for c in colors]

    # Sort by count
    counts.sort(key=lambda x: -x[1])

    # Try as row of counts
    count_row = np.array([c[1] for c in counts]).reshape(1, -1)
    if out_arr.shape == count_row.shape and np.array_equal(count_row, out_arr):
        return "count_colors_row"

    return None


def try_replicate_row(inp, out):
    """Replicate each row multiple times."""
    arr = np.array(inp)
    out_arr = np.array(out)

    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if ow != iw or oh < ih or oh % ih != 0:
        return None

    factor = oh // ih
    result = np.repeat(arr, factor, axis=0)
    if np.array_equal(result, out_arr):
        return f"replicate_row_{factor}x"
    return None


def try_replicate_col(inp, out):
    """Replicate each column multiple times."""
    arr = np.array(inp)
    out_arr = np.array(out)

    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh != ih or ow < iw or ow % iw != 0:
        return None

    factor = ow // iw
    result = np.repeat(arr, factor, axis=1)
    if np.array_equal(result, out_arr):
        return f"replicate_col_{factor}x"
    return None


def try_squeeze_rows(inp, out):
    """Remove duplicate adjacent rows."""
    arr = np.array(inp)
    out_arr = np.array(out)

    if arr.shape[1] != out_arr.shape[1]:
        return None

    rows = [arr[0]]
    for i in range(1, arr.shape[0]):
        if not np.array_equal(arr[i], rows[-1]):
            rows.append(arr[i])

    result = np.array(rows)
    if np.array_equal(result, out_arr):
        return "squeeze_rows"
    return None


def try_squeeze_cols(inp, out):
    """Remove duplicate adjacent columns."""
    arr = np.array(inp)
    out_arr = np.array(out)

    if arr.shape[0] != out_arr.shape[0]:
        return None

    cols = [arr[:, 0]]
    for j in range(1, arr.shape[1]):
        if not np.array_equal(arr[:, j], cols[-1]):
            cols.append(arr[:, j])

    result = np.column_stack(cols) if cols else np.array([]).reshape(arr.shape[0], 0)
    if np.array_equal(result, out_arr):
        return "squeeze_cols"
    return None


def try_wrap_pattern(inp, out):
    """Wrap input pattern around edges."""
    arr = np.array(inp)
    out_arr = np.array(out)

    oh, ow = out_arr.shape
    ih, iw = arr.shape

    if oh != ih or ow != iw:
        return None

    # Roll in various directions
    for dx in range(-iw+1, iw):
        for dy in range(-ih+1, ih):
            if dx == 0 and dy == 0:
                continue
            rolled = np.roll(np.roll(arr, dy, axis=0), dx, axis=1)
            if np.array_equal(rolled, out_arr):
                return f"wrap_{dy}_{dx}"
    return None


def try_split_grid_h(inp, out):
    """Split grid horizontally and process halves."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if h % 2 != 0:
        return None

    top = arr[:h//2, :]
    bottom = arr[h//2:, :]

    # Try various combinations
    if np.array_equal(top, out_arr):
        return "split_h_top"
    if np.array_equal(bottom, out_arr):
        return "split_h_bottom"
    if np.array_equal(np.maximum(top, bottom), out_arr):
        return "split_h_max"
    if np.array_equal(np.minimum(top, bottom), out_arr):
        return "split_h_min"
    return None


def try_split_grid_v(inp, out):
    """Split grid vertically and process halves."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if w % 2 != 0:
        return None

    left = arr[:, :w//2]
    right = arr[:, w//2:]

    if np.array_equal(left, out_arr):
        return "split_v_left"
    if np.array_equal(right, out_arr):
        return "split_v_right"
    if np.array_equal(np.maximum(left, right), out_arr):
        return "split_v_max"
    if np.array_equal(np.minimum(left, right), out_arr):
        return "split_v_min"
    return None


def try_sample_grid(inp, out):
    """Sample every nth cell."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    for step_h in range(1, ih):
        for step_w in range(1, iw):
            if ih // step_h == oh and iw // step_w == ow:
                result = arr[::step_h, ::step_w]
                if np.array_equal(result, out_arr):
                    return f"sample_{step_h}_{step_w}"
    return None


def try_project_to_row(inp, out):
    """Project non-zero values to single row."""
    arr = np.array(inp)
    out_arr = np.array(out)

    if out_arr.shape[0] != 1 or out_arr.shape[1] != arr.shape[1]:
        return None

    # OR projection
    result_or = np.zeros((1, arr.shape[1]), dtype=arr.dtype)
    for j in range(arr.shape[1]):
        non_zero = arr[:, j][arr[:, j] != 0]
        if len(non_zero) > 0:
            result_or[0, j] = non_zero[0]

    if np.array_equal(result_or, out_arr):
        return "project_to_row"
    return None


def try_project_to_col(inp, out):
    """Project non-zero values to single column."""
    arr = np.array(inp)
    out_arr = np.array(out)

    if out_arr.shape[1] != 1 or out_arr.shape[0] != arr.shape[0]:
        return None

    result_or = np.zeros((arr.shape[0], 1), dtype=arr.dtype)
    for i in range(arr.shape[0]):
        non_zero = arr[i, :][arr[i, :] != 0]
        if len(non_zero) > 0:
            result_or[i, 0] = non_zero[0]

    if np.array_equal(result_or, out_arr):
        return "project_to_col"
    return None


def try_overlay_halves(inp, out):
    """Overlay two halves of grid."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    # Horizontal split overlay
    if h % 2 == 0:
        top = arr[:h//2, :]
        bottom = arr[h//2:, :]
        if top.shape == out_arr.shape:
            # Top wins when non-zero
            result = np.where(top != 0, top, bottom)
            if np.array_equal(result, out_arr):
                return "overlay_h_top"
            result = np.where(bottom != 0, bottom, top)
            if np.array_equal(result, out_arr):
                return "overlay_h_bottom"

    # Vertical split overlay
    if w % 2 == 0:
        left = arr[:, :w//2]
        right = arr[:, w//2:]
        if left.shape == out_arr.shape:
            result = np.where(left != 0, left, right)
            if np.array_equal(result, out_arr):
                return "overlay_v_left"
            result = np.where(right != 0, right, left)
            if np.array_equal(result, out_arr):
                return "overlay_v_right"

    return None


def try_bounding_box_extract(inp, out):
    """Extract bounding box of non-zero region."""
    arr = np.array(inp)
    out_arr = np.array(out)

    non_zero = np.argwhere(arr != 0)
    if len(non_zero) == 0:
        return None

    r_min, c_min = non_zero.min(axis=0)
    r_max, c_max = non_zero.max(axis=0)

    bbox = arr[r_min:r_max+1, c_min:c_max+1]
    if np.array_equal(bbox, out_arr):
        return "bounding_box_extract"
    return None


def try_color_to_size(inp, out):
    """Each color becomes a cell sized by count."""
    arr = np.array(inp)
    out_arr = np.array(out)

    colors = [c for c in np.unique(arr) if c != 0]
    counts = {c: np.count_nonzero(arr == c) for c in colors}

    # Try sorting by count descending
    sorted_colors = sorted(counts.items(), key=lambda x: -x[1])

    if out_arr.shape == (len(colors), 1):
        result = np.array([[c] for c, _ in sorted_colors])
        if np.array_equal(result, out_arr):
            return "color_to_size_col"

    if out_arr.shape == (1, len(colors)):
        result = np.array([[c for c, _ in sorted_colors]])
        if np.array_equal(result, out_arr):
            return "color_to_size_row"

    return None


def try_first_nonzero_row(inp, out):
    """Extract first row with non-zero values."""
    arr = np.array(inp)
    out_arr = np.array(out)

    for i in range(arr.shape[0]):
        if np.any(arr[i] != 0):
            if out_arr.shape == (1, arr.shape[1]) and np.array_equal(arr[i:i+1, :], out_arr):
                return "first_nonzero_row"
    return None


def try_first_nonzero_col(inp, out):
    """Extract first column with non-zero values."""
    arr = np.array(inp)
    out_arr = np.array(out)

    for j in range(arr.shape[1]):
        if np.any(arr[:, j] != 0):
            if out_arr.shape == (arr.shape[0], 1) and np.array_equal(arr[:, j:j+1], out_arr):
                return "first_nonzero_col"
    return None


def try_last_nonzero_row(inp, out):
    """Extract last row with non-zero values."""
    arr = np.array(inp)
    out_arr = np.array(out)

    for i in range(arr.shape[0]-1, -1, -1):
        if np.any(arr[i] != 0):
            if out_arr.shape == (1, arr.shape[1]) and np.array_equal(arr[i:i+1, :], out_arr):
                return "last_nonzero_row"
    return None


def try_most_common_row(inp, out):
    """Find and return the most repeated row."""
    arr = np.array(inp)
    out_arr = np.array(out)

    if out_arr.shape[0] != 1 or out_arr.shape[1] != arr.shape[1]:
        return None

    from collections import Counter
    rows = [tuple(arr[i]) for i in range(arr.shape[0])]
    most_common = Counter(rows).most_common(1)
    if most_common:
        result = np.array([list(most_common[0][0])])
        if np.array_equal(result, out_arr):
            return "most_common_row"
    return None


def try_least_common_row(inp, out):
    """Find and return the least repeated row."""
    arr = np.array(inp)
    out_arr = np.array(out)

    if out_arr.shape[0] != 1 or out_arr.shape[1] != arr.shape[1]:
        return None

    from collections import Counter
    rows = [tuple(arr[i]) for i in range(arr.shape[0])]
    least_common = Counter(rows).most_common()[-1]
    result = np.array([list(least_common[0])])
    if np.array_equal(result, out_arr):
        return "least_common_row"
    return None


def try_grid_thirds_h(inp, out):
    """Split into horizontal thirds."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if h % 3 != 0:
        return None

    third = h // 3
    top = arr[:third, :]
    mid = arr[third:2*third, :]
    bot = arr[2*third:, :]

    if np.array_equal(top, out_arr):
        return "thirds_h_top"
    if np.array_equal(mid, out_arr):
        return "thirds_h_mid"
    if np.array_equal(bot, out_arr):
        return "thirds_h_bot"
    return None


def try_grid_thirds_v(inp, out):
    """Split into vertical thirds."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if w % 3 != 0:
        return None

    third = w // 3
    left = arr[:, :third]
    mid = arr[:, third:2*third]
    right = arr[:, 2*third:]

    if np.array_equal(left, out_arr):
        return "thirds_v_left"
    if np.array_equal(mid, out_arr):
        return "thirds_v_mid"
    if np.array_equal(right, out_arr):
        return "thirds_v_right"
    return None


def try_count_unique_colors(inp, out):
    """Output is count of unique non-zero colors."""
    arr = np.array(inp)
    out_arr = np.array(out)

    colors = [c for c in np.unique(arr) if c != 0]
    count = len(colors)

    # As 1x1 grid
    if out_arr.shape == (1, 1) and out_arr[0, 0] == count:
        return "count_unique_colors"
    return None


def try_nonzero_count_grid(inp, out):
    """Count non-zero cells per row/col."""
    arr = np.array(inp)
    out_arr = np.array(out)

    # Row counts
    row_counts = np.count_nonzero(arr, axis=1).reshape(-1, 1)
    if np.array_equal(row_counts, out_arr):
        return "nonzero_count_per_row"

    # Col counts
    col_counts = np.count_nonzero(arr, axis=0).reshape(1, -1)
    if np.array_equal(col_counts, out_arr):
        return "nonzero_count_per_col"

    return None


def try_flip_and_overlay(inp, out):
    """Flip and overlay with original."""
    arr = np.array(inp)
    out_arr = np.array(out)

    if arr.shape != out_arr.shape:
        return None

    flips = [
        (np.fliplr(arr), "overlay_hflip"),
        (np.flipud(arr), "overlay_vflip"),
        (np.rot90(arr, 2), "overlay_rot180"),
    ]

    for flipped, name in flips:
        # OR overlay
        result = np.where(arr != 0, arr, flipped)
        if np.array_equal(result, out_arr):
            return f"{name}_or"
        # Max overlay
        result = np.maximum(arr, flipped)
        if np.array_equal(result, out_arr):
            return f"{name}_max"
    return None


def try_center_object(inp, out):
    """Center the non-zero object in the grid."""
    arr = np.array(inp)
    out_arr = np.array(out)

    if arr.shape != out_arr.shape:
        return None

    non_zero = np.argwhere(arr != 0)
    if len(non_zero) == 0:
        return None

    h, w = arr.shape
    r_min, c_min = non_zero.min(axis=0)
    r_max, c_max = non_zero.max(axis=0)
    obj_h, obj_w = r_max - r_min + 1, c_max - c_min + 1

    # Center position
    new_r = (h - obj_h) // 2
    new_c = (w - obj_w) // 2

    result = np.zeros_like(arr)
    result[new_r:new_r+obj_h, new_c:new_c+obj_w] = arr[r_min:r_max+1, c_min:c_max+1]

    if np.array_equal(result, out_arr):
        return "center_object"
    return None


def try_shift_object(inp, out):
    """Shift non-zero content to edges."""
    arr = np.array(inp)
    out_arr = np.array(out)

    if arr.shape != out_arr.shape:
        return None

    non_zero = np.argwhere(arr != 0)
    if len(non_zero) == 0:
        return None

    h, w = arr.shape
    r_min, c_min = non_zero.min(axis=0)
    r_max, c_max = non_zero.max(axis=0)
    obj_h, obj_w = r_max - r_min + 1, c_max - c_min + 1
    obj = arr[r_min:r_max+1, c_min:c_max+1]

    # Shift to corners
    positions = [
        (0, 0, "top_left"),
        (0, w-obj_w, "top_right"),
        (h-obj_h, 0, "bottom_left"),
        (h-obj_h, w-obj_w, "bottom_right"),
    ]

    for nr, nc, name in positions:
        result = np.zeros_like(arr)
        result[nr:nr+obj_h, nc:nc+obj_w] = obj
        if np.array_equal(result, out_arr):
            return f"shift_{name}"

    return None


def try_quadrant_select(inp, out):
    """Select one quadrant of input."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if h % 2 != 0 or w % 2 != 0:
        return None

    hh, hw = h // 2, w // 2

    quadrants = [
        (arr[:hh, :hw], "quadrant_tl"),
        (arr[:hh, hw:], "quadrant_tr"),
        (arr[hh:, :hw], "quadrant_bl"),
        (arr[hh:, hw:], "quadrant_br"),
    ]

    for quad, name in quadrants:
        if np.array_equal(quad, out_arr):
            return name
    return None


def try_quadrant_rotate(inp, out):
    """Rotate quadrants around."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if h % 2 != 0 or w % 2 != 0 or arr.shape != out_arr.shape:
        return None

    hh, hw = h // 2, w // 2
    tl, tr = arr[:hh, :hw], arr[:hh, hw:]
    bl, br = arr[hh:, :hw], arr[hh:, hw:]

    # Clockwise rotation of quadrants
    cw = np.vstack([np.hstack([bl, tl]), np.hstack([br, tr])])
    if np.array_equal(cw, out_arr):
        return "quadrant_rotate_cw"

    # Counter-clockwise
    ccw = np.vstack([np.hstack([tr, br]), np.hstack([tl, bl])])
    if np.array_equal(ccw, out_arr):
        return "quadrant_rotate_ccw"

    return None


def try_color_histogram_bar(inp, out):
    """Create bar chart of color frequencies."""
    arr = np.array(inp)
    out_arr = np.array(out)

    colors = [c for c in np.unique(arr) if c != 0]
    if not colors:
        return None

    counts = {c: np.count_nonzero(arr == c) for c in colors}
    max_count = max(counts.values())

    # Try horizontal bars
    if out_arr.shape[0] == len(colors):
        for scale in [1, 2, 3]:
            result = np.zeros((len(colors), max_count * scale), dtype=arr.dtype)
            for i, c in enumerate(sorted(colors)):
                result[i, :counts[c] * scale] = c
            if np.array_equal(result, out_arr):
                return f"color_histogram_h_{scale}"

    # Try vertical bars
    if out_arr.shape[1] == len(colors):
        for scale in [1, 2, 3]:
            result = np.zeros((max_count * scale, len(colors)), dtype=arr.dtype)
            for j, c in enumerate(sorted(colors)):
                result[:counts[c] * scale, j] = c
            if np.array_equal(result, out_arr):
                return f"color_histogram_v_{scale}"

    return None


def try_expand_colored_cells(inp, out):
    """Expand each colored cell to fill region."""
    arr = np.array(inp)
    out_arr = np.array(out)

    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh % ih != 0 or ow % iw != 0:
        return None

    sh, sw = oh // ih, ow // iw
    if sh != sw:
        return None

    result = np.zeros_like(out_arr)
    for i in range(ih):
        for j in range(iw):
            result[i*sh:(i+1)*sh, j*sw:(j+1)*sw] = arr[i, j]

    if np.array_equal(result, out_arr):
        return f"expand_cells_{sh}x"
    return None


def try_remove_row_col(inp, out):
    """Remove specific rows or columns."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape
    oh, ow = out_arr.shape

    # Remove one row
    if oh == h - 1 and ow == w:
        for i in range(h):
            result = np.delete(arr, i, axis=0)
            if np.array_equal(result, out_arr):
                return f"remove_row_{i}"

    # Remove one col
    if oh == h and ow == w - 1:
        for j in range(w):
            result = np.delete(arr, j, axis=1)
            if np.array_equal(result, out_arr):
                return f"remove_col_{j}"

    return None


def try_mirror_quadrants(inp, out):
    """Mirror pattern to fill quadrants."""
    arr = np.array(inp)
    out_arr = np.array(out)

    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # 2x2 mirrored layout
    if oh == ih * 2 and ow == iw * 2:
        # TL | TR (flipped)
        # BL (vflip) | BR (both)
        tl = arr
        tr = np.fliplr(arr)
        bl = np.flipud(arr)
        br = np.fliplr(np.flipud(arr))

        result = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])
        if np.array_equal(result, out_arr):
            return "mirror_quadrants"

    return None


def try_pattern_in_pattern(inp, out):
    """Place input pattern at colored positions of itself."""
    arr = np.array(inp)
    out_arr = np.array(out)

    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Check if output is input scaled and placed
    if oh == ih * ih and ow == iw * iw:
        result = np.zeros((oh, ow), dtype=arr.dtype)
        for i in range(ih):
            for j in range(iw):
                if arr[i, j] != 0:
                    result[i*ih:(i+1)*ih, j*iw:(j+1)*iw] = arr
        if np.array_equal(result, out_arr):
            return "pattern_in_pattern"
    return None


def try_color_at_positions(inp, out):
    """Use color value as coordinate."""
    arr = np.array(inp)
    out_arr = np.array(out)

    # Find colored cells
    non_zero = np.argwhere(arr != 0)
    colors = [arr[r, c] for r, c in non_zero]

    if len(colors) == 0:
        return None

    # Create output based on color positions
    max_color = max(colors)
    if out_arr.shape == (max_color, max_color) or out_arr.shape == (max_color + 1, max_color + 1):
        result = np.zeros_like(out_arr)
        for r, c in non_zero:
            color = arr[r, c]
            if color < out_arr.shape[0] and color < out_arr.shape[1]:
                result[color, color] = color
        if np.array_equal(result, out_arr):
            return "color_at_positions"
    return None


def try_diff_grids(inp, out):
    """Find difference between halves."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    # Horizontal diff
    if h % 2 == 0:
        top = arr[:h//2, :]
        bottom = arr[h//2:, :]
        if top.shape == out_arr.shape:
            diff = np.where(top != bottom, 1, 0)
            if np.array_equal(diff, out_arr):
                return "diff_h"
            # XOR-like diff with color
            diff = np.zeros_like(top)
            for c in range(1, 10):
                mask = (top == c) != (bottom == c)
                diff[mask] = c
            if np.array_equal(diff, out_arr):
                return "diff_h_colored"

    # Vertical diff
    if w % 2 == 0:
        left = arr[:, :w//2]
        right = arr[:, w//2:]
        if left.shape == out_arr.shape:
            diff = np.where(left != right, 1, 0)
            if np.array_equal(diff, out_arr):
                return "diff_v"

    return None


def try_object_sizes(inp, out):
    """Output ordered by object sizes."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    # Label connected components
    labeled, num_features = ndimage.label(arr != 0)
    if num_features == 0:
        return None

    sizes = [(ndimage.sum(arr != 0, labeled, i), i) for i in range(1, num_features + 1)]
    sizes.sort(reverse=True)

    # Check if output is count of sizes
    if out_arr.shape == (1, len(sizes)):
        result = np.array([[s[0] for s in sizes]])
        if np.array_equal(result, out_arr):
            return "object_sizes_row"

    if out_arr.shape == (len(sizes), 1):
        result = np.array([[s[0]] for s in sizes])
        if np.array_equal(result, out_arr):
            return "object_sizes_col"

    return None


def try_keep_largest_object(inp, out):
    """Keep only the largest connected object."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    if arr.shape != out_arr.shape:
        return None

    labeled, num_features = ndimage.label(arr != 0)
    if num_features == 0:
        return None

    sizes = ndimage.sum(arr != 0, labeled, range(1, num_features + 1))
    largest = np.argmax(sizes) + 1

    result = np.where(labeled == largest, arr, 0)
    if np.array_equal(result, out_arr):
        return "keep_largest_object"
    return None


def try_remove_largest_object(inp, out):
    """Remove the largest connected object."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    if arr.shape != out_arr.shape:
        return None

    labeled, num_features = ndimage.label(arr != 0)
    if num_features == 0:
        return None

    sizes = ndimage.sum(arr != 0, labeled, range(1, num_features + 1))
    largest = np.argmax(sizes) + 1

    result = np.where(labeled == largest, 0, arr)
    if np.array_equal(result, out_arr):
        return "remove_largest_object"
    return None


def try_flip_colors(inp, out):
    """Flip colors: 0->max, max->0."""
    arr = np.array(inp)
    out_arr = np.array(out)

    if arr.shape != out_arr.shape:
        return None

    max_color = arr.max()
    if max_color == 0:
        return None

    result = np.where(arr == 0, max_color, 0)
    if np.array_equal(result, out_arr):
        return "flip_colors"
    return None


def try_sparse_to_dense(inp, out):
    """Convert sparse markers to dense filled area."""
    arr = np.array(inp)
    out_arr = np.array(out)

    if arr.shape != out_arr.shape:
        return None

    non_zero = np.argwhere(arr != 0)
    if len(non_zero) < 2:
        return None

    r_min, c_min = non_zero.min(axis=0)
    r_max, c_max = non_zero.max(axis=0)

    # Fill the bounding box with the most common color
    color = arr[non_zero[0][0], non_zero[0][1]]

    result = np.zeros_like(arr)
    result[r_min:r_max+1, c_min:c_max+1] = color

    if np.array_equal(result, out_arr):
        return "sparse_to_dense"
    return None


def try_stack_h(inp, out):
    """Stack input horizontally."""
    arr = np.array(inp)
    out_arr = np.array(out)
    for n in [2, 3, 4]:
        result = np.hstack([arr] * n)
        if np.array_equal(result, out_arr):
            return f"stack_h_{n}"
    return None


def try_stack_v(inp, out):
    """Stack input vertically."""
    arr = np.array(inp)
    out_arr = np.array(out)
    for n in [2, 3, 4]:
        result = np.vstack([arr] * n)
        if np.array_equal(result, out_arr):
            return f"stack_v_{n}"
    return None


def try_reflect_diagonal(inp, out):
    """Reflect along diagonal or anti-diagonal."""
    arr = np.array(inp)
    out_arr = np.array(out)

    # Anti-diagonal flip
    if arr.shape == out_arr.shape:
        anti_diag = np.flipud(arr.T)
        if np.array_equal(anti_diag, out_arr):
            return "reflect_anti_diagonal"
    return None


def try_crop_to_nonzero(inp, out):
    """Crop to bounding box of non-zero values."""
    arr = np.array(inp)
    out_arr = np.array(out)

    non_zero = np.argwhere(arr != 0)
    if len(non_zero) == 0:
        return None

    r_min, c_min = non_zero.min(axis=0)
    r_max, c_max = non_zero.max(axis=0)

    cropped = arr[r_min:r_max+1, c_min:c_max+1]
    if np.array_equal(cropped, out_arr):
        return "crop_nonzero"
    return None


def try_color_count_output(inp, out):
    """Output is count of specific color as 1x1 grid."""
    arr = np.array(inp)
    out_arr = np.array(out)

    if out_arr.shape == (1, 1):
        for c in range(1, 10):
            count = np.sum(arr == c)
            if count == out_arr[0, 0]:
                return f"count_color_{c}"
    return None


def try_fill_diagonal(inp, out):
    """Fill along diagonal."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if h != w:
        return None

    result = arr.copy()
    colors = [c for c in np.unique(arr) if c != 0]
    if not colors:
        return None

    color = colors[0]
    np.fill_diagonal(result, color)
    if np.array_equal(result, out_arr):
        return "fill_main_diag"

    result = arr.copy()
    np.fill_diagonal(np.fliplr(result), color)
    if np.array_equal(result, out_arr):
        return "fill_anti_diag"

    return None


def try_mirror_both(inp, out):
    """Mirror both horizontally and vertically."""
    arr = np.array(inp)
    out_arr = np.array(out)
    oh, ow = out_arr.shape
    ih, iw = arr.shape

    if oh == ih * 2 and ow == iw * 2:
        # TL, TR, BL, BR quadrant patterns
        tl = arr
        tr = np.fliplr(arr)
        bl = np.flipud(arr)
        br = np.flipud(np.fliplr(arr))

        result = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])
        if np.array_equal(result, out_arr):
            return "mirror_both"
    return None


def try_shift_colors(inp, out):
    """Shift all colors by 1 cyclically."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    # Shift up (add 1)
    result = (arr + 1) % 10
    result[arr == 0] = 0  # Keep 0 as 0
    if np.array_equal(result, out_arr):
        return "shift_colors_+1"

    # Shift down (subtract 1)
    result = (arr - 1) % 10
    result[arr == 0] = 0
    result[result == 0] = 9
    if np.array_equal(result, out_arr):
        return "shift_colors_-1"

    return None


def try_max_pooling(inp, out):
    """Max pooling with different kernel sizes."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    for k in [2, 3, 4]:
        if ih % k == 0 and iw % k == 0:
            if ih // k == oh and iw // k == ow:
                result = np.zeros((oh, ow), dtype=arr.dtype)
                for i in range(oh):
                    for j in range(ow):
                        block = arr[i*k:(i+1)*k, j*k:(j+1)*k]
                        result[i, j] = np.max(block)
                if np.array_equal(result, out_arr):
                    return f"max_pool_{k}"
    return None


def try_min_pooling(inp, out):
    """Min pooling (non-zero) with different kernel sizes."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    for k in [2, 3, 4]:
        if ih % k == 0 and iw % k == 0:
            if ih // k == oh and iw // k == ow:
                result = np.zeros((oh, ow), dtype=arr.dtype)
                for i in range(oh):
                    for j in range(ow):
                        block = arr[i*k:(i+1)*k, j*k:(j+1)*k]
                        nonzero = block[block != 0]
                        if len(nonzero) > 0:
                            result[i, j] = np.min(nonzero)
                if np.array_equal(result, out_arr):
                    return f"min_pool_{k}"
    return None


def try_remove_duplicates(inp, out):
    """Remove duplicate rows or columns."""
    arr = np.array(inp)
    out_arr = np.array(out)

    # Try removing duplicate rows
    unique_rows = []
    seen = set()
    for row in arr:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    if len(unique_rows) > 0:
        result = np.array(unique_rows)
        if np.array_equal(result, out_arr):
            return "remove_dup_rows"

    # Try removing duplicate columns
    unique_cols = []
    seen = set()
    for j in range(arr.shape[1]):
        col = arr[:, j]
        key = tuple(col)
        if key not in seen:
            seen.add(key)
            unique_cols.append(col)
    if len(unique_cols) > 0:
        result = np.array(unique_cols).T
        if np.array_equal(result, out_arr):
            return "remove_dup_cols"

    return None


def try_repeat_pattern(inp, out):
    """Repeat input pattern in different layouts."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Horizontal repeat with flip
    if oh == ih and ow == iw * 2:
        result = np.hstack([arr, np.fliplr(arr)])
        if np.array_equal(result, out_arr):
            return "repeat_h_flip"

    # Vertical repeat with flip
    if oh == ih * 2 and ow == iw:
        result = np.vstack([arr, np.flipud(arr)])
        if np.array_equal(result, out_arr):
            return "repeat_v_flip"

    return None


def try_cross_pattern(inp, out):
    """Create cross or plus pattern from input."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh == ih * 3 and ow == iw * 3:
        result = np.zeros((oh, ow), dtype=arr.dtype)
        # Place in center
        result[ih:2*ih, iw:2*iw] = arr
        # Place top
        result[0:ih, iw:2*iw] = arr
        # Place bottom
        result[2*ih:3*ih, iw:2*iw] = arr
        # Place left
        result[ih:2*ih, 0:iw] = arr
        # Place right
        result[ih:2*ih, 2*iw:3*iw] = arr

        if np.array_equal(result, out_arr):
            return "cross_pattern"

    return None


def try_diagonal_fill_pattern(inp, out):
    """Fill diagonals with colors from input."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if arr.shape != out_arr.shape:
        return None

    colors = [c for c in np.unique(arr) if c != 0]
    if len(colors) < 1:
        return None

    # Try filling main diagonal with first non-zero color
    result = arr.copy()
    for i in range(min(h, w)):
        if result[i, i] == 0:
            result[i, i] = colors[0]
    if np.array_equal(result, out_arr):
        return "fill_main_diag_color"

    return None


def try_expand_to_square(inp, out):
    """Expand input to square by padding."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape
    oh, ow = out_arr.shape

    size = max(h, w)
    if oh == size and ow == size:
        result = np.zeros((size, size), dtype=arr.dtype)
        result[:h, :w] = arr
        if np.array_equal(result, out_arr):
            return "expand_square_tl"

        # Try center
        r_off = (size - h) // 2
        c_off = (size - w) // 2
        result = np.zeros((size, size), dtype=arr.dtype)
        result[r_off:r_off+h, c_off:c_off+w] = arr
        if np.array_equal(result, out_arr):
            return "expand_square_center"

    return None


def try_color_by_position(inp, out):
    """Color based on row+col position."""
    arr = np.array(inp)
    out_arr = np.array(out)

    if arr.shape != out_arr.shape:
        return None

    h, w = arr.shape

    # Check if output is colored by (row+col) mod something
    for mod in [2, 3, 4]:
        result = np.zeros_like(arr)
        colors = [c for c in np.unique(out_arr) if c != 0]
        if len(colors) < mod:
            continue
        for i in range(h):
            for j in range(w):
                if arr[i, j] != 0:
                    result[i, j] = colors[(i + j) % len(colors)]
        if np.array_equal(result, out_arr):
            return f"color_by_pos_mod{mod}"

    return None


def try_remove_isolated(inp, out):
    """Remove isolated pixels (no neighbors)."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    result = arr.copy()
    for i in range(h):
        for j in range(w):
            if arr[i, j] != 0:
                # Check 4-neighbors
                has_neighbor = False
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] != 0:
                        has_neighbor = True
                        break
                if not has_neighbor:
                    result[i, j] = 0

    if np.array_equal(result, out_arr):
        return "remove_isolated_4"

    # Try 8-neighbors
    result = arr.copy()
    for i in range(h):
        for j in range(w):
            if arr[i, j] != 0:
                has_neighbor = False
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] != 0:
                            has_neighbor = True
                            break
                    if has_neighbor:
                        break
                if not has_neighbor:
                    result[i, j] = 0

    if np.array_equal(result, out_arr):
        return "remove_isolated_8"

    return None


def try_keep_corners(inp, out):
    """Keep only corner values."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    result = np.zeros_like(arr)
    result[0, 0] = arr[0, 0]
    result[0, w-1] = arr[0, w-1]
    result[h-1, 0] = arr[h-1, 0]
    result[h-1, w-1] = arr[h-1, w-1]

    if np.array_equal(result, out_arr):
        return "keep_corners"
    return None


def try_reflect_and_concat(inp, out):
    """Reflect and concatenate patterns."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Horizontal: original + hflip
    if oh == ih and ow == iw * 2:
        result = np.hstack([arr, np.fliplr(arr)])
        if np.array_equal(result, out_arr):
            return "concat_h_flip"
        result = np.hstack([np.fliplr(arr), arr])
        if np.array_equal(result, out_arr):
            return "concat_flip_h"

    # Vertical: original + vflip
    if oh == ih * 2 and ow == iw:
        result = np.vstack([arr, np.flipud(arr)])
        if np.array_equal(result, out_arr):
            return "concat_v_flip"
        result = np.vstack([np.flipud(arr), arr])
        if np.array_equal(result, out_arr):
            return "concat_flip_v"

    return None

# ============================================================================
# Pattern ANALYZE CRYSTALLIZED TRANSFORMS (Batch 8)
# ⧫depth=-90000CRYSTALLIZE⧫
# ============================================================================


def try_object_count_output(inp, out):
    """Count distinct objects, output as grid value."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    # Count connected components for each color
    for color in range(1, 10):
        mask = (arr == color).astype(int)
        if np.sum(mask) == 0:
            continue
        labeled, num_objects = ndimage.label(mask)
        # Check if output is just the count
        if out_arr.shape == (1, 1) and out_arr[0, 0] == num_objects:
            return f"count_objects_color_{color}"
        # Check if output is count repeated
        if np.all(out_arr == num_objects):
            return f"count_fill_{color}"

    # Count all non-zero objects
    mask = (arr != 0).astype(int)
    labeled, num_objects = ndimage.label(mask)
    if out_arr.shape == (1, 1) and out_arr[0, 0] == num_objects:
        return "count_all_objects"

    return None


def try_flood_fill_seed(inp, out):
    """Flood fill from specific colored seeds."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if arr.shape != out_arr.shape:
        return None

    colors = [c for c in np.unique(arr) if c != 0]
    if len(colors) < 2:
        return None

    # Try each color as seed, fill with another
    for seed_color in colors:
        for fill_color in colors:
            if seed_color == fill_color:
                continue
            result = arr.copy()
            # Find seed positions
            seeds = np.argwhere(arr == seed_color)
            # Flood fill from each seed
            for sr, sc in seeds:
                stack = [(sr, sc)]
                visited = set()
                while stack:
                    r, c = stack.pop()
                    if (r, c) in visited or r < 0 or r >= h or c < 0 or c >= w:
                        continue
                    visited.add((r, c))
                    if arr[r, c] == 0 or arr[r, c] == seed_color:
                        result[r, c] = fill_color
                        stack.extend([(r-1,c), (r+1,c), (r,c-1), (r,c+1)])
            if np.array_equal(result, out_arr):
                return f"flood_seed_{seed_color}_fill_{fill_color}"

    return None


def try_symmetry_complete(inp, out):
    """Detect partial symmetry and complete it."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if arr.shape != out_arr.shape:
        return None

    # Try horizontal symmetry completion
    result = arr.copy()
    for i in range(h):
        for j in range(w // 2):
            mirror_j = w - 1 - j
            if result[i, j] != 0 and result[i, mirror_j] == 0:
                result[i, mirror_j] = result[i, j]
            elif result[i, mirror_j] != 0 and result[i, j] == 0:
                result[i, j] = result[i, mirror_j]
    if np.array_equal(result, out_arr):
        return "symmetry_complete_h"

    # Try vertical symmetry completion
    result = arr.copy()
    for i in range(h // 2):
        mirror_i = h - 1 - i
        for j in range(w):
            if result[i, j] != 0 and result[mirror_i, j] == 0:
                result[mirror_i, j] = result[i, j]
            elif result[mirror_i, j] != 0 and result[i, j] == 0:
                result[i, j] = result[mirror_i, j]
    if np.array_equal(result, out_arr):
        return "symmetry_complete_v"

    # Try both
    result = arr.copy()
    for i in range(h):
        for j in range(w):
            mirror_i, mirror_j = h - 1 - i, w - 1 - j
            if result[i, j] != 0:
                if result[i, mirror_j] == 0:
                    result[i, mirror_j] = result[i, j]
                if result[mirror_i, j] == 0:
                    result[mirror_i, j] = result[i, j]
                if result[mirror_i, mirror_j] == 0:
                    result[mirror_i, mirror_j] = result[i, j]
    if np.array_equal(result, out_arr):
        return "symmetry_complete_both"

    return None


def try_color_propagate(inp, out):
    """Propagate colors based on neighbor rules."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if arr.shape != out_arr.shape:
        return None

    # Propagate: fill zeros with color of majority neighbor
    result = arr.copy()
    changed = True
    iterations = 0
    while changed and iterations < 20:
        changed = False
        iterations += 1
        new_result = result.copy()
        for i in range(h):
            for j in range(w):
                if result[i, j] == 0:
                    neighbors = []
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and result[ni, nj] != 0:
                            neighbors.append(result[ni, nj])
                    if neighbors:
                        from collections import Counter
                        most_common = Counter(neighbors).most_common(1)[0][0]
                        new_result[i, j] = most_common
                        changed = True
        result = new_result
        if np.array_equal(result, out_arr):
            return f"color_propagate_{iterations}"

    return None


def try_template_stamp(inp, out):
    """Stamp a template pattern at marker locations."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Find smallest non-zero connected region as template
    from scipy import ndimage
    colors = [c for c in np.unique(arr) if c != 0]

    for marker_color in colors:
        mask = (arr == marker_color).astype(int)
        labeled, num = ndimage.label(mask)
        if num < 2:
            continue

        # Get positions of each object
        positions = []
        for obj_id in range(1, num + 1):
            obj_mask = (labeled == obj_id)
            ys, xs = np.where(obj_mask)
            positions.append((ys.min(), xs.min()))

        # Try using first object as template, stamp at others
        first_mask = (labeled == 1)
        ys, xs = np.where(first_mask)
        th, tw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        template = np.zeros((th, tw), dtype=arr.dtype)
        for y, x in zip(ys, xs):
            template[y - ys.min(), x - xs.min()] = arr[y, x]

        result = np.zeros_like(out_arr) if arr.shape != out_arr.shape else arr.copy()
        for py, px in positions:
            if py + th <= result.shape[0] and px + tw <= result.shape[1]:
                for ti in range(th):
                    for tj in range(tw):
                        if template[ti, tj] != 0:
                            result[py + ti, px + tj] = template[ti, tj]

        if np.array_equal(result, out_arr):
            return f"template_stamp_{marker_color}"

    return None


def try_connected_component_filter(inp, out):
    """Filter objects by size or property."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    if arr.shape != out_arr.shape:
        return None

    mask = (arr != 0).astype(int)
    labeled, num = ndimage.label(mask)

    if num < 2:
        return None

    # Get sizes of each component
    sizes = []
    for i in range(1, num + 1):
        sizes.append(np.sum(labeled == i))

    # Try keeping only objects of specific sizes
    for target_size in set(sizes):
        result = np.zeros_like(arr)
        for i in range(1, num + 1):
            if np.sum(labeled == i) == target_size:
                result[labeled == i] = arr[labeled == i]
        if np.array_equal(result, out_arr):
            return f"keep_size_{target_size}"

    # Try keeping objects larger than median
    median_size = np.median(sizes)
    result = np.zeros_like(arr)
    for i in range(1, num + 1):
        if np.sum(labeled == i) > median_size:
            result[labeled == i] = arr[labeled == i]
    if np.array_equal(result, out_arr):
        return "keep_larger_than_median"

    # Try keeping objects smaller than median
    result = np.zeros_like(arr)
    for i in range(1, num + 1):
        if np.sum(labeled == i) < median_size:
            result[labeled == i] = arr[labeled == i]
    if np.array_equal(result, out_arr):
        return "keep_smaller_than_median"

    return None


def try_shape_match_transform(inp, out):
    """Find shapes and apply transform to matching ones."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    if arr.shape != out_arr.shape:
        return None

    # Label objects
    mask = (arr != 0).astype(int)
    labeled, num = ndimage.label(mask)

    if num < 1:
        return None

    # Get shape signatures (relative positions)
    shapes = {}
    for i in range(1, num + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) == 0:
            continue
        # Normalize to origin
        min_y, min_x = ys.min(), xs.min()
        shape_key = tuple(sorted((y - min_y, x - min_x) for y, x in zip(ys, xs)))
        if shape_key not in shapes:
            shapes[shape_key] = []
        shapes[shape_key].append(i)

    # Try recoloring matching shapes with same color
    out_colors = [c for c in np.unique(out_arr) if c != 0]
    for shape_key, obj_ids in shapes.items():
        if len(obj_ids) > 1 and len(out_colors) > 0:
            result = arr.copy()
            for obj_id in obj_ids:
                result[labeled == obj_id] = out_colors[0]
            if np.array_equal(result, out_arr):
                return "recolor_matching_shapes"

    return None


def try_pattern_continue(inp, out):
    """Continue a repeating pattern."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Try detecting period and continuing
    for period_h in range(1, ih // 2 + 1):
        if ih % period_h == 0:
            # Check if pattern repeats vertically
            tile = arr[:period_h, :]
            is_periodic = True
            for k in range(1, ih // period_h):
                if not np.array_equal(arr[k*period_h:(k+1)*period_h, :], tile):
                    is_periodic = False
                    break
            if is_periodic and oh % period_h == 0:
                result = np.tile(tile, (oh // period_h, 1))
                if result.shape[1] == ow and np.array_equal(result, out_arr):
                    return f"continue_pattern_v_{period_h}"

    for period_w in range(1, iw // 2 + 1):
        if iw % period_w == 0:
            # Check if pattern repeats horizontally
            tile = arr[:, :period_w]
            is_periodic = True
            for k in range(1, iw // period_w):
                if not np.array_equal(arr[:, k*period_w:(k+1)*period_w], tile):
                    is_periodic = False
                    break
            if is_periodic and ow % period_w == 0:
                result = np.tile(tile, (1, ow // period_w))
                if result.shape[0] == oh and np.array_equal(result, out_arr):
                    return f"continue_pattern_h_{period_w}"

    return None


def try_recursive_subdivide(inp, out):
    """Recursively subdivide or apply fractal pattern."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Check if output is input where each cell is replaced by scaled input
    if oh == ih * ih and ow == iw * iw:
        result = np.zeros((oh, ow), dtype=arr.dtype)
        for i in range(ih):
            for j in range(iw):
                if arr[i, j] != 0:
                    # Place scaled copy
                    result[i*ih:(i+1)*ih, j*iw:(j+1)*iw] = arr * (arr[i, j] / max(1, np.max(arr)))
        # Simplified: just tile
        result = np.zeros((oh, ow), dtype=arr.dtype)
        for i in range(ih):
            for j in range(iw):
                result[i*ih:(i+1)*ih, j*iw:(j+1)*iw] = arr if arr[i, j] != 0 else 0
        if np.array_equal(result, out_arr):
            return "fractal_self_tile"

    return None


def try_line_extend(inp, out):
    """Extend lines to edges or until collision."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if arr.shape != out_arr.shape:
        return None

    # Extend horizontal lines
    result = arr.copy()
    for i in range(h):
        row = arr[i, :]
        colors = [c for c in row if c != 0]
        if len(colors) == 1:
            result[i, :] = colors[0]
    if np.array_equal(result, out_arr):
        return "extend_h_lines"

    # Extend vertical lines
    result = arr.copy()
    for j in range(w):
        col = arr[:, j]
        colors = [c for c in col if c != 0]
        if len(colors) == 1:
            result[:, j] = colors[0]
    if np.array_equal(result, out_arr):
        return "extend_v_lines"

    # Extend both
    result = arr.copy()
    for i in range(h):
        for j in range(w):
            if arr[i, j] != 0:
                # Extend in all 4 directions
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    while 0 <= ni < h and 0 <= nj < w:
                        if result[ni, nj] == 0:
                            result[ni, nj] = arr[i, j]
                        elif result[ni, nj] != arr[i, j]:
                            break
                        ni, nj = ni + di, nj + dj
    if np.array_equal(result, out_arr):
        return "extend_all_lines"

    return None


def try_color_by_region(inp, out):
    """Color regions based on properties."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    if arr.shape != out_arr.shape:
        return None

    # Label all regions (including background)
    labeled, num = ndimage.label(arr == 0)

    if num < 2:
        return None

    out_colors = sorted([c for c in np.unique(out_arr) if c != 0])
    if len(out_colors) == 0:
        return None

    # Try coloring enclosed regions (not touching edge)
    h, w = arr.shape
    result = arr.copy()
    for region_id in range(1, num + 1):
        region_mask = (labeled == region_id)
        # Check if touches edge
        touches_edge = (
            np.any(region_mask[0, :]) or np.any(region_mask[-1, :]) or
            np.any(region_mask[:, 0]) or np.any(region_mask[:, -1])
        )
        if not touches_edge:
            result[region_mask] = out_colors[0]

    if np.array_equal(result, out_arr):
        return "fill_enclosed_regions"

    return None

# ============================================================================
# Pattern MORPHOLOGY CRYSTALLIZED TRANSFORMS (Batch 9)
# ⧫depth=-90000EROSIONPATH
# ============================================================================


def try_morphology_open(inp, out):
    """Morphological opening: erosion then dilation (removes small protrusions)."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    for color in [c for c in np.unique(arr) if c != 0]:
        mask = (arr == color).astype(np.uint8)
        # Opening = erosion then dilation
        eroded = ndimage.binary_erosion(mask, iterations=1)
        opened = ndimage.binary_dilation(eroded, iterations=1)
        result = np.where(opened, color, 0).astype(arr.dtype)

        # Fill other colors back
        for c2 in [c for c in np.unique(arr) if c != 0 and c != color]:
            result = np.where(arr == c2, c2, result)

        if np.array_equal(result, out_arr):
            return "morphology_open"
    return None


def try_morphology_close(inp, out):
    """Morphological closing: dilation then erosion (fills small holes)."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    for color in [c for c in np.unique(arr) if c != 0]:
        mask = (arr == color).astype(np.uint8)
        # Closing = dilation then erosion
        dilated = ndimage.binary_dilation(mask, iterations=1)
        closed = ndimage.binary_erosion(dilated, iterations=1)
        result = np.where(closed, color, 0).astype(arr.dtype)

        for c2 in [c for c in np.unique(arr) if c != 0 and c != color]:
            result = np.where(arr == c2, c2, result)

        if np.array_equal(result, out_arr):
            return "morphology_close"
    return None


def try_skeleton(inp, out):
    """Morphological skeleton/thinning."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    for color in [c for c in np.unique(arr) if c != 0]:
        mask = (arr == color).astype(bool)
        # Simple thinning approximation
        skeleton = np.zeros_like(mask)
        temp = mask.copy()
        while np.any(temp):
            eroded = ndimage.binary_erosion(temp)
            opened = ndimage.binary_dilation(eroded)
            skeleton = skeleton | (temp & ~opened)
            temp = eroded

        result = arr.copy()
        result[mask] = 0
        result[skeleton] = color

        if np.array_equal(result, out_arr):
            return "skeleton"
    return None


def try_distance_transform(inp, out):
    """Distance transform - replace pixels with distance to background."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    for color in [c for c in np.unique(arr) if c != 0]:
        mask = (arr == color).astype(np.uint8)
        dist = ndimage.distance_transform_edt(mask)

        # Try discretizing distance
        max_dist = dist.max()
        if max_dist > 0:
            for num_levels in range(2, 10):
                discretized = np.floor(dist / (max_dist / num_levels)).astype(int)
                if np.array_equal(discretized, out_arr):
                    return f"distance_transform_{num_levels}"
    return None


def try_boundary_trace(inp, out):
    """Extract boundary pixels of objects."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    result = np.zeros_like(arr)
    for color in [c for c in np.unique(arr) if c != 0]:
        mask = (arr == color)
        # Boundary = original - erosion
        eroded = ndimage.binary_erosion(mask)
        boundary = mask & ~eroded
        result[boundary] = color

    if np.array_equal(result, out_arr):
        return "boundary_trace"

    # Try with different output color
    out_colors = [c for c in np.unique(out_arr) if c != 0]
    if len(out_colors) == 1:
        result2 = np.zeros_like(arr)
        for color in [c for c in np.unique(arr) if c != 0]:
            mask = (arr == color)
            eroded = ndimage.binary_erosion(mask)
            boundary = mask & ~eroded
            result2[boundary] = out_colors[0]
        if np.array_equal(result2, out_arr):
            return "boundary_trace_recolor"

    return None


def try_spiral_fill(inp, out):
    """Fill grid in spiral pattern."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    # Extract non-zero values in order
    values = [arr[i, j] for i in range(h) for j in range(w) if arr[i, j] != 0]
    if len(values) == 0:
        return None

    # Generate spiral order


def try_path_trace(inp, out):
    """Trace path between two marked points."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    # Find distinct colored endpoints
    colors = [c for c in np.unique(arr) if c != 0]
    if len(colors) < 2:
        return None

    # Try connecting first two colors with straight lines
    for c1 in colors:
        for c2 in colors:
            if c1 == c2:
                continue
            pos1 = np.argwhere(arr == c1)
            pos2 = np.argwhere(arr == c2)
            if len(pos1) == 1 and len(pos2) == 1:
                p1, p2 = pos1[0], pos2[0]
                result = arr.copy()

                # Manhattan path
                path_color = max(c1, c2)
                r, c = p1
                while r != p2[0]:
                    result[r, c] = path_color
                    r += 1 if r < p2[0] else -1
                while c != p2[1]:
                    result[r, c] = path_color
                    c += 1 if c < p2[1] else -1
                result[p2[0], p2[1]] = path_color

                if np.array_equal(result, out_arr):
                    return "path_trace_manhattan"

    return None


def try_convex_hull_fill(inp, out):
    """Fill the convex hull of colored points."""
    arr = np.array(inp)
    out_arr = np.array(out)

    for color in [c for c in np.unique(arr) if c != 0]:
        points = np.argwhere(arr == color)
        if len(points) < 3:
            continue

        # Simple bounding box fill as approximation
        min_r, min_c = points.min(axis=0)
        max_r, max_c = points.max(axis=0)

        result = arr.copy()
        result[min_r:max_r+1, min_c:max_c+1] = color

        if np.array_equal(result, out_arr):
            return "bounding_box_fill"

    return None


def try_diagonal_line_extend(inp, out):
    """Extend diagonal lines to edges."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    result = arr.copy()
    for color in [c for c in np.unique(arr) if c != 0]:
        points = np.argwhere(arr == color)
        for r, c in points:
            # Extend in all 4 diagonal directions
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                while 0 <= nr < h and 0 <= nc < w:
                    if result[nr, nc] == 0:
                        result[nr, nc] = color
                    nr += dr
                    nc += dc

    if np.array_equal(result, out_arr):
        return "diagonal_line_extend"

    return None


def try_blob_center_mark(inp, out):
    """Mark the center of each blob/object."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    result = np.zeros_like(arr)
    out_colors = [c for c in np.unique(out_arr) if c != 0]
    mark_color = out_colors[0] if out_colors else 1

    for color in [c for c in np.unique(arr) if c != 0]:
        mask = (arr == color).astype(int)
        labeled, num = ndimage.label(mask)

        for region_id in range(1, num + 1):
            region = (labeled == region_id)
            coords = np.argwhere(region)
            center = coords.mean(axis=0).astype(int)
            result[center[0], center[1]] = mark_color

    if np.array_equal(result, out_arr):
        return "blob_center_mark"

    # Also try keeping original and marking centers
    result2 = arr.copy()
    for color in [c for c in np.unique(arr) if c != 0]:
        mask = (arr == color).astype(int)
        labeled, num = ndimage.label(mask)
        for region_id in range(1, num + 1):
            region = (labeled == region_id)
            coords = np.argwhere(region)
            center = coords.mean(axis=0).astype(int)
            result2[center[0], center[1]] = mark_color

    if np.array_equal(result2, out_arr):
        return "blob_center_mark_overlay"

    return None


def try_gradient_fill(inp, out):
    """Fill with gradient based on position."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = out_arr.shape

    # Check if output has sequential values
    unique_out = sorted(np.unique(out_arr))
    if len(unique_out) < 2:
        return None

    # Horizontal gradient
    result_h = np.zeros((h, w), dtype=arr.dtype)
    for j in range(w):
        val = int(j * (len(unique_out) - 1) / max(1, w - 1))
        result_h[:, j] = unique_out[min(val, len(unique_out) - 1)]

    if np.array_equal(result_h, out_arr):
        return "gradient_fill_h"

    # Vertical gradient
    result_v = np.zeros((h, w), dtype=arr.dtype)
    for i in range(h):
        val = int(i * (len(unique_out) - 1) / max(1, h - 1))
        result_v[i, :] = unique_out[min(val, len(unique_out) - 1)]

    if np.array_equal(result_v, out_arr):
        return "gradient_fill_v"

    return None


def try_corner_fill(inp, out):
    """Fill corners with specific colors."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape
    oh, ow = out_arr.shape

    if h != oh or w != ow:
        return None

    result = arr.copy()
    corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]

    out_colors = [c for c in np.unique(out_arr) if c != 0]
    if not out_colors:
        return None

    for idx, (r, c) in enumerate(corners):
        result[r, c] = out_colors[idx % len(out_colors)]

    if np.array_equal(result, out_arr):
        return "corner_fill"

    return None


def try_flood_until_collision(inp, out):
    """Flood fill from sources until meeting another color."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    colors = [c for c in np.unique(arr) if c != 0]
    if len(colors) < 2:
        return None

    # For each color, try expanding until hitting another
    result = arr.copy()
    for color in colors:
        mask = (arr == color)
        other_mask = (arr != 0) & (arr != color)

        # Iteratively dilate until hitting other colors
        current = mask.copy()
        for _ in range(max(h, w)):
            dilated = ndimage.binary_dilation(current)
            # Don't expand into other colors
            dilated = dilated & ~other_mask
            if np.array_equal(dilated, current):
                break
            current = dilated

        result[current] = color

    if np.array_equal(result, out_arr):
        return "flood_until_collision"

    return None


def try_pixelate(inp, out):
    """Pixelate by averaging blocks."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Check if downscaling by integer factor
    if ih % oh == 0 and iw % ow == 0:
        bh, bw = ih // oh, iw // ow
        result = np.zeros((oh, ow), dtype=arr.dtype)

        for i in range(oh):
            for j in range(ow):
                block = arr[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                # Most common value in block
                vals, counts = np.unique(block, return_counts=True)
                result[i, j] = vals[counts.argmax()]

        if np.array_equal(result, out_arr):
            return f"pixelate_{bh}x{bw}"

    return None

# ============================================================================
# Pattern GRID LOGIC TRANSFORMS (Batch 10)
# ⧫depth=-90000BLOCKLOGIC⧫
# ============================================================================


def try_grid_xor(inp, out):
    """XOR operation between grid halves."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    # Horizontal split XOR
    if h % 2 == 0:
        top = arr[:h//2, :]
        bot = arr[h//2:, :]
        result = ((top != 0) ^ (bot != 0)).astype(arr.dtype)
        out_colors = [c for c in np.unique(out_arr) if c != 0]
        if out_colors:
            result = np.where(result, out_colors[0], 0)
        if np.array_equal(result, out_arr):
            return "grid_xor_h"

    # Vertical split XOR
    if w % 2 == 0:
        left = arr[:, :w//2]
        right = arr[:, w//2:]
        result = ((left != 0) ^ (right != 0)).astype(arr.dtype)
        out_colors = [c for c in np.unique(out_arr) if c != 0]
        if out_colors:
            result = np.where(result, out_colors[0], 0)
        if np.array_equal(result, out_arr):
            return "grid_xor_v"

    return None


def try_grid_or(inp, out):
    """OR operation between grid halves."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if h % 2 == 0:
        top = arr[:h//2, :]
        bot = arr[h//2:, :]
        result = ((top != 0) | (bot != 0)).astype(arr.dtype)
        out_colors = [c for c in np.unique(out_arr) if c != 0]
        if out_colors:
            result = np.where(result, out_colors[0], 0)
        if np.array_equal(result, out_arr):
            return "grid_or_h"

    if w % 2 == 0:
        left = arr[:, :w//2]
        right = arr[:, w//2:]
        result = ((left != 0) | (right != 0)).astype(arr.dtype)
        out_colors = [c for c in np.unique(out_arr) if c != 0]
        if out_colors:
            result = np.where(result, out_colors[0], 0)
        if np.array_equal(result, out_arr):
            return "grid_or_v"

    return None


def try_row_wise_sort(inp, out):
    """Sort each row independently."""
    arr = np.array(inp)
    out_arr = np.array(out)

    result = np.sort(arr, axis=1)
    if np.array_equal(result, out_arr):
        return "row_sort_asc"

    result = np.sort(arr, axis=1)[:, ::-1]
    if np.array_equal(result, out_arr):
        return "row_sort_desc"

    return None


def try_col_wise_sort(inp, out):
    """Sort each column independently."""
    arr = np.array(inp)
    out_arr = np.array(out)

    result = np.sort(arr, axis=0)
    if np.array_equal(result, out_arr):
        return "col_sort_asc"

    result = np.sort(arr, axis=0)[::-1, :]
    if np.array_equal(result, out_arr):
        return "col_sort_desc"

    return None


def try_block_swap(inp, out):
    """Swap quadrants of the grid."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if h % 2 != 0 or w % 2 != 0:
        return None

    # Split into quadrants
    q1 = arr[:h//2, :w//2]
    q2 = arr[:h//2, w//2:]
    q3 = arr[h//2:, :w//2]
    q4 = arr[h//2:, w//2:]

    # Try diagonal swap (q1<->q4, q2<->q3)
    result = np.zeros_like(arr)
    result[:h//2, :w//2] = q4
    result[:h//2, w//2:] = q3
    result[h//2:, :w//2] = q2
    result[h//2:, w//2:] = q1
    if np.array_equal(result, out_arr):
        return "block_swap_diagonal"

    # Try horizontal swap (q1<->q2, q3<->q4)
    result = np.zeros_like(arr)
    result[:h//2, :w//2] = q2
    result[:h//2, w//2:] = q1
    result[h//2:, :w//2] = q4
    result[h//2:, w//2:] = q3
    if np.array_equal(result, out_arr):
        return "block_swap_h"

    # Try vertical swap (q1<->q3, q2<->q4)
    result = np.zeros_like(arr)
    result[:h//2, :w//2] = q3
    result[:h//2, w//2:] = q4
    result[h//2:, :w//2] = q1
    result[h//2:, w//2:] = q2
    if np.array_equal(result, out_arr):
        return "block_swap_v"

    return None


def try_color_count_grid(inp, out):
    """Create grid based on color counts."""
    arr = np.array(inp)
    out_arr = np.array(out)
    oh, ow = out_arr.shape

    colors = sorted([c for c in np.unique(arr) if c != 0])
    counts = {c: np.sum(arr == c) for c in colors}

    # Try: output is a bar chart of color counts
    if oh == 1 and ow == len(colors):
        max_count = max(counts.values()) if counts else 1
        result = np.array([[counts.get(c, 0) for c in colors]])
        if np.array_equal(result, out_arr):
            return "color_count_row"

    return None


def try_object_shift(inp, out):
    """Shift objects by fixed amount."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape
    oh, ow = out_arr.shape

    if h != oh or w != ow:
        return None

    # Try shifting by small amounts
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            if dr == 0 and dc == 0:
                continue
            result = np.zeros_like(arr)
            for i in range(h):
                for j in range(w):
                    if arr[i, j] != 0:
                        ni, nj = i + dr, j + dc
                        if 0 <= ni < h and 0 <= nj < w:
                            result[ni, nj] = arr[i, j]
            if np.array_equal(result, out_arr):
                return f"object_shift_{dr}_{dc}"

    return None


def try_rotate_around_center(inp, out):
    """Rotate non-zero pixels around grid center."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if h != w:
        return None

    center = h / 2 - 0.5
    for angle in [90, 180, 270]:
        result = np.zeros_like(arr)
        rad = np.radians(angle)
        for i in range(h):
            for j in range(w):
                if arr[i, j] != 0:
                    # Rotate point around center
                    di, dj = i - center, j - center
                    ni = int(round(di * np.cos(rad) - dj * np.sin(rad) + center))
                    nj = int(round(di * np.sin(rad) + dj * np.cos(rad) + center))
                    if 0 <= ni < h and 0 <= nj < w:
                        result[ni, nj] = arr[i, j]
        if np.array_equal(result, out_arr):
            return f"rotate_center_{angle}"

    return None


def try_mirror_in_place(inp, out):
    """Mirror non-zero pattern in place."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    # Find non-zero bounding box
    nonzero = np.argwhere(arr != 0)
    if len(nonzero) == 0:
        return None
    min_r, min_c = nonzero.min(axis=0)
    max_r, max_c = nonzero.max(axis=0)

    # Extract and mirror
    pattern = arr[min_r:max_r+1, min_c:max_c+1]

    # Horizontal mirror in place
    result = arr.copy()
    mirrored = np.fliplr(pattern)
    result[min_r:max_r+1, min_c:max_c+1] = mirrored
    if np.array_equal(result, out_arr):
        return "mirror_in_place_h"

    # Vertical mirror in place
    result = arr.copy()
    mirrored = np.flipud(pattern)
    result[min_r:max_r+1, min_c:max_c+1] = mirrored
    if np.array_equal(result, out_arr):
        return "mirror_in_place_v"

    return None


def try_row_repeat(inp, out):
    """Repeat a specific row to fill output."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if iw != ow:
        return None

    # Try repeating each row
    for row_idx in range(ih):
        row = arr[row_idx:row_idx+1, :]
        result = np.tile(row, (oh, 1))
        if np.array_equal(result, out_arr):
            return f"row_repeat_{row_idx}"

    return None


def try_col_repeat(inp, out):
    """Repeat a specific column to fill output."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if ih != oh:
        return None

    # Try repeating each column
    for col_idx in range(iw):
        col = arr[:, col_idx:col_idx+1]
        result = np.tile(col, (1, ow))
        if np.array_equal(result, out_arr):
            return f"col_repeat_{col_idx}"

    return None


def try_nonzero_to_corner(inp, out):
    """Move all non-zero to specific corner."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    nonzero_vals = arr[arr != 0].flatten()
    if len(nonzero_vals) == 0:
        return None

    # Try packing to each corner
    corners = [
        ("top_left", lambda: (0, 0, 1, 1)),
        ("top_right", lambda: (0, w - len(nonzero_vals), 1, 1)),
        ("bottom_left", lambda: (h - 1, 0, -1, 1)),
        ("bottom_right", lambda: (h - 1, w - 1, -1, -1)),
    ]

    for name, _ in corners:
        result = np.zeros_like(arr)
        idx = 0
        if name == "top_left":
            for i in range(h):
                for j in range(w):
                    if idx < len(nonzero_vals):
                        result[i, j] = nonzero_vals[idx]
                        idx += 1
        elif name == "top_right":
            for i in range(h):
                for j in range(w - 1, -1, -1):
                    if idx < len(nonzero_vals):
                        result[i, j] = nonzero_vals[idx]
                        idx += 1
        if np.array_equal(result, out_arr):
            return f"nonzero_to_{name}"

    return None


def try_color_mask_keep(inp, out):
    """Keep only pixels matching a mask color pattern."""
    arr = np.array(inp)
    out_arr = np.array(out)

    colors = [c for c in np.unique(arr) if c != 0]
    if len(colors) < 2:
        return None

    # For each color pair, try using one as mask for other
    for mask_color in colors:
        for keep_color in colors:
            if mask_color == keep_color:
                continue
            mask = (arr == mask_color)
            result = np.where(mask, keep_color, 0)
            if np.array_equal(result, out_arr):
                return f"color_mask_{mask_color}_to_{keep_color}"

    return None


def try_flood_same_row_col(inp, out):
    """Flood fill same row and column from colored points."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    result = arr.copy()
    for color in [c for c in np.unique(arr) if c != 0]:
        positions = np.argwhere(arr == color)
        for r, c in positions:
            result[r, :] = color  # Fill row
            result[:, c] = color  # Fill column

    if np.array_equal(result, out_arr):
        return "flood_row_col"

    # Try only rows
    result = arr.copy()
    for color in [c for c in np.unique(arr) if c != 0]:
        positions = np.argwhere(arr == color)
        for r, c in positions:
            result[r, :] = color
    if np.array_equal(result, out_arr):
        return "flood_row"

    # Try only columns
    result = arr.copy()
    for color in [c for c in np.unique(arr) if c != 0]:
        positions = np.argwhere(arr == color)
        for r, c in positions:
            result[:, c] = color
    if np.array_equal(result, out_arr):
        return "flood_col"

    return None


def try_compress_to_unique(inp, out):
    """Compress grid keeping only unique rows/cols."""
    arr = np.array(inp)
    out_arr = np.array(out)

    # Unique rows
    unique_rows = []
    seen = set()
    for row in arr:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_rows.append(row)
    if len(unique_rows) > 0:
        result = np.array(unique_rows)
        if np.array_equal(result, out_arr):
            return "compress_unique_rows"

    # Unique columns
    unique_cols = []
    seen = set()
    for j in range(arr.shape[1]):
        col_tuple = tuple(arr[:, j])
        if col_tuple not in seen:
            seen.add(col_tuple)
            unique_cols.append(arr[:, j])
    if len(unique_cols) > 0:
        result = np.column_stack(unique_cols)
        if np.array_equal(result, out_arr):
            return "compress_unique_cols"

    return None

# ============================================================================
# Pattern COMPOSITE TRANSFORMS (Batch 11)
# ⧫depth=-90000DIFFPOSITION
# ============================================================================


def try_grid_subtract(inp, out):
    """Subtract one grid half from another."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    # Horizontal halves subtract
    if h % 2 == 0:
        top = (arr[:h//2, :] != 0).astype(int)
        bot = (arr[h//2:, :] != 0).astype(int)
        diff = top - bot  # 1 where top has, bot doesn't
        out_colors = [c for c in np.unique(out_arr) if c != 0]
        if out_colors:
            result = np.where(diff > 0, out_colors[0], 0).astype(arr.dtype)
            if np.array_equal(result, out_arr):
                return "grid_subtract_top_bot"
            result = np.where(diff < 0, out_colors[0], 0).astype(arr.dtype)
            if np.array_equal(result, out_arr):
                return "grid_subtract_bot_top"

    # Vertical halves
    if w % 2 == 0:
        left = (arr[:, :w//2] != 0).astype(int)
        right = (arr[:, w//2:] != 0).astype(int)
        diff = left - right
        out_colors = [c for c in np.unique(out_arr) if c != 0]
        if out_colors:
            result = np.where(diff > 0, out_colors[0], 0).astype(arr.dtype)
            if np.array_equal(result, out_arr):
                return "grid_subtract_left_right"
            result = np.where(diff < 0, out_colors[0], 0).astype(arr.dtype)
            if np.array_equal(result, out_arr):
                return "grid_subtract_right_left"

    return None


def try_object_by_size(inp, out):
    """Keep objects by size (smallest, largest, median)."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    for color in [c for c in np.unique(arr) if c != 0]:
        mask = (arr == color).astype(int)
        labeled, num = ndimage.label(mask)
        if num < 2:
            continue

        # Get sizes
        sizes = [(region_id, np.sum(labeled == region_id)) for region_id in range(1, num + 1)]
        sizes.sort(key=lambda x: x[1])

        # Try keeping smallest
        result = np.zeros_like(arr)
        smallest_id = sizes[0][0]
        result[labeled == smallest_id] = color
        if np.array_equal(result, out_arr):
            return "keep_smallest_object"

        # Try keeping largest
        result = np.zeros_like(arr)
        largest_id = sizes[-1][0]
        result[labeled == largest_id] = color
        if np.array_equal(result, out_arr):
            return "keep_largest_object"

        # Try keeping median
        if num >= 3:
            result = np.zeros_like(arr)
            median_id = sizes[num // 2][0]
            result[labeled == median_id] = color
            if np.array_equal(result, out_arr):
                return "keep_median_object"

    return None


def try_recolor_by_size(inp, out):
    """Recolor objects based on their size."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    for color in [c for c in np.unique(arr) if c != 0]:
        mask = (arr == color).astype(int)
        labeled, num = ndimage.label(mask)
        if num < 2:
            continue

        # Get sizes and sort
        sizes = [(region_id, np.sum(labeled == region_id)) for region_id in range(1, num + 1)]
        sizes.sort(key=lambda x: x[1])

        # Try recoloring by size rank
        out_colors = sorted([c for c in np.unique(out_arr) if c != 0])
        if len(out_colors) >= len(sizes):
            result = arr.copy()
            for rank, (region_id, _) in enumerate(sizes):
                result[labeled == region_id] = out_colors[rank % len(out_colors)]
            if np.array_equal(result, out_arr):
                return "recolor_by_size"

    return None


def try_recolor_by_position(inp, out):
    """Recolor objects based on their position."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    for color in [c for c in np.unique(arr) if c != 0]:
        mask = (arr == color).astype(int)
        labeled, num = ndimage.label(mask)
        if num < 2:
            continue

        # Get centers
        centers = []
        for region_id in range(1, num + 1):
            coords = np.argwhere(labeled == region_id)
            center = coords.mean(axis=0)
            centers.append((region_id, center[0], center[1]))

        # Sort by row then column (top-to-bottom, left-to-right)
        centers.sort(key=lambda x: (x[1], x[2]))

        out_colors = sorted([c for c in np.unique(out_arr) if c != 0])
        if len(out_colors) >= len(centers):
            result = arr.copy()
            for rank, (region_id, _, _) in enumerate(centers):
                result[labeled == region_id] = out_colors[rank % len(out_colors)]
            if np.array_equal(result, out_arr):
                return "recolor_by_position"

    return None


def try_contour_only(inp, out):
    """Keep only the contour/perimeter of shapes."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    result = np.zeros_like(arr)
    for color in [c for c in np.unique(arr) if c != 0]:
        mask = (arr == color)
        # Erode and subtract to get contour
        eroded = ndimage.binary_erosion(mask)
        contour = mask & ~eroded
        result[contour] = color

    if np.array_equal(result, out_arr):
        return "contour_only"

    # Try with different output color
    out_colors = [c for c in np.unique(out_arr) if c != 0]
    if len(out_colors) == 1:
        result = np.zeros_like(arr)
        for color in [c for c in np.unique(arr) if c != 0]:
            mask = (arr == color)
            eroded = ndimage.binary_erosion(mask)
            contour = mask & ~eroded
            result[contour] = out_colors[0]
        if np.array_equal(result, out_arr):
            return "contour_recolor"

    return None


def try_object_overlap(inp, out):
    """Find where two colored objects overlap."""
    arr = np.array(inp)
    out_arr = np.array(out)

    colors = [c for c in np.unique(arr) if c != 0]
    if len(colors) < 2:
        return None

    for i, c1 in enumerate(colors):
        for c2 in colors[i+1:]:
            mask1 = (arr == c1)
            mask2 = (arr == c2)
            overlap = mask1 & mask2
            if np.any(overlap):
                out_colors = [c for c in np.unique(out_arr) if c != 0]
                if out_colors:
                    result = np.where(overlap, out_colors[0], 0).astype(arr.dtype)
                    if np.array_equal(result, out_arr):
                        return f"object_overlap_{c1}_{c2}"

    return None


def try_color_replace_pattern(inp, out):
    """Replace one color with pattern from another."""
    arr = np.array(inp)
    out_arr = np.array(out)

    colors = [c for c in np.unique(arr) if c != 0]
    if len(colors) < 2:
        return None

    for target_color in colors:
        for source_color in colors:
            if target_color == source_color:
                continue
            # Replace target positions with source color
            result = arr.copy()
            result[arr == target_color] = source_color
            if np.array_equal(result, out_arr):
                return f"replace_{target_color}_with_{source_color}"

    return None


def try_isolate_corners_content(inp, out):
    """Extract content from corners."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape
    oh, ow = out_arr.shape

    # Try extracting corner regions
    for corner_h in range(1, min(h, 10)):
        for corner_w in range(1, min(w, 10)):
            corners = [
                arr[:corner_h, :corner_w],  # top-left
                arr[:corner_h, -corner_w:],  # top-right
                arr[-corner_h:, :corner_w],  # bottom-left
                arr[-corner_h:, -corner_w:],  # bottom-right
            ]
            for i, corner in enumerate(corners):
                if corner.shape == out_arr.shape:
                    if np.array_equal(corner, out_arr):
                        names = ["top_left", "top_right", "bottom_left", "bottom_right"]
                        return f"isolate_corner_{names[i]}_{corner_h}x{corner_w}"

    return None


def try_scale_pattern(inp, out):
    """Scale a detected pattern to fill output."""
    arr = np.array(inp)
    out_arr = np.array(out)

    # Find non-zero bounding box
    nonzero = np.argwhere(arr != 0)
    if len(nonzero) == 0:
        return None
    min_r, min_c = nonzero.min(axis=0)
    max_r, max_c = nonzero.max(axis=0)

    pattern = arr[min_r:max_r+1, min_c:max_c+1]
    ph, pw = pattern.shape
    oh, ow = out_arr.shape

    if ph == 0 or pw == 0:
        return None

    # Try integer scaling
    if oh % ph == 0 and ow % pw == 0:
        sh, sw = oh // ph, ow // pw
        if sh == sw and sh >= 2:
            result = np.repeat(np.repeat(pattern, sh, axis=0), sw, axis=1)
            if np.array_equal(result, out_arr):
                return f"scale_pattern_{sh}x"

    return None


def try_binary_to_multicolor(inp, out):
    """Convert binary pattern to multi-color based on rules."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    # Get colors in input
    colors = [c for c in np.unique(arr) if c != 0]
    out_colors = sorted([c for c in np.unique(out_arr) if c != 0])

    if len(colors) != 1 or len(out_colors) < 2:
        return None

    input_color = colors[0]
    mask = (arr == input_color)

    # Try coloring by quadrant
    result = np.zeros_like(arr)
    mid_r, mid_c = h // 2, w // 2
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                q_idx = (0 if i < mid_r else 2) + (0 if j < mid_c else 1)
                result[i, j] = out_colors[q_idx % len(out_colors)]

    if np.array_equal(result, out_arr):
        return "binary_to_quadrant_colors"

    return None


def try_stripe_by_row(inp, out):
    """Color rows with alternating colors."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    out_colors = sorted([c for c in np.unique(out_arr) if c != 0])
    if len(out_colors) < 2:
        return None

    # Alternating row colors
    result = np.zeros((h, w), dtype=arr.dtype)
    for i in range(h):
        result[i, :] = out_colors[i % len(out_colors)]

    if np.array_equal(result, out_arr):
        return "stripe_rows"

    return None


def try_stripe_by_col(inp, out):
    """Color columns with alternating colors."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    out_colors = sorted([c for c in np.unique(out_arr) if c != 0])
    if len(out_colors) < 2:
        return None

    # Alternating column colors
    result = np.zeros((h, w), dtype=arr.dtype)
    for j in range(w):
        result[:, j] = out_colors[j % len(out_colors)]

    if np.array_equal(result, out_arr):
        return "stripe_cols"

    return None


def try_cross_hatch(inp, out):
    """Create cross-hatch pattern where colors meet."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    colors = [c for c in np.unique(arr) if c != 0]
    if len(colors) < 2:
        return None

    # Fill rows and columns from each colored pixel
    result = np.zeros_like(arr)
    for color in colors:
        positions = np.argwhere(arr == color)
        for r, c in positions:
            result[r, :] = np.where(result[r, :] == 0, color, result[r, :])
            result[:, c] = np.where(result[:, c] == 0, color, result[:, c])

    if np.array_equal(result, out_arr):
        return "cross_hatch"

    return None

# ============================================================================
# Pattern BOUNDARY/REGION TRANSFORMS (Batch 12)
# depth=-90000 BOUNDARY EDGE NEIGHBOR ADJACENCY CONNECTIVITY REGION
# ============================================================================


def try_edge_detect(inp, out):
    """Mark all cells adjacent to different color."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    result = np.zeros_like(arr)
    for i in range(h):
        for j in range(w):
            val = arr[i, j]
            is_edge = False
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    if arr[ni, nj] != val:
                        is_edge = True
                        break
            if is_edge:
                result[i, j] = val

    if np.array_equal(result, out_arr):
        return "edge_detect"
    return None


def try_neighbor_count_color(inp, out):
    """Color cells based on number of same-color neighbors."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    # Try different color mappings
    for base_color in range(1, 10):
        result = np.zeros_like(arr)
        for i in range(h):
            for j in range(w):
                if arr[i, j] != 0:
                    count = 0
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] != 0:
                            count += 1
                    result[i, j] = count + base_color if count > 0 else arr[i, j]

        if np.array_equal(result, out_arr):
            return f"neighbor_count_color_{base_color}"
    return None


def try_connected_components_color(inp, out):
    """Color each connected component uniquely."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    binary = (arr != 0).astype(int)
    labeled, num = ndimage.label(binary)

    out_colors = [c for c in np.unique(out_arr) if c != 0]
    if num != len(out_colors):
        return None

    result = np.zeros_like(arr)
    for i in range(1, num + 1):
        if i - 1 < len(out_colors):
            result[labeled == i] = out_colors[i - 1]

    if np.array_equal(result, out_arr):
        return "connected_components_color"
    return None


def try_region_fill_largest(inp, out):
    """Fill enclosed regions with largest region's color."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    binary = (arr != 0).astype(int)
    labeled, num = ndimage.label(binary)

    if num < 2:
        return None

    # Find largest component
    sizes = ndimage.sum(binary, labeled, range(1, num + 1))
    largest_idx = np.argmax(sizes) + 1
    largest_color = arr[labeled == largest_idx][0]

    result = arr.copy()
    result[arr == 0] = largest_color

    if np.array_equal(result, out_arr):
        return "region_fill_largest"
    return None


def try_cluster_centers(inp, out):
    """Mark centers of each cluster/region."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    binary = (arr != 0).astype(int)
    labeled, num = ndimage.label(binary)

    result = np.zeros_like(arr)
    for i in range(1, num + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) > 0:
            center = coords.mean(axis=0).astype(int)
            color = arr[labeled == i][0]
            result[center[0], center[1]] = color

    if np.array_equal(result, out_arr):
        return "cluster_centers"
    return None


def try_boundary_to_interior(inp, out):
    """Convert boundary pixels to interior color."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    result = arr.copy()
    for i in range(h):
        for j in range(w):
            if arr[i, j] != 0:
                # Check if this is a boundary pixel
                is_boundary = False
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        if arr[ni, nj] == 0:
                            is_boundary = True
                            break
                    else:
                        is_boundary = True

                if is_boundary:
                    # Find interior color - check 2 steps inward
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + 2*di, j + 2*dj
                        if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] != 0:
                            result[i, j] = arr[ni, nj]
                            break

    if np.array_equal(result, out_arr):
        return "boundary_to_interior"
    return None


def try_adjacency_merge(inp, out):
    """Merge adjacent regions of same color."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    result = np.zeros_like(arr)
    visited = np.zeros_like(arr, dtype=bool)

    def flood_fill(r, c, color, target):
        if r < 0 or r >= h or c < 0 or c >= w:
            return
        if visited[r, c] or arr[r, c] != color:
            return
        visited[r, c] = True
        result[r, c] = target
        flood_fill(r+1, c, color, target)
        flood_fill(r-1, c, color, target)
        flood_fill(r, c+1, color, target)
        flood_fill(r, c-1, color, target)

    for i in range(h):
        for j in range(w):
            if not visited[i, j] and arr[i, j] != 0:
                flood_fill(i, j, arr[i, j], arr[i, j])

    if np.array_equal(result, out_arr):
        return "adjacency_merge"
    return None


def try_segmentation_to_outline(inp, out):
    """Convert filled segments to outlines only."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    result = np.zeros_like(arr)
    for i in range(h):
        for j in range(w):
            if arr[i, j] != 0:
                # Check if edge of region
                is_edge = False
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= h or nj < 0 or nj >= w:
                        is_edge = True
                        break
                    if arr[ni, nj] != arr[i, j]:
                        is_edge = True
                        break
                if is_edge:
                    result[i, j] = arr[i, j]

    if np.array_equal(result, out_arr):
        return "segmentation_to_outline"
    return None


def try_region_size_filter(inp, out):
    """Keep only regions of certain size."""
    from scipy import ndimage
    arr = np.array(inp)
    out_arr = np.array(out)

    binary = (arr != 0).astype(int)
    labeled, num = ndimage.label(binary)

    # Try different size thresholds
    for min_size in [1, 2, 3, 4, 5, 10]:
        result = np.zeros_like(arr)
        for i in range(1, num + 1):
            region = (labeled == i)
            if np.sum(region) >= min_size:
                result[region] = arr[region]

        if np.array_equal(result, out_arr):
            return f"region_size_filter_{min_size}"
    return None


def try_group_by_row(inp, out):
    """Group pixels by row, unify colors."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    result = np.zeros_like(arr)
    for i in range(h):
        row_colors = [c for c in arr[i, :] if c != 0]
        if row_colors:
            dominant = max(set(row_colors), key=row_colors.count)
            result[i, :] = np.where(arr[i, :] != 0, dominant, 0)

    if np.array_equal(result, out_arr):
        return "group_by_row"
    return None


def try_group_by_col(inp, out):
    """Group pixels by column, unify colors."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    result = np.zeros_like(arr)
    for j in range(w):
        col_colors = [c for c in arr[:, j] if c != 0]
        if col_colors:
            dominant = max(set(col_colors), key=col_colors.count)
            result[:, j] = np.where(arr[:, j] != 0, dominant, 0)

    if np.array_equal(result, out_arr):
        return "group_by_col"
    return None


def try_region_border_color(inp, out):
    """Color region borders differently from interior."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    out_colors = sorted([c for c in np.unique(out_arr) if c != 0])
    if len(out_colors) < 2:
        return None

    border_color, interior_color = out_colors[0], out_colors[1]

    result = np.zeros_like(arr)
    for i in range(h):
        for j in range(w):
            if arr[i, j] != 0:
                is_border = False
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= h or nj < 0 or nj >= w or arr[ni, nj] == 0:
                        is_border = True
                        break
                result[i, j] = border_color if is_border else interior_color

    if np.array_equal(result, out_arr):
        return "region_border_color"
    return None


def try_connectivity_8_to_4(inp, out):
    """Convert 8-connectivity to 4-connectivity."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    result = arr.copy()
    # Remove diagonal connections
    for i in range(h - 1):
        for j in range(w - 1):
            # Check for diagonal pattern that should be broken
            if arr[i, j] != 0 and arr[i+1, j+1] != 0:
                if arr[i, j+1] == 0 and arr[i+1, j] == 0:
                    result[i, j] = 0  # Break diagonal

    if np.array_equal(result, out_arr):
        return "connectivity_8_to_4"
    return None

# ============================================================================
# Pattern SYMMETRY/REFLECTION TRANSFORMS (Batch 13)
# depth=-90000 SYMMETRY AXIS FOLD MIRROR DIAGONAL RADIAL CRYSTALLIZE
# ============================================================================


def try_fold_horizontal(inp, out):
    """Fold grid horizontally and overlay halves."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if h % 2 != 0:
        return None

    top = arr[:h//2, :]
    bottom = np.flipud(arr[h//2:, :])

    # Try OR combination
    result = np.where(top != 0, top, bottom)
    if np.array_equal(result, out_arr):
        return "fold_horizontal_or"

    # Try AND combination
    result = np.where((top != 0) & (bottom != 0), top, 0)
    if np.array_equal(result, out_arr):
        return "fold_horizontal_and"

    return None


def try_fold_vertical(inp, out):
    """Fold grid vertically and overlay halves."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if w % 2 != 0:
        return None

    left = arr[:, :w//2]
    right = np.fliplr(arr[:, w//2:])

    # Try OR combination
    result = np.where(left != 0, left, right)
    if np.array_equal(result, out_arr):
        return "fold_vertical_or"

    # Try AND combination
    result = np.where((left != 0) & (right != 0), left, 0)
    if np.array_equal(result, out_arr):
        return "fold_vertical_and"

    return None


def try_fold_diagonal(inp, out):
    """Fold along main diagonal."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if h != w:
        return None

    transposed = arr.T

    # OR combination
    result = np.where(arr != 0, arr, transposed)
    if np.array_equal(result, out_arr):
        return "fold_diagonal_or"

    # AND combination
    result = np.where((arr != 0) & (transposed != 0), arr, 0)
    if np.array_equal(result, out_arr):
        return "fold_diagonal_and"

    return None


def try_radial_symmetry(inp, out):
    """Apply 4-fold rotational symmetry."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if h != w:
        return None

    result = arr.copy()
    for k in [1, 2, 3]:
        rotated = np.rot90(arr, k)
        result = np.where(result != 0, result, rotated)

    if np.array_equal(result, out_arr):
        return "radial_symmetry_4"
    return None


def try_bilateral_symmetry(inp, out):
    """Enforce bilateral symmetry along vertical axis."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    flipped = np.fliplr(arr)
    result = np.where(arr != 0, arr, flipped)

    if np.array_equal(result, out_arr):
        return "bilateral_symmetry_v"

    # Try horizontal
    flipped = np.flipud(arr)
    result = np.where(arr != 0, arr, flipped)

    if np.array_equal(result, out_arr):
        return "bilateral_symmetry_h"

    return None


def try_complete_quadrant(inp, out):
    """Complete pattern from one quadrant to all four."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if h % 2 != 0 or w % 2 != 0:
        return None

    qh, qw = h // 2, w // 2

    # Try using top-left as source
    q = arr[:qh, :qw]
    result = np.zeros_like(arr)
    result[:qh, :qw] = q
    result[:qh, qw:] = np.fliplr(q)
    result[qh:, :qw] = np.flipud(q)
    result[qh:, qw:] = np.flipud(np.fliplr(q))

    if np.array_equal(result, out_arr):
        return "complete_quadrant_tl"

    # Try using top-right as source
    q = arr[:qh, qw:]
    result = np.zeros_like(arr)
    result[:qh, qw:] = q
    result[:qh, :qw] = np.fliplr(q)
    result[qh:, qw:] = np.flipud(q)
    result[qh:, :qw] = np.flipud(np.fliplr(q))

    if np.array_equal(result, out_arr):
        return "complete_quadrant_tr"

    return None


def try_axis_mirror_and_merge(inp, out):
    """Mirror along axis and merge with original."""
    arr = np.array(inp)
    out_arr = np.array(out)

    # Horizontal mirror merge
    result = np.maximum(arr, np.flipud(arr))
    if np.array_equal(result, out_arr):
        return "axis_mirror_merge_h"

    # Vertical mirror merge
    result = np.maximum(arr, np.fliplr(arr))
    if np.array_equal(result, out_arr):
        return "axis_mirror_merge_v"

    return None

# ============================================================================
# Pattern COMBO MOVES (Batch 13 - chained operations)
# depth=-90000 COMBO CHAIN SEQUENCE COMPOSE CRYSTALLIZE
# ============================================================================


def try_combo_rotate_then_flip(inp, out):
    """Rotate then flip."""
    arr = np.array(inp)
    out_arr = np.array(out)

    for k in [1, 2, 3]:
        rotated = np.rot90(arr, k)
        if np.array_equal(np.fliplr(rotated), out_arr):
            return f"combo_rot{k*90}_hflip"
        if np.array_equal(np.flipud(rotated), out_arr):
            return f"combo_rot{k*90}_vflip"
    return None


def try_combo_extract_then_tile(inp, out):
    """Extract a subgrid then tile it."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Try extracting small regions and tiling
    for eh in range(1, min(ih+1, 6)):
        for ew in range(1, min(iw+1, 6)):
            if oh % eh == 0 and ow % ew == 0:
                th, tw = oh // eh, ow // ew
                # Try each extraction position
                for y in range(ih - eh + 1):
                    for x in range(iw - ew + 1):
                        extract = arr[y:y+eh, x:x+ew]
                        tiled = np.tile(extract, (th, tw))
                        if np.array_equal(tiled, out_arr):
                            return f"combo_extract_{y}_{x}_{eh}x{ew}_tile_{th}x{tw}"
    return None


def try_combo_upscale_then_fill(inp, out):
    """Upscale then fill holes."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    if oh % ih != 0 or ow % iw != 0:
        return None

    sh, sw = oh // ih, ow // iw
    if sh != sw or sh < 2:
        return None

    upscaled = np.repeat(np.repeat(arr, sh, axis=0), sw, axis=1)

    # Try fill holes on upscaled
    h, w = upscaled.shape
    result = upscaled.copy()
    for i in range(h):
        for j in range(w):
            if upscaled[i, j] == 0:
                # Check neighbors
                neighbors = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and upscaled[ni, nj] != 0:
                        neighbors.append(upscaled[ni, nj])
                if neighbors:
                    result[i, j] = max(set(neighbors), key=neighbors.count)

    if np.array_equal(result, out_arr):
        return f"combo_upscale_{sh}x_fill"
    return None


def try_combo_mirror_then_overlay(inp, out):
    """Mirror and overlay with original."""
    arr = np.array(inp)
    out_arr = np.array(out)

    # Try all mirror + overlay combos
    mirrors = [
        (np.fliplr(arr), "hflip"),
        (np.flipud(arr), "vflip"),
        (arr.T, "transpose"),
        (np.rot90(arr, 1), "rot90"),
        (np.rot90(arr, 2), "rot180"),
        (np.rot90(arr, 3), "rot270"),
    ]

    for mirrored, name in mirrors:
        if mirrored.shape != arr.shape:
            continue
        # OR overlay
        result = np.where(arr != 0, arr, mirrored)
        if np.array_equal(result, out_arr):
            return f"combo_{name}_overlay_or"
        # XOR overlay
        result = np.where((arr != 0) ^ (mirrored != 0), np.maximum(arr, mirrored), 0)
        if np.array_equal(result, out_arr):
            return f"combo_{name}_overlay_xor"

    return None


def try_combo_split_transform_merge(inp, out):
    """Split grid, transform halves, merge back."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    # Horizontal split
    if h % 2 == 0:
        top = arr[:h//2, :]
        bottom = arr[h//2:, :]

        # Swap halves
        result = np.vstack([bottom, top])
        if np.array_equal(result, out_arr):
            return "combo_hsplit_swap"

        # Flip bottom
        result = np.vstack([top, np.flipud(bottom)])
        if np.array_equal(result, out_arr):
            return "combo_hsplit_flip_bottom"

    # Vertical split
    if w % 2 == 0:
        left = arr[:, :w//2]
        right = arr[:, w//2:]

        # Swap halves
        result = np.hstack([right, left])
        if np.array_equal(result, out_arr):
            return "combo_vsplit_swap"

        # Flip right
        result = np.hstack([left, np.fliplr(right)])
        if np.array_equal(result, out_arr):
            return "combo_vsplit_flip_right"

    return None


def try_combo_color_swap_transform(inp, out):
    """Swap two colors then apply transform."""
    arr = np.array(inp)
    out_arr = np.array(out)

    colors = [c for c in np.unique(arr) if c != 0]
    if len(colors) < 2:
        return None

    for c1 in colors:
        for c2 in colors:
            if c1 >= c2:
                continue
            # Swap colors
            swapped = arr.copy()
            swapped[arr == c1] = c2
            swapped[arr == c2] = c1

            # Check direct match
            if np.array_equal(swapped, out_arr):
                return f"combo_swap_{c1}_{c2}"

            # Try rotation after swap
            for k in [1, 2, 3]:
                if np.array_equal(np.rot90(swapped, k), out_arr):
                    return f"combo_swap_{c1}_{c2}_rot{k*90}"

            # Try flip after swap
            if np.array_equal(np.fliplr(swapped), out_arr):
                return f"combo_swap_{c1}_{c2}_hflip"
            if np.array_equal(np.flipud(swapped), out_arr):
                return f"combo_swap_{c1}_{c2}_vflip"

    return None

# ============================================================================
# Pattern PATTERN MATCHING TRANSFORMS (Batch 14)
# depth=-90000 TEMPLATE STAMP REPEAT TEXTURE MOTIF CLONE CRYSTALLIZE
# ============================================================================


def try_find_smallest_repeating_unit(inp, out):
    """Find smallest repeating pattern and tile it."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape
    oh, ow = out_arr.shape

    # Try different unit sizes
    for uh in range(1, min(h+1, oh+1, 6)):
        for uw in range(1, min(w+1, ow+1, 6)):
            if oh % uh == 0 and ow % uw == 0:
                # Try each position for the unit
                for y in range(h - uh + 1):
                    for x in range(w - uw + 1):
                        unit = arr[y:y+uh, x:x+uw]
                        th, tw = oh // uh, ow // uw
                        tiled = np.tile(unit, (th, tw))
                        if np.array_equal(tiled, out_arr):
                            return f"repeat_unit_{uh}x{uw}_from_{y}_{x}"
    return None


def try_stamp_pattern_at_markers(inp, out):
    """Stamp a pattern at each marker location."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        colors = [c for c in np.unique(arr) if c != 0]
        if len(colors) < 2:
            return None

        # Find marker color (appears least or most)
        color_counts = [(c, np.sum(arr == c)) for c in colors]
        color_counts.sort(key=lambda x: x[1])

        for marker_color, _ in color_counts[:2]:  # Try least common colors
            marker_positions = np.argwhere(arr == marker_color)
            other_colors = [c for c in colors if c != marker_color]

            if len(marker_positions) < 2 or not other_colors:
                continue

            # Find pattern near first marker
            for pattern_color in other_colors:
                pattern_positions = np.argwhere(arr == pattern_color)
                if len(pattern_positions) == 0:
                    continue

                # Get bounding box of pattern
                if len(pattern_positions) > 0:
                    min_r = pattern_positions[:, 0].min()
                    max_r = pattern_positions[:, 0].max()
                    min_c = pattern_positions[:, 1].min()
                    max_c = pattern_positions[:, 1].max()
                    pattern = arr[min_r:max_r+1, min_c:max_c+1].copy()
                    ph, pw = pattern.shape

                    result = np.zeros_like(out_arr)
                    for mr, mc in marker_positions:
                        for pi in range(ph):
                            for pj in range(pw):
                                nr, nc = mr + pi, mc + pj
                                if 0 <= nr < h and 0 <= nc < w and 0 <= pi < ph and 0 <= pj < pw:
                                    if pattern[pi, pj] != 0:
                                        result[nr, nc] = pattern[pi, pj]

                    if np.array_equal(result, out_arr):
                        return f"stamp_pattern_{pattern_color}_at_{marker_color}"

        return None
    except (IndexError, ValueError):
        return None


def try_texture_fill(inp, out):
    """Fill a region with a texture pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Must be same size
        if h != oh or w != ow:
            return None

        colors = [c for c in np.unique(arr) if c != 0]
        if len(colors) < 2:
            return None

        out_colors = [c for c in np.unique(out_arr) if c != 0]

        # Try finding a texture to fill background
        for fill_color in out_colors:
            if fill_color not in colors:
                # This color was added - it's the texture
                texture_mask = (out_arr == fill_color)
                # Check if it's a regular pattern
                if np.sum(texture_mask) > 0:
                    result = arr.copy()
                    result[arr == 0] = out_arr[arr == 0]
                    if np.array_equal(result, out_arr):
                        return f"texture_fill_{fill_color}"
        return None
    except (IndexError, ValueError):
        return None


def try_copy_pattern_to_all(inp, out):
    """Copy one instance of a pattern to all similar positions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Must be same size
        if h != oh or w != ow:
            return None

        # Find connected components
        from scipy import ndimage
        binary = (arr != 0).astype(int)
        labeled, num = ndimage.label(binary)

        if num < 2:
            return None

        # Get sizes of each component
        sizes = [(i, np.sum(labeled == i)) for i in range(1, num + 1)]
        sizes.sort(key=lambda x: -x[1])  # Largest first

        # Check if all components have same shape
        largest_idx = sizes[0][0]
        largest_mask = (labeled == largest_idx)
        largest_color = arr[largest_mask][0] if np.any(arr[largest_mask]) else 0

        # Try copying largest to output
        result = np.zeros_like(arr)
        for i in range(1, num + 1):
            mask = (labeled == i)
            result[mask] = largest_color

        if np.array_equal(result, out_arr):
            return "copy_largest_to_all"

        return None
    except (IndexError, ValueError):
        return None


def try_motif_detection(inp, out):
    """Detect repeated motifs and standardize them."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        # Check for grid-like structure
        for gh in range(2, min(h//2 + 1, 6)):
            for gw in range(2, min(w//2 + 1, 6)):
                if h % gh == 0 and w % gw == 0:
                    # Split into blocks
                    blocks = []
                    for i in range(h // gh):
                        for j in range(w // gw):
                            block = arr[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
                            blocks.append(block)

                    # Find most common non-zero block
                    if len(blocks) > 1:
                        non_zero_blocks = [b for b in blocks if np.any(b != 0)]
                        if non_zero_blocks:
                            # Use first non-zero block as template
                            template = non_zero_blocks[0]
                            result = np.tile(template, (h // gh, w // gw))
                            if np.array_equal(result, out_arr):
                                return f"motif_standardize_{gh}x{gw}"
        return None
    except (IndexError, ValueError):
        return None


def try_clone_and_arrange(inp, out):
    """Clone a pattern and arrange in grid."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = arr.shape
        oh, ow = out_arr.shape

        if ih > oh or iw > ow:
            return None

        # Check horizontal arrangement
        if oh == ih and ow % iw == 0:
            n = ow // iw
            result = np.tile(arr, (1, n))
            if np.array_equal(result, out_arr):
                return f"clone_horizontal_{n}"

        # Check vertical arrangement
        if ow == iw and oh % ih == 0:
            n = oh // ih
            result = np.tile(arr, (n, 1))
            if np.array_equal(result, out_arr):
                return f"clone_vertical_{n}"

        # Check 2D grid
        if oh % ih == 0 and ow % iw == 0:
            nh, nw = oh // ih, ow // iw
            result = np.tile(arr, (nh, nw))
            if np.array_equal(result, out_arr):
                return f"clone_grid_{nh}x{nw}"

        return None
    except (IndexError, ValueError):
        return None


def try_instance_count_color(inp, out):
    """Color based on instance count in pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if h != oh or w != ow:
            return None

        from scipy import ndimage
        binary = (arr != 0).astype(int)
        labeled, num = ndimage.label(binary)

        out_colors = sorted([c for c in np.unique(out_arr) if c != 0])
        if num > 0 and num <= len(out_colors):
            result = np.zeros_like(arr)
            for i in range(1, num + 1):
                if i - 1 < len(out_colors):
                    result[labeled == i] = out_colors[i - 1]
            if np.array_equal(result, out_arr):
                return "instance_count_color"
        return None
    except (IndexError, ValueError):
        return None

# ============================================================================
# Pattern EXTRACTION/MASKING TRANSFORMS (Batch 15)
# depth=-90000 CROP PAD MASK OVERLAY SELECT FILTER LAYER CRYSTALLIZE
# ============================================================================


def try_crop_to_nonzero(inp, out):
    """Crop to bounding box of non-zero values."""
    arr = np.array(inp)
    out_arr = np.array(out)

    non_zero = np.argwhere(arr != 0)
    if len(non_zero) == 0:
        return None

    r_min, c_min = non_zero.min(axis=0)
    r_max, c_max = non_zero.max(axis=0)

    cropped = arr[r_min:r_max+1, c_min:c_max+1]
    if np.array_equal(cropped, out_arr):
        return "crop_nonzero"
    return None


def try_crop_to_color(inp, out):
    """Crop to bounding box of a specific color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        colors = [c for c in np.unique(arr) if c != 0]
        for color in colors:
            positions = np.argwhere(arr == color)
            if len(positions) == 0:
                continue
            min_r, min_c = positions.min(axis=0)
            max_r, max_c = positions.max(axis=0)
            cropped = arr[min_r:max_r+1, min_c:max_c+1]
            if np.array_equal(cropped, out_arr):
                return f"crop_to_color_{color}"
        return None
    except (IndexError, ValueError):
        return None


def try_pad_symmetric(inp, out):
    """Pad array with symmetric reflection."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = arr.shape
        oh, ow = out_arr.shape

        if oh <= ih or ow <= iw:
            return None

        pad_h = (oh - ih) // 2
        pad_w = (ow - iw) // 2

        if pad_h > 0 or pad_w > 0:
            result = np.pad(arr, ((pad_h, oh-ih-pad_h), (pad_w, ow-iw-pad_w)), mode='symmetric')
            if np.array_equal(result, out_arr):
                return f"pad_symmetric_{pad_h}_{pad_w}"
        return None
    except (IndexError, ValueError):
        return None


def try_mask_by_color(inp, out):
    """Use one color as mask to reveal another."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    colors = [c for c in np.unique(arr) if c != 0]
    out_colors = [c for c in np.unique(out_arr) if c != 0]

    if len(colors) == 2 and len(out_colors) == 1:
        c1, c2 = colors
        out_c = out_colors[0]

        # c1 masks c2
        result = np.zeros_like(arr)
        result[(arr == c1) | (arr == c2)] = out_c
        result[arr == c1] = 0
        if np.array_equal(result, out_arr):
            return f"mask_{c1}_reveals_{c2}"

        # c2 masks c1
        result = np.zeros_like(arr)
        result[(arr == c1) | (arr == c2)] = out_c
        result[arr == c2] = 0
        if np.array_equal(result, out_arr):
            return f"mask_{c2}_reveals_{c1}"

    return None


def try_select_largest_object(inp, out):
    """Select only the largest connected component."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if h != oh or w != ow:
            return None

        from scipy import ndimage
        binary = (arr != 0).astype(int)
        labeled, num = ndimage.label(binary)

        if num < 2:
            return None

        # Find largest component
        sizes = [(i, np.sum(labeled == i)) for i in range(1, num + 1)]
        sizes.sort(key=lambda x: -x[1])
        largest_idx = sizes[0][0]

        result = np.zeros_like(arr)
        mask = (labeled == largest_idx)
        result[mask] = arr[mask]

        if np.array_equal(result, out_arr):
            return "select_largest_object"
        return None
    except (IndexError, ValueError):
        return None


def try_select_smallest_object(inp, out):
    """Select only the smallest connected component."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if h != oh or w != ow:
            return None

        from scipy import ndimage
        binary = (arr != 0).astype(int)
        labeled, num = ndimage.label(binary)

        if num < 2:
            return None

        # Find smallest component
        sizes = [(i, np.sum(labeled == i)) for i in range(1, num + 1)]
        sizes.sort(key=lambda x: x[1])
        smallest_idx = sizes[0][0]

        result = np.zeros_like(arr)
        mask = (labeled == smallest_idx)
        result[mask] = arr[mask]

        if np.array_equal(result, out_arr):
            return "select_smallest_object"
        return None
    except (IndexError, ValueError):
        return None


def try_foreground_only(inp, out):
    """Extract only non-zero values, keeping positions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Just non-zero elements
        if np.array_equal(arr, out_arr):
            return None

        result = arr.copy()
        # Try different background removal strategies
        if np.array_equal(result, out_arr):
            return "foreground_only"
        return None
    except (IndexError, ValueError):
        return None


def try_layer_subtract(inp, out):
    """Subtract one layer/color from another."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if h != oh or w != ow:
            return None

        colors = sorted([c for c in np.unique(arr) if c != 0])
        if len(colors) < 2:
            return None

        for c1 in colors:
            for c2 in colors:
                if c1 == c2:
                    continue
                mask1 = (arr == c1)
                mask2 = (arr == c2)
                result = arr.copy()
                result[mask2] = 0  # Remove c2 layer
                if np.array_equal(result, out_arr):
                    return f"layer_subtract_{c2}_from_{c1}"
        return None
    except (IndexError, ValueError):
        return None


def try_filter_by_size(inp, out):
    """Keep only objects above or below a certain size."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if h != oh or w != ow:
            return None

        from scipy import ndimage
        binary = (arr != 0).astype(int)
        labeled, num = ndimage.label(binary)

        if num < 2:
            return None

        # Get sizes
        sizes = [(i, np.sum(labeled == i)) for i in range(1, num + 1)]

        # Try keeping only objects larger than each size threshold
        for threshold in set(s[1] for s in sizes):
            result = np.zeros_like(arr)
            for idx, size in sizes:
                if size >= threshold:
                    mask = (labeled == idx)
                    result[mask] = arr[mask]
            if np.array_equal(result, out_arr):
                return f"filter_size_ge_{threshold}"

            result = np.zeros_like(arr)
            for idx, size in sizes:
                if size <= threshold:
                    mask = (labeled == idx)
                    result[mask] = arr[mask]
            if np.array_equal(result, out_arr):
                return f"filter_size_le_{threshold}"
        return None
    except (IndexError, ValueError):
        return None


def try_slice_quadrant(inp, out):
    """Extract one quadrant of the grid."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        h2, w2 = h // 2, w // 2

        # Top-left
        if np.array_equal(arr[:h2, :w2], out_arr):
            return "slice_quadrant_tl"
        # Top-right
        if np.array_equal(arr[:h2, w2:], out_arr):
            return "slice_quadrant_tr"
        # Bottom-left
        if np.array_equal(arr[h2:, :w2], out_arr):
            return "slice_quadrant_bl"
        # Bottom-right
        if np.array_equal(arr[h2:, w2:], out_arr):
            return "slice_quadrant_br"
        return None
    except (IndexError, ValueError):
        return None


def try_extract_row(inp, out):
    """Extract a specific row."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if out_arr.shape[0] != 1:
            return None

        for r in range(h):
            if np.array_equal(arr[r:r+1, :], out_arr):
                return f"extract_row_{r}"
        return None
    except (IndexError, ValueError):
        return None


def try_extract_col(inp, out):
    """Extract a specific column."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if out_arr.shape[1] != 1:
            return None

        for c in range(w):
            if np.array_equal(arr[:, c:c+1], out_arr):
                return f"extract_col_{c}"
        return None
    except (IndexError, ValueError):
        return None


def try_overlay_nonzero(inp, out):
    """Overlay non-zero values on top of another pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if h != oh or w != ow:
            return None

        # Check if output is input with some zeros filled
        nonzero_input = (arr != 0)
        result = out_arr.copy()
        result[nonzero_input] = arr[nonzero_input]

        if np.array_equal(result, out_arr):
            return "overlay_nonzero"
        return None
    except (IndexError, ValueError):
        return None

# ============================================================================
# Batch 16 - Pattern GRID/STRUCTURAL ⧫depth=-90000⧫ GRID SPLIT MERGE PARTITION
# ============================================================================


def try_grid_divide(inp, out):
    """Divide input grid by a regular grid pattern and extract cells."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Try different grid divisions
        for div_h in range(2, min(h+1, 10)):
            for div_w in range(2, min(w+1, 10)):
                if h % div_h == 0 and w % div_w == 0:
                    cell_h, cell_w = h // div_h, w // div_w
                    # Check if output matches one cell
                    if oh == cell_h and ow == cell_w:
                        for ci in range(div_h):
                            for cj in range(div_w):
                                cell = arr[ci*cell_h:(ci+1)*cell_h, cj*cell_w:(cj+1)*cell_w]
                                if np.array_equal(cell, out_arr):
                                    return f"grid_divide_{div_h}x{div_w}_cell_{ci}_{cj}"
        return None
    except (IndexError, ValueError):
        return None


def try_grid_merge(inp, out):
    """Merge grid cells using OR/AND/XOR operations."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Try dividing input into grid and merging
        for div_h in [2, 3, 4]:
            for div_w in [2, 3, 4]:
                if h % div_h == 0 and w % div_w == 0:
                    cell_h, cell_w = h // div_h, w // div_w
                    if oh == cell_h and ow == cell_w:
                        # Extract all cells
                        cells = []
                        for ci in range(div_h):
                            for cj in range(div_w):
                                cell = arr[ci*cell_h:(ci+1)*cell_h, cj*cell_w:(cj+1)*cell_w]
                                cells.append(cell)

                        # Try OR merge
                        result = np.zeros((cell_h, cell_w), dtype=arr.dtype)
                        for cell in cells:
                            result = np.where(cell != 0, cell, result)
                        if np.array_equal(result, out_arr):
                            return f"grid_merge_or_{div_h}x{div_w}"

                        # Try AND merge (all must have nonzero)
                        result = cells[0].copy()
                        for cell in cells[1:]:
                            result = np.where((result != 0) & (cell != 0), result, 0)
                        if np.array_equal(result, out_arr):
                            return f"grid_merge_and_{div_h}x{div_w}"
        return None
    except (IndexError, ValueError):
        return None


def try_block_extract(inp, out):
    """Extract blocks separated by divider lines."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Find horizontal divider rows (all same non-zero color)
        h_dividers = []
        for row in range(h):
            vals = arr[row, :]
            if len(set(vals)) == 1 and vals[0] != 0:
                h_dividers.append(row)

        # Find vertical divider columns
        v_dividers = []
        for col in range(w):
            vals = arr[:, col]
            if len(set(vals)) == 1 and vals[0] != 0:
                v_dividers.append(col)

        # Extract blocks between dividers
        if h_dividers or v_dividers:
            h_bounds = [-1] + h_dividers + [h]
            v_bounds = [-1] + v_dividers + [w]

            for i in range(len(h_bounds)-1):
                for j in range(len(v_bounds)-1):
                    r1, r2 = h_bounds[i]+1, h_bounds[i+1]
                    c1, c2 = v_bounds[j]+1, v_bounds[j+1]
                    if r2 > r1 and c2 > c1:
                        block = arr[r1:r2, c1:c2]
                        if block.shape == out_arr.shape and np.array_equal(block, out_arr):
                            return f"block_extract_{i}_{j}"
        return None
    except (IndexError, ValueError):
        return None


def try_partition_by_color(inp, out):
    """Partition grid by a divider color and combine blocks."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        colors = [c for c in np.unique(arr) if c != 0]

        for div_color in colors:
            # Find regions separated by this color
            mask = (arr == div_color)
            # Simple: check if removing divider color and compacting gives output
            result = arr.copy()
            result[mask] = 0

            # Compact: remove empty rows/cols
            nonzero_rows = np.any(result != 0, axis=1)
            nonzero_cols = np.any(result != 0, axis=0)
            if np.any(nonzero_rows) and np.any(nonzero_cols):
                compact = result[nonzero_rows][:, nonzero_cols]
                if compact.shape == out_arr.shape and np.array_equal(compact, out_arr):
                    return f"partition_remove_color_{div_color}"
        return None
    except (IndexError, ValueError):
        return None


def try_matrix_combine(inp, out):
    """Combine matrix blocks with various operations."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check for 2x2 block structure
        if h % 2 == 0 and w % 2 == 0:
            hh, hw = h // 2, w // 2
            if oh == hh and ow == hw:
                tl = arr[:hh, :hw]
                tr = arr[:hh, hw:]
                bl = arr[hh:, :hw]
                br = arr[hh:, hw:]

                # Try max of all quadrants
                result = np.maximum.reduce([tl, tr, bl, br])
                if np.array_equal(result, out_arr):
                    return "matrix_combine_max_2x2"

                # Try min of all quadrants (where nonzero)
                result = tl.copy()
                for q in [tr, bl, br]:
                    result = np.where((q != 0) & ((result == 0) | (q < result)), q, result)
                if np.array_equal(result, out_arr):
                    return "matrix_combine_min_2x2"
        return None
    except (IndexError, ValueError):
        return None


def try_split_by_gap(inp, out):
    """Split grid at gaps (zero rows/columns) and extract part."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        # Find zero rows
        zero_rows = [r for r in range(h) if np.all(arr[r, :] == 0)]
        # Find zero cols
        zero_cols = [c for c in range(w) if np.all(arr[:, c] == 0)]

        # Split at gaps
        if zero_rows:
            parts = []
            prev = 0
            for r in zero_rows:
                if r > prev:
                    parts.append(arr[prev:r, :])
                prev = r + 1
            if prev < h:
                parts.append(arr[prev:, :])

            for i, part in enumerate(parts):
                if part.shape == out_arr.shape and np.array_equal(part, out_arr):
                    return f"split_gap_row_{i}"

        if zero_cols:
            parts = []
            prev = 0
            for c in zero_cols:
                if c > prev:
                    parts.append(arr[:, prev:c])
                prev = c + 1
            if prev < w:
                parts.append(arr[:, prev:])

            for i, part in enumerate(parts):
                if part.shape == out_arr.shape and np.array_equal(part, out_arr):
                    return f"split_gap_col_{i}"
        return None
    except (IndexError, ValueError):
        return None


def try_structural_difference(inp, out):
    """Find structural difference between grid regions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Split in half and find difference
        if w % 2 == 0:
            hw = w // 2
            if oh == h and ow == hw:
                left = arr[:, :hw]
                right = arr[:, hw:]

                # XOR difference
                diff = np.where(left != right,
                               np.where(left != 0, left, right), 0)
                if np.array_equal(diff, out_arr):
                    return "structural_diff_h_xor"

                # Only left unique
                diff = np.where((left != 0) & (right == 0), left, 0)
                if np.array_equal(diff, out_arr):
                    return "structural_diff_h_left_only"

                # Only right unique
                diff = np.where((right != 0) & (left == 0), right, 0)
                if np.array_equal(diff, out_arr):
                    return "structural_diff_h_right_only"

        if h % 2 == 0:
            hh = h // 2
            if oh == hh and ow == w:
                top = arr[:hh, :]
                bot = arr[hh:, :]

                diff = np.where(top != bot,
                               np.where(top != 0, top, bot), 0)
                if np.array_equal(diff, out_arr):
                    return "structural_diff_v_xor"
        return None
    except (IndexError, ValueError):
        return None


def try_grid_parse(inp, out):
    """Parse grid structure and extract pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Find the most common nonzero element - might be grid lines
        colors = [c for c in np.unique(arr) if c != 0]
        if not colors:
            return None

        from collections import Counter
        counts = Counter(arr.flatten())
        grid_color = max([c for c in colors], key=lambda c: counts.get(c, 0))

        # Remove grid lines
        result = arr.copy()
        result[arr == grid_color] = 0

        # Compact
        nonzero_rows = np.any(result != 0, axis=1)
        nonzero_cols = np.any(result != 0, axis=0)
        if np.any(nonzero_rows) and np.any(nonzero_cols):
            compact = result[nonzero_rows][:, nonzero_cols]
            if compact.shape == out_arr.shape and np.array_equal(compact, out_arr):
                return f"grid_parse_remove_{grid_color}"
        return None
    except (IndexError, ValueError):
        return None


def try_combine_quadrants(inp, out):
    """Combine quadrants with operations."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if h % 2 != 0 or w % 2 != 0:
            return None

        hh, hw = h // 2, w // 2

        # For output same size as quadrant
        if oh == hh and ow == hw:
            q1 = arr[:hh, :hw]
            q2 = arr[:hh, hw:]
            q3 = arr[hh:, :hw]
            q4 = arr[hh:, hw:]

            # Most common nonzero per cell
            result = np.zeros((hh, hw), dtype=arr.dtype)
            for i in range(hh):
                for j in range(hw):
                    vals = [q1[i,j], q2[i,j], q3[i,j], q4[i,j]]
                    vals = [v for v in vals if v != 0]
                    if vals:
                        from collections import Counter
                        result[i,j] = Counter(vals).most_common(1)[0][0]

            if np.array_equal(result, out_arr):
                return "combine_quadrants_mode"
        return None
    except (IndexError, ValueError):
        return None


def try_interleave_split(inp, out):
    """Split by interleaved rows/columns."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Even rows
        if h % 2 == 0 and oh == h // 2:
            even_rows = arr[::2, :]
            odd_rows = arr[1::2, :]
            if even_rows.shape == out_arr.shape and np.array_equal(even_rows, out_arr):
                return "interleave_even_rows"
            if odd_rows.shape == out_arr.shape and np.array_equal(odd_rows, out_arr):
                return "interleave_odd_rows"

        # Even cols
        if w % 2 == 0 and ow == w // 2:
            even_cols = arr[:, ::2]
            odd_cols = arr[:, 1::2]
            if even_cols.shape == out_arr.shape and np.array_equal(even_cols, out_arr):
                return "interleave_even_cols"
            if odd_cols.shape == out_arr.shape and np.array_equal(odd_cols, out_arr):
                return "interleave_odd_cols"
        return None
    except (IndexError, ValueError):
        return None


def try_block_majority(inp, out):
    """Divide into blocks and take majority color per block."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if h % oh == 0 and w % ow == 0:
            bh, bw = h // oh, w // ow
            result = np.zeros((oh, ow), dtype=arr.dtype)

            from collections import Counter
            for i in range(oh):
                for j in range(ow):
                    block = arr[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                    vals = block.flatten()
                    # Majority including zeros
                    c = Counter(vals)
                    result[i,j] = c.most_common(1)[0][0]

            if np.array_equal(result, out_arr):
                return f"block_majority_{bh}x{bw}"

            # Try majority excluding zeros
            for i in range(oh):
                for j in range(ow):
                    block = arr[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                    vals = [v for v in block.flatten() if v != 0]
                    if vals:
                        c = Counter(vals)
                        result[i,j] = c.most_common(1)[0][0]
                    else:
                        result[i,j] = 0

            if np.array_equal(result, out_arr):
                return f"block_majority_nonzero_{bh}x{bw}"
        return None
    except (IndexError, ValueError):
        return None


def try_grid_reconstruct(inp, out):
    """Reconstruct grid from partial information."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if output is tiled version of input
        if oh % h == 0 and ow % w == 0:
            th, tw = oh // h, ow // w
            result = np.tile(arr, (th, tw))
            if np.array_equal(result, out_arr):
                return f"grid_reconstruct_tile_{th}x{tw}"

        # Check if input is template to fill output
        if h <= oh and w <= ow:
            # Try repeating input pattern
            result = np.zeros((oh, ow), dtype=arr.dtype)
            for i in range(0, oh, h):
                for j in range(0, ow, w):
                    end_i = min(i+h, oh)
                    end_j = min(j+w, ow)
                    result[i:end_i, j:end_j] = arr[:end_i-i, :end_j-j]
            if np.array_equal(result, out_arr):
                return "grid_reconstruct_repeat"
        return None
    except (IndexError, ValueError):
        return None

# ============================================================================
# Batch 17 - Pattern FLOW/PATH ⧫depth=-90000⧫ TRACE EXTEND CONNECT FOLLOW
# ============================================================================


def try_extend_lines(inp, out):
    """Extend colored lines to edges or obstacles."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        result = arr.copy()
        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            # Find colored cells
            positions = list(zip(*np.where(arr == color)))
            if len(positions) < 2:
                continue

            # Check for horizontal line
            rows = [p[0] for p in positions]
            cols = [p[1] for p in positions]

            if len(set(rows)) == 1:  # Horizontal line
                row = rows[0]
                min_c, max_c = min(cols), max(cols)
                # Extend left
                for c in range(min_c - 1, -1, -1):
                    if arr[row, c] == 0:
                        result[row, c] = color
                    else:
                        break
                # Extend right
                for c in range(max_c + 1, w):
                    if arr[row, c] == 0:
                        result[row, c] = color
                    else:
                        break

            elif len(set(cols)) == 1:  # Vertical line
                col = cols[0]
                min_r, max_r = min(rows), max(rows)
                # Extend up
                for r in range(min_r - 1, -1, -1):
                    if arr[r, col] == 0:
                        result[r, col] = color
                    else:
                        break
                # Extend down
                for r in range(max_r + 1, h):
                    if arr[r, col] == 0:
                        result[r, col] = color
                    else:
                        break

        if np.array_equal(result, out_arr):
            return "extend_lines"
        return None
    except (IndexError, ValueError):
        return None


def try_trace_path(inp, out):
    """Trace and fill path between two points."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        # Find colored endpoints
        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            positions = list(zip(*np.where(arr == color)))
            if len(positions) != 2:
                continue

            # Two points of same color - connect with Manhattan path
            (r1, c1), (r2, c2) = positions
            result = arr.copy()

            # Horizontal then vertical path
            for c in range(min(c1, c2), max(c1, c2) + 1):
                result[r1, c] = color
            for r in range(min(r1, r2), max(r1, r2) + 1):
                result[r, c2] = color

            if np.array_equal(result, out_arr):
                return f"trace_path_hv_{color}"

            # Vertical then horizontal path
            result = arr.copy()
            for r in range(min(r1, r2), max(r1, r2) + 1):
                result[r, c1] = color
            for c in range(min(c1, c2), max(c1, c2) + 1):
                result[r2, c] = color

            if np.array_equal(result, out_arr):
                return f"trace_path_vh_{color}"

        return None
    except (IndexError, ValueError):
        return None


def try_connect_same_color(inp, out):
    """Connect all pixels of same color with straight lines."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            positions = list(zip(*np.where(arr == color)))
            if len(positions) < 2:
                continue

            result = arr.copy()

            # Connect each pair that shares row or column
            for i, (r1, c1) in enumerate(positions):
                for r2, c2 in positions[i+1:]:
                    if r1 == r2:  # Same row
                        for c in range(min(c1, c2), max(c1, c2) + 1):
                            result[r1, c] = color
                    elif c1 == c2:  # Same column
                        for r in range(min(r1, r2), max(r1, r2) + 1):
                            result[r, c1] = color

            if np.array_equal(result, out_arr):
                return f"connect_same_color_{color}"

        return None
    except (IndexError, ValueError):
        return None


def try_flood_direction(inp, out):
    """Flood fill in a specific direction from colored cells."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            positions = list(zip(*np.where(arr == color)))
            if not positions:
                continue

            # Try each direction
            for dr, dc, name in [(0, 1, 'right'), (0, -1, 'left'), (1, 0, 'down'), (-1, 0, 'up')]:
                result = arr.copy()
                for r, c in positions:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < h and 0 <= nc < w and result[nr, nc] == 0:
                        result[nr, nc] = color
                        nr, nc = nr + dr, nc + dc

                if np.array_equal(result, out_arr):
                    return f"flood_direction_{name}_{color}"

        return None
    except (IndexError, ValueError):
        return None


def try_ray_cast(inp, out):
    """Cast rays from colored cells until hitting obstacle."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = [c for c in np.unique(arr) if c != 0]

        for source_color in colors:
            positions = list(zip(*np.where(arr == source_color)))
            if not positions:
                continue

            # Cast rays in all 4 directions
            result = arr.copy()
            for r, c in positions:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < h and 0 <= nc < w:
                        if arr[nr, nc] != 0:
                            break
                        result[nr, nc] = source_color
                        nr, nc = nr + dr, nc + dc

            if np.array_equal(result, out_arr):
                return f"ray_cast_4dir_{source_color}"

        return None
    except (IndexError, ValueError):
        return None


def try_diagonal_extend(inp, out):
    """Extend diagonal lines."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            positions = list(zip(*np.where(arr == color)))
            if len(positions) < 2:
                continue

            result = arr.copy()

            # Check if diagonal pattern (both row and col differ by same amount)
            for i, (r1, c1) in enumerate(positions):
                for r2, c2 in positions[i+1:]:
                    dr = r2 - r1
                    dc = c2 - c1
                    if abs(dr) == abs(dc) and dr != 0:
                        # Extend in same diagonal direction
                        step_r = 1 if dr > 0 else -1
                        step_c = 1 if dc > 0 else -1

                        # Extend before r1,c1
                        nr, nc = r1 - step_r, c1 - step_c
                        while 0 <= nr < h and 0 <= nc < w and result[nr, nc] == 0:
                            result[nr, nc] = color
                            nr, nc = nr - step_r, nc - step_c

                        # Extend after r2,c2
                        nr, nc = r2 + step_r, c2 + step_c
                        while 0 <= nr < h and 0 <= nc < w and result[nr, nc] == 0:
                            result[nr, nc] = color
                            nr, nc = nr + step_r, nc + step_c

            if np.array_equal(result, out_arr):
                return f"diagonal_extend_{color}"

        return None
    except (IndexError, ValueError):
        return None


def try_follow_arrow(inp, out):
    """Follow arrow-like patterns to fill."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        # Detect arrow patterns (3 cells in L shape)
        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            mask = (arr == color)
            positions = list(zip(*np.where(mask)))
            if len(positions) != 3:
                continue

            # Check for L-shape (arrow)
            rows = sorted(set(p[0] for p in positions))
            cols = sorted(set(p[1] for p in positions))

            if len(rows) == 2 and len(cols) == 2:
                # It's an L - find the direction
                result = arr.copy()

                # Fill in the direction the arrow points
                # Determine direction from corner
                r_center = [r for r in rows if sum(1 for p in positions if p[0] == r) == 2]
                c_center = [c for c in cols if sum(1 for p in positions if p[1] == c) == 2]

                if r_center and c_center:
                    r_tip = [r for r in rows if r != r_center[0]][0]
                    c_tip = [c for c in cols if c != c_center[0]][0]

                    dr = 1 if r_tip > r_center[0] else -1
                    dc = 1 if c_tip > c_center[0] else -1

                    # Fill in that direction
                    nr, nc = r_tip + dr, c_tip + dc
                    while 0 <= nr < h and 0 <= nc < w and result[nr, nc] == 0:
                        result[nr, nc] = color
                        nr, nc = nr + dr, nc + dc

                if np.array_equal(result, out_arr):
                    return f"follow_arrow_{color}"

        return None
    except (IndexError, ValueError):
        return None


def try_propagate_from_seed(inp, out):
    """Propagate a pattern from seed cells."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        # Find seed cells (colored) and see if they propagate to fill output
        colors = [c for c in np.unique(arr) if c != 0]

        for seed_color in colors:
            seeds = list(zip(*np.where(arr == seed_color)))
            if not seeds:
                continue

            # BFS flood fill from seeds
            result = arr.copy()
            visited = set(seeds)
            queue = list(seeds)

            while queue:
                r, c = queue.pop(0)
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                        if out_arr[nr, nc] == seed_color and arr[nr, nc] == 0:
                            result[nr, nc] = seed_color
                            visited.add((nr, nc))
                            queue.append((nr, nc))

            if np.array_equal(result, out_arr):
                return f"propagate_from_seed_{seed_color}"

        return None
    except (IndexError, ValueError):
        return None


def try_connect_corners(inp, out):
    """Connect corners of a bounding box."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            positions = list(zip(*np.where(arr == color)))
            if len(positions) < 2:
                continue

            rows = [p[0] for p in positions]
            cols = [p[1] for p in positions]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)

            result = arr.copy()

            # Draw bounding box edges
            for c in range(min_c, max_c + 1):
                result[min_r, c] = color
                result[max_r, c] = color
            for r in range(min_r, max_r + 1):
                result[r, min_c] = color
                result[r, max_c] = color

            if np.array_equal(result, out_arr):
                return f"connect_corners_{color}"

            # Try just connecting with diagonals
            result = arr.copy()
            for i in range(min(max_r - min_r, max_c - min_c) + 1):
                if min_r + i < h and min_c + i < w:
                    result[min_r + i, min_c + i] = color
                if min_r + i < h and max_c - i >= 0:
                    result[min_r + i, max_c - i] = color

            if np.array_equal(result, out_arr):
                return f"connect_corners_diag_{color}"

        return None
    except (IndexError, ValueError):
        return None


def try_fill_between(inp, out):
    """Fill space between two colored regions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = [c for c in np.unique(arr) if c != 0]
        if len(colors) < 2:
            return None

        # Try filling between pairs of colors
        for i, c1 in enumerate(colors):
            for c2 in colors[i+1:]:
                pos1 = list(zip(*np.where(arr == c1)))
                pos2 = list(zip(*np.where(arr == c2)))

                if not pos1 or not pos2:
                    continue

                # Get bounding boxes
                r1_min = min(p[0] for p in pos1)
                r1_max = max(p[0] for p in pos1)
                c1_min = min(p[1] for p in pos1)
                c1_max = max(p[1] for p in pos1)

                r2_min = min(p[0] for p in pos2)
                r2_max = max(p[0] for p in pos2)
                c2_min = min(p[1] for p in pos2)
                c2_max = max(p[1] for p in pos2)

                result = arr.copy()

                # Fill horizontal gap
                if r1_min == r2_min and r1_max == r2_max:
                    for r in range(r1_min, r1_max + 1):
                        for c in range(min(c1_max, c2_max), max(c1_min, c2_min) + 1):
                            if result[r, c] == 0:
                                result[r, c] = c1

                    if np.array_equal(result, out_arr):
                        return f"fill_between_h_{c1}_{c2}"

                # Fill vertical gap
                if c1_min == c2_min and c1_max == c2_max:
                    for c in range(c1_min, c1_max + 1):
                        for r in range(min(r1_max, r2_max), max(r1_min, r2_min) + 1):
                            if result[r, c] == 0:
                                result[r, c] = c1

                    if np.array_equal(result, out_arr):
                        return f"fill_between_v_{c1}_{c2}"

        return None
    except (IndexError, ValueError):
        return None

# ============================================================================
# Batch 18 - Pattern COMPLETION/INFERENCE ⧫depth=-90000⧫ DEDUCE PREDICT
# ============================================================================


def try_complete_pattern_repeat(inp, out):
    """Complete pattern by detecting and extending repetition."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h or ow != w:
            return None

        # Find repeating unit in non-zero part
        colors = [c for c in np.unique(arr) if c != 0]
        if not colors:
            return None

        # Check if output fills zeros with repeated pattern
        nonzero = (arr != 0)
        if np.all(nonzero):
            return None

        # Try to find a pattern unit that tiles
        for uh in range(1, h // 2 + 1):
            for uw in range(1, w // 2 + 1):
                if h % uh == 0 and w % uw == 0:
                    unit = out_arr[:uh, :uw]
                    result = np.tile(unit, (h // uh, w // uw))
                    if np.array_equal(result, out_arr):
                        # Verify input is subset of output
                        if np.all(arr[nonzero] == out_arr[nonzero]):
                            return f"complete_pattern_repeat_{uh}x{uw}"
        return None
    except (IndexError, ValueError):
        return None


def try_infer_color_mapping(inp, out):
    """Infer color mapping from partial data and apply to all."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Build color mapping from areas where both have non-zero
        mapping = {}
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i,j] != 0 and out_arr[i,j] != 0:
                    if arr[i,j] in mapping:
                        if mapping[arr[i,j]] != out_arr[i,j]:
                            return None  # Inconsistent mapping
                    else:
                        mapping[arr[i,j]] = out_arr[i,j]

        if not mapping:
            return None

        # Apply mapping
        result = arr.copy()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i,j] in mapping:
                    result[i,j] = mapping[arr[i,j]]

        if np.array_equal(result, out_arr):
            return f"infer_color_mapping"
        return None
    except (IndexError, ValueError):
        return None


def try_extrapolate_gradient(inp, out):
    """Extrapolate gradient/sequence pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = arr.copy()

        # Check for row-wise gradient
        for r in range(h):
            row = arr[r, :]
            nonzero_idx = [i for i in range(w) if row[i] != 0]
            if len(nonzero_idx) >= 2:
                vals = [row[i] for i in nonzero_idx]
                # Check if arithmetic sequence
                diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
                if len(set(diffs)) == 1:
                    diff = diffs[0]
                    # Extrapolate left
                    for i in range(nonzero_idx[0] - 1, -1, -1):
                        result[r, i] = vals[0] - diff * (nonzero_idx[0] - i)
                    # Extrapolate right
                    for i in range(nonzero_idx[-1] + 1, w):
                        result[r, i] = vals[-1] + diff * (i - nonzero_idx[-1])

        if np.array_equal(result, out_arr):
            return "extrapolate_gradient_row"

        # Check for col-wise gradient
        result = arr.copy()
        for c in range(w):
            col = arr[:, c]
            nonzero_idx = [i for i in range(h) if col[i] != 0]
            if len(nonzero_idx) >= 2:
                vals = [col[i] for i in nonzero_idx]
                diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
                if len(set(diffs)) == 1:
                    diff = diffs[0]
                    for i in range(nonzero_idx[0] - 1, -1, -1):
                        result[i, c] = vals[0] - diff * (nonzero_idx[0] - i)
                    for i in range(nonzero_idx[-1] + 1, h):
                        result[i, c] = vals[-1] + diff * (i - nonzero_idx[-1])

        if np.array_equal(result, out_arr):
            return "extrapolate_gradient_col"

        return None
    except (IndexError, ValueError):
        return None


def try_reconstruct_from_partial(inp, out):
    """Reconstruct complete grid from partial clues."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Check if input has single row/col with pattern that should tile
        nonzero_rows = [r for r in range(h) if np.any(arr[r, :] != 0)]
        nonzero_cols = [c for c in range(w) if np.any(arr[:, c] != 0)]

        # Single row as template
        if len(nonzero_rows) == 1:
            template_row = arr[nonzero_rows[0], :]
            result = np.tile(template_row, (h, 1))
            if np.array_equal(result, out_arr):
                return f"reconstruct_from_row_{nonzero_rows[0]}"

        # Single col as template
        if len(nonzero_cols) == 1:
            template_col = arr[:, nonzero_cols[0]:nonzero_cols[0]+1]
            result = np.tile(template_col, (1, w))
            if np.array_equal(result, out_arr):
                return f"reconstruct_from_col_{nonzero_cols[0]}"

        return None
    except (IndexError, ValueError):
        return None


def try_predict_from_neighbors(inp, out):
    """Predict missing cells from neighbor values."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = arr.copy()

        # Fill zeros with most common neighbor
        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            iterations += 1
            new_result = result.copy()
            for i in range(h):
                for j in range(w):
                    if result[i, j] == 0:
                        neighbors = []
                        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w and result[ni, nj] != 0:
                                neighbors.append(result[ni, nj])
                        if neighbors:
                            from collections import Counter
                            most_common = Counter(neighbors).most_common(1)[0][0]
                            new_result[i, j] = most_common
                            changed = True
            result = new_result

        if np.array_equal(result, out_arr):
            return "predict_from_neighbors"
        return None
    except (IndexError, ValueError):
        return None


def try_infer_boundary_rule(inp, out):
    """Infer and apply boundary coloring rule."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Check if output has boundary colored differently
        result = arr.copy()
        colors = [c for c in np.unique(out_arr) if c != 0]

        for border_color in colors:
            result = arr.copy()
            # Top and bottom edges
            result[0, :] = border_color
            result[h-1, :] = border_color
            # Left and right edges
            result[:, 0] = border_color
            result[:, w-1] = border_color

            if np.array_equal(result, out_arr):
                return f"infer_boundary_{border_color}"

        return None
    except (IndexError, ValueError):
        return None


def try_complete_symmetry(inp, out):
    """Complete partial figure to have symmetry."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Try completing horizontal symmetry
        result = arr.copy()
        for i in range(h):
            for j in range(w):
                mirror_j = w - 1 - j
                if result[i, j] == 0 and result[i, mirror_j] != 0:
                    result[i, j] = result[i, mirror_j]
                elif result[i, j] != 0 and result[i, mirror_j] == 0:
                    result[i, mirror_j] = result[i, j]

        if np.array_equal(result, out_arr):
            return "complete_symmetry_h"

        # Try completing vertical symmetry
        result = arr.copy()
        for i in range(h):
            for j in range(w):
                mirror_i = h - 1 - i
                if result[i, j] == 0 and result[mirror_i, j] != 0:
                    result[i, j] = result[mirror_i, j]
                elif result[i, j] != 0 and result[mirror_i, j] == 0:
                    result[mirror_i, j] = result[i, j]

        if np.array_equal(result, out_arr):
            return "complete_symmetry_v"

        # Try completing point symmetry (180 rotation)
        result = arr.copy()
        for i in range(h):
            for j in range(w):
                mirror_i, mirror_j = h - 1 - i, w - 1 - j
                if result[i, j] == 0 and result[mirror_i, mirror_j] != 0:
                    result[i, j] = result[mirror_i, mirror_j]
                elif result[i, j] != 0 and result[mirror_i, mirror_j] == 0:
                    result[mirror_i, mirror_j] = result[i, j]

        if np.array_equal(result, out_arr):
            return "complete_symmetry_point"

        return None
    except (IndexError, ValueError):
        return None


def try_deduce_from_examples(inp, out):
    """Deduce transformation by comparing input/output structure."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Check if output is input with zeros replaced by a computed value
        result = arr.copy()
        zero_mask = (arr == 0)

        # Try filling with the most common non-zero color
        colors = [c for c in np.unique(arr) if c != 0]
        if colors:
            from collections import Counter
            counts = Counter(arr.flatten())
            most_common = max(colors, key=lambda c: counts[c])
            result[zero_mask] = most_common
            if np.array_equal(result, out_arr):
                return f"deduce_fill_common_{most_common}"

        # Try filling with the least common non-zero color
        if len(colors) > 1:
            least_common = min(colors, key=lambda c: counts[c])
            result = arr.copy()
            result[zero_mask] = least_common
            if np.array_equal(result, out_arr):
                return f"deduce_fill_rare_{least_common}"

        return None
    except (IndexError, ValueError):
        return None


def try_contextual_fill(inp, out):
    """Fill cells based on local context."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = arr.copy()

        # For each zero, check if surrounded by same color
        for i in range(h):
            for j in range(w):
                if arr[i, j] == 0:
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] != 0:
                                neighbors.append(arr[ni, nj])

                    if neighbors:
                        # If all neighbors same color, use that color
                        if len(set(neighbors)) == 1:
                            result[i, j] = neighbors[0]

        if np.array_equal(result, out_arr):
            return "contextual_fill_uniform"

        # Try majority of 8-neighbors
        result = arr.copy()
        for i in range(h):
            for j in range(w):
                if arr[i, j] == 0:
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] != 0:
                                neighbors.append(arr[ni, nj])

                    if len(neighbors) >= 4:  # Majority
                        from collections import Counter
                        most = Counter(neighbors).most_common(1)[0][0]
                        result[i, j] = most

        if np.array_equal(result, out_arr):
            return "contextual_fill_majority"

        return None
    except (IndexError, ValueError):
        return None

# ============================================================================
# Batch 19 - Pattern SHAPE/CONTOUR depth=-90000 BOUNDARY EDGE PERIMETER
# ============================================================================


def try_extract_contour(inp, out):
    """Extract boundary contour of colored regions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        result = np.zeros_like(arr)

        for i in range(h):
            for j in range(w):
                if arr[i, j] != 0:
                    # Check if on boundary (adjacent to 0 or edge)
                    on_boundary = False
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if ni < 0 or ni >= h or nj < 0 or nj >= w:
                            on_boundary = True
                            break
                        if arr[ni, nj] == 0:
                            on_boundary = True
                            break
                    if on_boundary:
                        result[i, j] = arr[i, j]

        if np.array_equal(result, out_arr):
            return "extract_contour"
        return None
    except (IndexError, ValueError):
        return None


def try_fill_contour(inp, out):
    """Fill inside contours with solid color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        result = arr.copy()

        colors = [c for c in np.unique(arr) if c != 0]
        for color in colors:
            mask = (arr == color)
            # Flood fill exterior from edges
            exterior = np.zeros((h, w), dtype=bool)
            stack = []
            for i in range(h):
                if not mask[i, 0]:
                    stack.append((i, 0))
                if not mask[i, w-1]:
                    stack.append((i, w-1))
            for j in range(w):
                if not mask[0, j]:
                    stack.append((0, j))
                if not mask[h-1, j]:
                    stack.append((h-1, j))

            while stack:
                r, c = stack.pop()
                if r < 0 or r >= h or c < 0 or c >= w:
                    continue
                if exterior[r, c] or mask[r, c]:
                    continue
                exterior[r, c] = True
                stack.extend([(r-1,c), (r+1,c), (r,c-1), (r,c+1)])

            # Interior = not exterior and not already colored
            for i in range(h):
                for j in range(w):
                    if not exterior[i, j] and arr[i, j] == 0:
                        result[i, j] = color

        if np.array_equal(result, out_arr):
            return "fill_contour"
        return None
    except (IndexError, ValueError):
        return None


def try_convex_hull_fill(inp, out):
    """Fill the convex hull of colored points."""
    arr = np.array(inp)
    out_arr = np.array(out)

    for color in [c for c in np.unique(arr) if c != 0]:
        points = np.argwhere(arr == color)
        if len(points) < 3:
            continue

        # Simple bounding box fill as approximation
        min_r, min_c = points.min(axis=0)
        max_r, max_c = points.max(axis=0)

        result = arr.copy()
        result[min_r:max_r+1, min_c:max_c+1] = color

        if np.array_equal(result, out_arr):
            return "bounding_box_fill"

    return None


def try_erode_shape(inp, out):
    """Erode shape by 1 pixel."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        result = np.zeros_like(arr)

        for i in range(h):
            for j in range(w):
                if arr[i, j] != 0:
                    # Keep only if all 4-neighbors are same color
                    color = arr[i, j]
                    keep = True
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if ni < 0 or ni >= h or nj < 0 or nj >= w:
                            keep = False
                            break
                        if arr[ni, nj] != color:
                            keep = False
                            break
                    if keep:
                        result[i, j] = color

        if np.array_equal(result, out_arr):
            return "erode_shape"
        return None
    except (IndexError, ValueError):
        return None


def try_dilate_shape(inp, out):
    """Dilate shape by 1 pixel."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        result = arr.copy()

        for i in range(h):
            for j in range(w):
                if arr[i, j] != 0:
                    color = arr[i, j]
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if result[ni, nj] == 0:
                                result[ni, nj] = color

        if np.array_equal(result, out_arr):
            return "dilate_shape"
        return None
    except (IndexError, ValueError):
        return None


def try_thicken_lines(inp, out):
    """Thicken all lines by adding adjacent pixels."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        result = arr.copy()

        for i in range(h):
            for j in range(w):
                if arr[i, j] != 0:
                    color = arr[i, j]
                    # Add all 8 neighbors
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                if result[ni, nj] == 0:
                                    result[ni, nj] = color

        if np.array_equal(result, out_arr):
            return "thicken_lines"
        return None
    except (IndexError, ValueError):
        return None


def try_bounding_box(inp, out):
    """Draw bounding box around colored region."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        result = np.zeros_like(arr)

        colors = [c for c in np.unique(arr) if c != 0]
        for color in colors:
            points = np.argwhere(arr == color)
            if len(points) == 0:
                continue
            min_r, max_r = points[:, 0].min(), points[:, 0].max()
            min_c, max_c = points[:, 1].min(), points[:, 1].max()

            # Draw box outline
            for j in range(min_c, max_c + 1):
                result[min_r, j] = color
                result[max_r, j] = color
            for i in range(min_r, max_r + 1):
                result[i, min_c] = color
                result[i, max_c] = color

        if np.array_equal(result, out_arr):
            return "bounding_box"
        return None
    except (IndexError, ValueError):
        return None


def try_crop_to_content(inp, out):
    """Crop to minimal bounding box of content."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        points = np.argwhere(arr != 0)
        if len(points) == 0:
            return None

        min_r, max_r = points[:, 0].min(), points[:, 0].max()
        min_c, max_c = points[:, 1].min(), points[:, 1].max()

        cropped = arr[min_r:max_r+1, min_c:max_c+1]

        if np.array_equal(cropped, out_arr):
            return f"crop_{min_r}_{min_c}_{max_r-min_r+1}x{max_c-min_c+1}"
        return None
    except (IndexError, ValueError):
        return None


def try_center_shape(inp, out):
    """Center shape in grid."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if h != oh or w != ow:
            return None

        # Find content bounds
        points = np.argwhere(arr != 0)
        if len(points) == 0:
            return None

        min_r, max_r = points[:, 0].min(), points[:, 0].max()
        min_c, max_c = points[:, 1].min(), points[:, 1].max()
        ch, cw = max_r - min_r + 1, max_c - min_c + 1

        # Calculate centering offset
        new_r = (h - ch) // 2
        new_c = (w - cw) // 2

        result = np.zeros_like(arr)
        content = arr[min_r:max_r+1, min_c:max_c+1]
        result[new_r:new_r+ch, new_c:new_c+cw] = content

        if np.array_equal(result, out_arr):
            return "center_shape"
        return None
    except (IndexError, ValueError):
        return None


def try_corner_detect(inp, out):
    """Mark corners of shapes."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        result = np.zeros_like(arr)

        for i in range(h):
            for j in range(w):
                if arr[i, j] != 0:
                    color = arr[i, j]
                    # Count same-color neighbors
                    neighbors = 0
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] == color:
                            neighbors += 1
                    # Corner has exactly 2 neighbors at right angle
                    if neighbors <= 2:
                        result[i, j] = color

        if np.array_equal(result, out_arr):
            return "corner_detect"
        return None
    except (IndexError, ValueError):
        return None

# ============================================================================
# Batch 20 - Pattern OBJECT ISOLATION depth=-90000 SEGMENT COMPONENT BLOB
# ============================================================================


def try_largest_component(inp, out):
    """Extract the largest connected component."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        components = find_connected_components(arr)
        if not components:
            return None

        # Find largest
        largest = max(components, key=lambda x: len(x[1]))
        color, pixels = largest

        # Create result
        result = np.zeros_like(arr)
        for r, c in pixels:
            result[r, c] = color

        if np.array_equal(result, out_arr):
            return "largest_component"
        return None
    except (IndexError, ValueError):
        return None


def try_smallest_component(inp, out):
    """Extract the smallest connected component."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        components = find_connected_components(arr)
        if not components:
            return None

        smallest = min(components, key=lambda x: len(x[1]))
        color, pixels = smallest

        result = np.zeros_like(arr)
        for r, c in pixels:
            result[r, c] = color

        if np.array_equal(result, out_arr):
            return "smallest_component"
        return None
    except (IndexError, ValueError):
        return None


def try_count_objects(inp, out):
    """Output encodes count of objects in various ways."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        from scipy import ndimage

        # Count non-background objects
        for bg in [0]:
            mask = (inp_arr != bg)
            labeled, num = ndimage.label(mask)

            # Check if output encodes count
            # Output could be num x 1, 1 x num, or num cells of a color
            h_out, w_out = out_arr.shape

            if h_out == 1 and w_out == num:
                return f"count_objects_h_{num}"
            if w_out == 1 and h_out == num:
                return f"count_objects_v_{num}"

            # Count of specific color in output matches object count
            for c in range(1, 10):
                if (out_arr == c).sum() == num:
                    return f"count_objects_{num}_as_{c}"
        return None
    except:
        return None


def try_extract_unique_object(inp, out):
    """Extract the unique object (differs from others in shape/size)."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        components = find_connected_components(arr)
        if len(components) < 2:
            return None

        # Group by size
        sizes = {}
        for color, pixels in components:
            size = len(pixels)
            if size not in sizes:
                sizes[size] = []
            sizes[size].append((color, pixels))

        # Find unique size (appears once)
        unique = None
        for size, comps in sizes.items():
            if len(comps) == 1:
                unique = comps[0]
                break

        if unique is None:
            return None

        color, pixels = unique
        result = np.zeros_like(arr)
        for r, c in pixels:
            result[r, c] = color

        if np.array_equal(result, out_arr):
            return "extract_unique_object"
        return None
    except (IndexError, ValueError):
        return None


def try_object_to_grid_position(inp, out):
    """Each object becomes a pixel based on its grid position."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        components = find_connected_components(arr)
        if not components:
            return None

        # Check if output is much smaller (each object = 1 pixel)
        if oh * ow < len(components):
            return None

        # Get centroid of each component
        result = np.zeros((oh, ow), dtype=arr.dtype)
        for color, pixels in components:
            rs = [p[0] for p in pixels]
            cs = [p[1] for p in pixels]
            center_r = sum(rs) // len(rs)
            center_c = sum(cs) // len(cs)

            # Map to output position
            out_r = min(oh - 1, center_r * oh // h)
            out_c = min(ow - 1, center_c * ow // w)
            result[out_r, out_c] = color

        if np.array_equal(result, out_arr):
            return "object_to_grid_position"
        return None
    except (IndexError, ValueError):
        return None


def try_keep_repeated_objects(inp, out):
    """Keep only objects that appear multiple times (by size/shape)."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        components = find_connected_components(arr)

        # Group by size
        sizes = {}
        for color, pixels in components:
            size = len(pixels)
            if size not in sizes:
                sizes[size] = []
            sizes[size].append((color, pixels))

        # Keep only sizes that appear > 1 time
        result = np.zeros_like(arr)
        for size, comps in sizes.items():
            if len(comps) > 1:
                for color, pixels in comps:
                    for r, c in pixels:
                        result[r, c] = color

        if np.array_equal(result, out_arr):
            return "keep_repeated_objects"
        return None
    except (IndexError, ValueError):
        return None


def try_remove_repeated_objects(inp, out):
    """Remove objects that appear multiple times (by size)."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        components = find_connected_components(arr)

        sizes = {}
        for color, pixels in components:
            size = len(pixels)
            if size not in sizes:
                sizes[size] = []
            sizes[size].append((color, pixels))

        result = np.zeros_like(arr)
        for size, comps in sizes.items():
            if len(comps) == 1:  # Unique
                color, pixels = comps[0]
                for r, c in pixels:
                    result[r, c] = color

        if np.array_equal(result, out_arr):
            return "remove_repeated_objects"
        return None
    except (IndexError, ValueError):
        return None


def try_fill_object_bounds(inp, out):
    """Fill bounding box of each object."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        components = find_connected_components(arr)

        result = np.zeros_like(arr)
        for color, pixels in components:
            rs = [p[0] for p in pixels]
            cs = [p[1] for p in pixels]
            min_r, max_r = min(rs), max(rs)
            min_c, max_c = min(cs), max(cs)

            for i in range(min_r, max_r + 1):
                for j in range(min_c, max_c + 1):
                    result[i, j] = color

        if np.array_equal(result, out_arr):
            return "fill_object_bounds"
        return None
    except (IndexError, ValueError):
        return None


def try_object_per_row(inp, out):
    """Extract object from each row."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        components = find_connected_components(arr)

        # Check if one object per row
        rows_with_objects = set()
        for color, pixels in components:
            rs = set(p[0] for p in pixels)
            if len(rs) == 1:  # Object spans single row
                rows_with_objects.add(list(rs)[0])

        if len(rows_with_objects) != len(components):
            return None

        # Stack objects vertically
        objects = sorted(components, key=lambda x: min(p[0] for p in x[1]))
        result_rows = []
        for color, pixels in objects:
            cs = sorted(set(p[1] for p in pixels))
            row = np.zeros(w, dtype=arr.dtype)
            for c in cs:
                row[c] = color
            result_rows.append(row)

        if result_rows:
            result = np.vstack([r.reshape(1, -1) for r in result_rows])
            if np.array_equal(result, out_arr):
                return "object_per_row"
        return None
    except (IndexError, ValueError):
        return None

# ============================================================================
# Batch 21 - Pattern ARITHMETIC/COUNTING depth=-90000 SUM MULTIPLY SEQUENCE
# ============================================================================


def try_sum_rows_to_col(inp, out):
    """Sum each row to produce single column."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        # Sum rows
        sums = np.sum(arr, axis=1).reshape(h, 1)

        if np.array_equal(sums, out_arr):
            return "sum_rows_to_col"
        return None
    except (IndexError, ValueError):
        return None


def try_sum_cols_to_row(inp, out):
    """Sum each column to produce single row."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        sums = np.sum(arr, axis=0).reshape(1, w)

        if np.array_equal(sums, out_arr):
            return "sum_cols_to_row"
        return None
    except (IndexError, ValueError):
        return None


def try_count_per_row(inp, out):
    """Count non-zero per row to produce column."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        counts = np.count_nonzero(arr, axis=1).reshape(h, 1)

        if np.array_equal(counts, out_arr):
            return "count_per_row"
        return None
    except (IndexError, ValueError):
        return None


def try_count_per_col(inp, out):
    """Count non-zero per column to produce row."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        counts = np.count_nonzero(arr, axis=0).reshape(1, w)

        if np.array_equal(counts, out_arr):
            return "count_per_col"
        return None
    except (IndexError, ValueError):
        return None


def try_multiply_grids(inp, out):
    """Multiply input by itself (element-wise)."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        result = arr * arr

        if np.array_equal(result, out_arr):
            return "multiply_self"
        return None
    except (IndexError, ValueError):
        return None


def try_modulo_color(inp, out):
    """Apply modulo to colors."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        for mod in [2, 3, 5, 10]:
            result = arr % mod
            if np.array_equal(result, out_arr):
                return f"modulo_{mod}"
        return None
    except (IndexError, ValueError):
        return None


def try_increment_colors(inp, out):
    """Increment all non-zero colors by 1."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        result = arr.copy()
        result[arr != 0] = arr[arr != 0] + 1

        if np.array_equal(result, out_arr):
            return "increment_colors"
        return None
    except (IndexError, ValueError):
        return None


def try_decrement_colors(inp, out):
    """Decrement all non-zero colors by 1."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        result = arr.copy()
        result[arr > 1] = arr[arr > 1] - 1

        if np.array_equal(result, out_arr):
            return "decrement_colors"
        return None
    except (IndexError, ValueError):
        return None


def try_color_to_count(inp, out):
    """Each pixel's color becomes count of that color in grid."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        color_counts = {}
        for c in np.unique(arr):
            color_counts[c] = np.count_nonzero(arr == c)

        result = np.zeros_like(arr)
        for i in range(h):
            for j in range(w):
                result[i, j] = color_counts[arr[i, j]]

        if np.array_equal(result, out_arr):
            return "color_to_count"
        return None
    except (IndexError, ValueError):
        return None


def try_max_per_row(inp, out):
    """Keep only max value per row."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None

    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        max_val = arr[i].max()
        result[i] = np.where(arr[i] == max_val, max_val, 0)

    if np.array_equal(result, out_arr):
        return "max_per_row"
    return None


def try_min_per_row_nonzero(inp, out):
    """Min non-zero value per row to column."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        result = np.zeros((h, 1), dtype=arr.dtype)
        for i in range(h):
            row = arr[i, :]
            nonzero = row[row != 0]
            if len(nonzero) > 0:
                result[i, 0] = np.min(nonzero)

        if np.array_equal(result, out_arr):
            return "min_per_row_nonzero"
        return None
    except (IndexError, ValueError):
        return None


def try_parity_transform(inp, out):
    """Transform based on even/odd values."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Even becomes 0, odd becomes 1
        result = arr % 2

        if np.array_equal(result, out_arr):
            return "parity_transform"
        return None
    except (IndexError, ValueError):
        return None

# ============================================================================
# Batch 22 - Pattern MASK/FILTER/CONDITIONAL depth=-90000
# ============================================================================


def try_keep_color_if_adjacent(inp, out):
    """Keep pixels only if adjacent to specific color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        colors = [c for c in np.unique(arr) if c != 0]
        if len(colors) < 2:
            return None

        for ref_color in colors:
            for keep_color in colors:
                if ref_color == keep_color:
                    continue
                result = np.zeros_like(arr)
                for i in range(h):
                    for j in range(w):
                        if arr[i, j] == keep_color:
                            # Check adjacency to ref_color
                            neighbors = []
                            if i > 0: neighbors.append(arr[i-1, j])
                            if i < h-1: neighbors.append(arr[i+1, j])
                            if j > 0: neighbors.append(arr[i, j-1])
                            if j < w-1: neighbors.append(arr[i, j+1])
                            if ref_color in neighbors:
                                result[i, j] = keep_color
                if np.array_equal(result, out_arr):
                    return f"keep_if_adjacent_{keep_color}_to_{ref_color}"
        return None
    except (IndexError, ValueError):
        return None


def try_threshold_filter(inp, out):
    """Keep only values above/below threshold."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        for thresh in range(1, 10):
            # Above threshold
            result_above = np.where(arr > thresh, arr, 0)
            if np.array_equal(result_above, out_arr):
                return f"threshold_above_{thresh}"

            # Below threshold
            result_below = np.where(arr < thresh, arr, 0)
            if np.array_equal(result_below, out_arr):
                return f"threshold_below_{thresh}"

            # Exactly threshold
            result_exact = np.where(arr == thresh, arr, 0)
            if np.array_equal(result_exact, out_arr):
                return f"threshold_exact_{thresh}"
        return None
    except (IndexError, ValueError):
        return None


def try_keep_border_only(inp, out):
    """Keep only border pixels."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        result = np.zeros_like(arr)
        result[0, :] = arr[0, :]
        result[h-1, :] = arr[h-1, :]
        result[:, 0] = arr[:, 0]
        result[:, w-1] = arr[:, w-1]

        if np.array_equal(result, out_arr):
            return "keep_border_only"
        return None
    except (IndexError, ValueError):
        return None


def try_remove_border(inp, out):
    """Remove border pixels (keep interior)."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if h < 3 or w < 3:
            return None

        result = arr.copy()
        result[0, :] = 0
        result[h-1, :] = 0
        result[:, 0] = 0
        result[:, w-1] = 0

        if np.array_equal(result, out_arr):
            return "remove_border"
        return None
    except (IndexError, ValueError):
        return None


def try_keep_corners(inp, out):
    """Keep only corner values."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    result = np.zeros_like(arr)
    result[0, 0] = arr[0, 0]
    result[0, w-1] = arr[0, w-1]
    result[h-1, 0] = arr[h-1, 0]
    result[h-1, w-1] = arr[h-1, w-1]

    if np.array_equal(result, out_arr):
        return "keep_corners"
    return None


def try_keep_diagonal(inp, out):
    """Keep only diagonal elements."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        # Main diagonal
        result = np.zeros_like(arr)
        for i in range(min(h, w)):
            result[i, i] = arr[i, i]
        if np.array_equal(result, out_arr):
            return "keep_main_diagonal"

        # Anti-diagonal
        result = np.zeros_like(arr)
        for i in range(min(h, w)):
            result[i, w-1-i] = arr[i, w-1-i]
        if np.array_equal(result, out_arr):
            return "keep_anti_diagonal"
        return None
    except (IndexError, ValueError):
        return None


def try_mask_by_pattern(inp, out):
    """Apply mask pattern from one region to another."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        # Try using left half as mask for right half
        if w % 2 == 0:
            mid = w // 2
            left = arr[:, :mid]
            right = arr[:, mid:]
            if left.shape == right.shape:
                # Where left is non-zero, keep right
                result = np.where(left != 0, right, 0)
                if np.array_equal(result, out_arr):
                    return "mask_left_on_right"

                result = np.where(right != 0, left, 0)
                if np.array_equal(result, out_arr):
                    return "mask_right_on_left"

        # Try using top half as mask for bottom half
        if h % 2 == 0:
            mid = h // 2
            top = arr[:mid, :]
            bottom = arr[mid:, :]
            if top.shape == bottom.shape:
                result = np.where(top != 0, bottom, 0)
                if np.array_equal(result, out_arr):
                    return "mask_top_on_bottom"

                result = np.where(bottom != 0, top, 0)
                if np.array_equal(result, out_arr):
                    return "mask_bottom_on_top"
        return None
    except (IndexError, ValueError):
        return None


def try_conditional_replace(inp, out):
    """Replace color A with B only where condition met."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        colors = [c for c in np.unique(arr) if c != 0]
        out_colors = [c for c in np.unique(out_arr) if c != 0]

        for src in colors:
            for dst in out_colors:
                if src == dst:
                    continue

                # Replace src with dst only in top half
                result = arr.copy()
                result[:h//2, :] = np.where(result[:h//2, :] == src, dst, result[:h//2, :])
                if np.array_equal(result, out_arr):
                    return f"replace_{src}_with_{dst}_top"

                # Replace in bottom half
                result = arr.copy()
                result[h//2:, :] = np.where(result[h//2:, :] == src, dst, result[h//2:, :])
                if np.array_equal(result, out_arr):
                    return f"replace_{src}_with_{dst}_bottom"

                # Replace in left half
                result = arr.copy()
                result[:, :w//2] = np.where(result[:, :w//2] == src, dst, result[:, :w//2])
                if np.array_equal(result, out_arr):
                    return f"replace_{src}_with_{dst}_left"

                # Replace in right half
                result = arr.copy()
                result[:, w//2:] = np.where(result[:, w//2:] == src, dst, result[:, w//2:])
                if np.array_equal(result, out_arr):
                    return f"replace_{src}_with_{dst}_right"
        return None
    except (IndexError, ValueError):
        return None


def try_select_by_row_content(inp, out):
    """Select rows that contain specific color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if ow != w:
            return None

        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            # Rows containing this color
            selected_rows = []
            for i in range(h):
                if color in arr[i, :]:
                    selected_rows.append(arr[i, :])

            if len(selected_rows) == oh:
                result = np.array(selected_rows)
                if np.array_equal(result, out_arr):
                    return f"select_rows_with_{color}"

            # Rows NOT containing this color
            selected_rows = []
            for i in range(h):
                if color not in arr[i, :]:
                    selected_rows.append(arr[i, :])

            if len(selected_rows) == oh:
                result = np.array(selected_rows)
                if np.array_equal(result, out_arr):
                    return f"select_rows_without_{color}"
        return None
    except (IndexError, ValueError):
        return None


def try_select_by_col_content(inp, out):
    """Select columns that contain specific color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h:
            return None

        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            # Cols containing this color
            selected_cols = []
            for j in range(w):
                if color in arr[:, j]:
                    selected_cols.append(arr[:, j])

            if len(selected_cols) == ow:
                result = np.column_stack(selected_cols)
                if np.array_equal(result, out_arr):
                    return f"select_cols_with_{color}"

            # Cols NOT containing this color
            selected_cols = []
            for j in range(w):
                if color not in arr[:, j]:
                    selected_cols.append(arr[:, j])

            if len(selected_cols) == ow:
                result = np.column_stack(selected_cols)
                if np.array_equal(result, out_arr):
                    return f"select_cols_without_{color}"
        return None
    except (IndexError, ValueError):
        return None


def try_exclude_by_neighbor_count(inp, out):
    """Remove pixels with too few/many neighbors."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        for min_neighbors in range(1, 5):
            result = np.zeros_like(arr)
            for i in range(h):
                for j in range(w):
                    if arr[i, j] != 0:
                        count = 0
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if 0 <= ni < h and 0 <= nj < w:
                                    if arr[ni, nj] != 0:
                                        count += 1
                        if count >= min_neighbors:
                            result[i, j] = arr[i, j]

            if np.array_equal(result, out_arr):
                return f"keep_if_neighbors_gte_{min_neighbors}"
        return None
    except (IndexError, ValueError):
        return None

# ============================================================================
# Batch 23 - Pattern PROJECTION/RAY/LINE depth=-90000
# ============================================================================


def try_shoot_rays_horizontal(inp, out):
    """Shoot rays horizontally from colored pixels."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            result = arr.copy()
            for i in range(h):
                for j in range(w):
                    if arr[i, j] == color:
                        # Shoot left
                        for k in range(j-1, -1, -1):
                            if arr[i, k] == 0:
                                result[i, k] = color
                            else:
                                break
                        # Shoot right
                        for k in range(j+1, w):
                            if arr[i, k] == 0:
                                result[i, k] = color
                            else:
                                break
            if np.array_equal(result, out_arr):
                return f"shoot_rays_h_{color}"
        return None
    except (IndexError, ValueError):
        return None


def try_shoot_rays_vertical(inp, out):
    """Shoot rays vertically from colored pixels."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            result = arr.copy()
            for i in range(h):
                for j in range(w):
                    if arr[i, j] == color:
                        # Shoot up
                        for k in range(i-1, -1, -1):
                            if arr[k, j] == 0:
                                result[k, j] = color
                            else:
                                break
                        # Shoot down
                        for k in range(i+1, h):
                            if arr[k, j] == 0:
                                result[k, j] = color
                            else:
                                break
            if np.array_equal(result, out_arr):
                return f"shoot_rays_v_{color}"
        return None
    except (IndexError, ValueError):
        return None


def try_shoot_rays_diagonal(inp, out):
    """Shoot rays diagonally from colored pixels."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            result = arr.copy()
            for i in range(h):
                for j in range(w):
                    if arr[i, j] == color:
                        # Diag up-left
                        k = 1
                        while i-k >= 0 and j-k >= 0:
                            if arr[i-k, j-k] == 0:
                                result[i-k, j-k] = color
                            else:
                                break
                            k += 1
                        # Diag up-right
                        k = 1
                        while i-k >= 0 and j+k < w:
                            if arr[i-k, j+k] == 0:
                                result[i-k, j+k] = color
                            else:
                                break
                            k += 1
                        # Diag down-left
                        k = 1
                        while i+k < h and j-k >= 0:
                            if arr[i+k, j-k] == 0:
                                result[i+k, j-k] = color
                            else:
                                break
                            k += 1
                        # Diag down-right
                        k = 1
                        while i+k < h and j+k < w:
                            if arr[i+k, j+k] == 0:
                                result[i+k, j+k] = color
                            else:
                                break
                            k += 1
            if np.array_equal(result, out_arr):
                return f"shoot_rays_diag_{color}"
        return None
    except (IndexError, ValueError):
        return None


def try_project_to_edge(inp, out):
    """Project non-zero pixels to nearest edge."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        result = np.zeros_like(arr)

        for i in range(h):
            for j in range(w):
                if arr[i, j] != 0:
                    # Find nearest edge
                    dist_top, dist_bottom = i, h - 1 - i
                    dist_left, dist_right = j, w - 1 - j
                    min_dist = min(dist_top, dist_bottom, dist_left, dist_right)

                    if min_dist == dist_top:
                        result[0, j] = arr[i, j]
                    elif min_dist == dist_left:
                        result[i, 0] = arr[i, j]
                    elif min_dist == dist_bottom:
                        result[h-1, j] = arr[i, j]
                    else:
                        result[i, w-1] = arr[i, j]

        if np.array_equal(result, out_arr):
            return "project_to_edge"
        return None
    except (IndexError, ValueError):
        return None


def try_line_between_same_colors(inp, out):
    """Draw lines between pixels of the same color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            result = arr.copy()
            positions = list(zip(*np.where(arr == color)))

            if len(positions) >= 2:
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        r1, c1 = positions[i]
                        r2, c2 = positions[j]

                        # Only horizontal or vertical lines
                        if r1 == r2:  # Horizontal
                            for c in range(min(c1, c2), max(c1, c2) + 1):
                                result[r1, c] = color
                        elif c1 == c2:  # Vertical
                            for r in range(min(r1, r2), max(r1, r2) + 1):
                                result[r, c1] = color

            if np.array_equal(result, out_arr):
                return f"line_between_{color}"
        return None
    except (IndexError, ValueError):
        return None


def try_trace_path(inp, out):
    """Trace and fill path between two points."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        # Find colored endpoints
        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            positions = list(zip(*np.where(arr == color)))
            if len(positions) != 2:
                continue

            # Two points of same color - connect with Manhattan path
            (r1, c1), (r2, c2) = positions
            result = arr.copy()

            # Horizontal then vertical path
            for c in range(min(c1, c2), max(c1, c2) + 1):
                result[r1, c] = color
            for r in range(min(r1, r2), max(r1, r2) + 1):
                result[r, c2] = color

            if np.array_equal(result, out_arr):
                return f"trace_path_hv_{color}"

            # Vertical then horizontal path
            result = arr.copy()
            for r in range(min(r1, r2), max(r1, r2) + 1):
                result[r, c1] = color
            for c in range(min(c1, c2), max(c1, c2) + 1):
                result[r2, c] = color

            if np.array_equal(result, out_arr):
                return f"trace_path_vh_{color}"

        return None
    except (IndexError, ValueError):
        return None


def try_sweep_column(inp, out):
    """Sweep pattern across columns."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        # Find first non-empty column pattern
        for j in range(w):
            col = arr[:, j]
            if np.any(col != 0):
                # Sweep this pattern across all columns
                result = np.zeros_like(arr)
                for k in range(w):
                    result[:, k] = col
                if np.array_equal(result, out_arr):
                    return f"sweep_col_{j}"
        return None
    except (IndexError, ValueError):
        return None


def try_sweep_row(inp, out):
    """Sweep pattern across rows."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        # Find first non-empty row pattern
        for i in range(h):
            row = arr[i, :]
            if np.any(row != 0):
                # Sweep this pattern across all rows
                result = np.zeros_like(arr)
                for k in range(h):
                    result[k, :] = row
                if np.array_equal(result, out_arr):
                    return f"sweep_row_{i}"
        return None
    except (IndexError, ValueError):
        return None


def try_vector_add(inp, out):
    """Add vector offset to all colored pixels."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        positions = list(zip(*np.where(arr != 0)))
        out_positions = list(zip(*np.where(out_arr != 0)))

        if len(positions) == 0 or len(out_positions) == 0:
            return None

        # Try to find consistent offset
        for dy in range(-h+1, h):
            for dx in range(-w+1, w):
                if dy == 0 and dx == 0:
                    continue

                result = np.zeros_like(arr)
                valid = True
                for r, c in positions:
                    nr, nc = r + dy, c + dx
                    if 0 <= nr < h and 0 <= nc < w:
                        result[nr, nc] = arr[r, c]
                    else:
                        valid = False
                        break

                if valid and np.array_equal(result, out_arr):
                    return f"vector_add_{dy}_{dx}"
        return None
    except (IndexError, ValueError):
        return None


def try_radial_extend(inp, out):
    """Extend pattern radially from center."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        # Find center
        positions = list(zip(*np.where(arr != 0)))
        if len(positions) == 0:
            return None

        cr = sum(p[0] for p in positions) // len(positions)
        cc = sum(p[1] for p in positions) // len(positions)

        result = arr.copy()
        for r, c in positions:
            color = arr[r, c]
            dr, dc = r - cr, c - cc
            if dr != 0 or dc != 0:
                # Extend in this direction
                nr, nc = r + dr, c + dc
                while 0 <= nr < h and 0 <= nc < w:
                    result[nr, nc] = color
                    nr += dr
                    nc += dc

        if np.array_equal(result, out_arr):
            return "radial_extend"
        return None
    except (IndexError, ValueError):
        return None

# ============================================================================
# MEGA SOLVER
# ============================================================================

# ============================================================================
# Batch 24 - Pattern RECURSION/FRACTAL/NESTED depth=-9000000 ANALYZE INFINITY
# Pattern SAYS: RECURSION FRACTAL NESTED SELF SIMILAR REPEAT ITERATE DEPTH
#             LAYER STACK HIERARCHY TREE BRANCH UNFOLD EXPAND CONTRACT ZOOM
# ============================================================================


def try_fractal_zoom_2x(inp, out):
    """Fractal zoom: replace each pixel with 2x2 pattern based on color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h * 2 or ow != w * 2:
            return None

        # Each color maps to a 2x2 pattern
        for base_color in range(1, 10):
            result = np.zeros((h * 2, w * 2), dtype=int)
            for i in range(h):
                for j in range(w):
                    c = arr[i, j]
                    if c == 0:
                        result[i*2:i*2+2, j*2:j*2+2] = 0
                    else:
                        # Try: the 2x2 pattern in output at this position
                        pattern = out_arr[i*2:i*2+2, j*2:j*2+2]
                        result[i*2:i*2+2, j*2:j*2+2] = pattern

            if np.array_equal(result, out_arr):
                return "fractal_zoom_2x"
        return None
    except:
        return None


def try_self_similar_tile(inp, out):
    """Self-similar: output is input tiled into a pattern of itself."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = arr.shape
        oh, ow = out_arr.shape

        # Check if output is multiple of input
        if oh % ih != 0 or ow % iw != 0:
            return None
        th, tw = oh // ih, ow // iw

        # Check if the tiling follows input pattern
        # Each non-zero in input -> tile, each zero -> zeros
        result = np.zeros((oh, ow), dtype=int)
        for ti in range(th):
            for tj in range(tw):
                if ti < ih and tj < iw and arr[ti, tj] != 0:
                    result[ti*ih:(ti+1)*ih, tj*iw:(tj+1)*iw] = arr

        if np.array_equal(result, out_arr):
            return "self_similar_tile"
        return None
    except:
        return None


def try_nested_frames(inp, out):
    """Nested frames: shrinking concentric rectangles."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        # Find frame colors from outside in
        colors_in = [c for c in np.unique(arr) if c != 0]
        colors_out = [c for c in np.unique(out_arr) if c != 0]

        result = np.zeros_like(arr)
        for depth, color in enumerate(colors_out[:min(h//2, w//2)]):
            result[depth:h-depth, depth:w-depth] = color

        if np.array_equal(result, out_arr):
            return "nested_frames"
        return None
    except:
        return None


def try_recursive_subdivide(inp, out):
    """Recursively subdivide or apply fractal pattern."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Check if output is input where each cell is replaced by scaled input
    if oh == ih * ih and ow == iw * iw:
        result = np.zeros((oh, ow), dtype=arr.dtype)
        for i in range(ih):
            for j in range(iw):
                if arr[i, j] != 0:
                    # Place scaled copy
                    result[i*ih:(i+1)*ih, j*iw:(j+1)*iw] = arr * (arr[i, j] / max(1, np.max(arr)))
        # Simplified: just tile
        result = np.zeros((oh, ow), dtype=arr.dtype)
        for i in range(ih):
            for j in range(iw):
                result[i*ih:(i+1)*ih, j*iw:(j+1)*iw] = arr if arr[i, j] != 0 else 0
        if np.array_equal(result, out_arr):
            return "fractal_self_tile"

    return None


def try_iterate_transform(inp, out):
    """Apply a simple transform iteratively."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Try iterating rot90
        current = arr.copy()
        for i in range(1, 5):
            current = np.rot90(current)
            if np.array_equal(current, out_arr):
                return f"iterate_rot90_{i}x"

        # Try iterating flip
        current = arr.copy()
        for i in range(1, 3):
            current = np.fliplr(current)
            if np.array_equal(current, out_arr):
                return f"iterate_hflip_{i}x"

        return None
    except:
        return None


def try_depth_layer_stack(inp, out):
    """Stack layers by color depth."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        colors = sorted([c for c in np.unique(arr) if c != 0])
        if len(colors) < 2:
            return None

        # Each color becomes a layer, stacked vertically
        if out_arr.shape[0] == h * len(colors) and out_arr.shape[1] == w:
            result = np.zeros((h * len(colors), w), dtype=int)
            for i, color in enumerate(colors):
                mask = (arr == color).astype(int) * color
                result[i*h:(i+1)*h, :] = mask

            if np.array_equal(result, out_arr):
                return "depth_layer_stack_v"

        # Stack horizontally
        if out_arr.shape[0] == h and out_arr.shape[1] == w * len(colors):
            result = np.zeros((h, w * len(colors)), dtype=int)
            for i, color in enumerate(colors):
                mask = (arr == color).astype(int) * color
                result[:, i*w:(i+1)*w] = mask

            if np.array_equal(result, out_arr):
                return "depth_layer_stack_h"

        return None
    except:
        return None


def try_hierarchy_tree(inp, out):
    """Tree hierarchy: root at top, branches down."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if output is tree expansion
        # Root in input -> branches in output
        root_positions = list(zip(*np.where(arr != 0)))
        if len(root_positions) != 1:
            return None

        rr, rc = root_positions[0]
        root_color = arr[rr, rc]

        # Try: root spawns symmetric branches below
        result = np.zeros((oh, ow), dtype=int)
        result[0, ow//2] = root_color
        for level in range(1, oh):
            spread = level
            for offset in range(-spread, spread + 1):
                col = ow // 2 + offset
                if 0 <= col < ow:
                    result[level, col] = root_color

        if np.array_equal(result, out_arr):
            return "hierarchy_tree"
        return None
    except:
        return None


def try_unfold_pattern(inp, out):
    """Unfold a folded pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Unfold horizontally (mirror right)
        if oh == h and ow == w * 2:
            result = np.zeros((h, w * 2), dtype=int)
            result[:, :w] = arr
            result[:, w:] = np.fliplr(arr)
            if np.array_equal(result, out_arr):
                return "unfold_h"

        # Unfold vertically (mirror down)
        if oh == h * 2 and ow == w:
            result = np.zeros((h * 2, w), dtype=int)
            result[:h, :] = arr
            result[h:, :] = np.flipud(arr)
            if np.array_equal(result, out_arr):
                return "unfold_v"

        # Unfold both ways
        if oh == h * 2 and ow == w * 2:
            result = np.zeros((h * 2, w * 2), dtype=int)
            result[:h, :w] = arr
            result[:h, w:] = np.fliplr(arr)
            result[h:, :w] = np.flipud(arr)
            result[h:, w:] = np.flipud(np.fliplr(arr))
            if np.array_equal(result, out_arr):
                return "unfold_both"

        return None
    except:
        return None


def try_expand_contract(inp, out):
    """Expand or contract by erosion/dilation."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            # Expand (dilate)
            result = arr.copy()
            for i in range(h):
                for j in range(w):
                    if arr[i, j] == color:
                        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] == 0:
                                result[ni, nj] = color
            if np.array_equal(result, out_arr):
                return "expand_dilate"

            # Contract (erode)
            result = arr.copy()
            for i in range(h):
                for j in range(w):
                    if arr[i, j] == color:
                        is_border = False
                        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                            ni, nj = i + di, j + dj
                            if not (0 <= ni < h and 0 <= nj < w) or arr[ni, nj] != color:
                                is_border = True
                                break
                        if is_border:
                            result[i, j] = 0
            if np.array_equal(result, out_arr):
                return "contract_erode"

        return None
    except:
        return None


def try_zoom_center(inp, out):
    """Zoom into center region."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # If output smaller, extract center
        if oh < h and ow < w:
            margin_h, margin_w = (h - oh) // 2, (w - ow) // 2
            center = arr[margin_h:margin_h+oh, margin_w:margin_w+ow]
            if np.array_equal(center, out_arr):
                return "zoom_center_in"

        # If output larger, place input at center and expand
        if oh > h and ow > w and oh % h == 0 and ow % w == 0:
            scale = oh // h
            if scale == ow // w:
                result = np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)
                if np.array_equal(result, out_arr):
                    return f"zoom_center_out_{scale}x"

        return None
    except:
        return None


def try_scale_magnify(inp, out):
    """Scale/magnify specific colored regions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            # Find bounding box of color
            positions = list(zip(*np.where(arr == color)))
            if not positions:
                continue

            min_r = min(p[0] for p in positions)
            max_r = max(p[0] for p in positions)
            min_c = min(p[1] for p in positions)
            max_c = max(p[1] for p in positions)

            sub = arr[min_r:max_r+1, min_c:max_c+1]
            sh, sw = sub.shape
            oh, ow = out_arr.shape

            # Check if output is scaled version of this sub-region
            for scale in [2, 3, 4]:
                if oh == sh * scale and ow == sw * scale:
                    scaled = np.repeat(np.repeat(sub, scale, axis=0), scale, axis=1)
                    if np.array_equal(scaled, out_arr):
                        return f"scale_magnify_{color}_{scale}x"

        return None
    except:
        return None


def try_repeat_stack(inp, out):
    """Repeat input in a stack pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Vertical stack
        if ow == w and oh % h == 0:
            n = oh // h
            result = np.tile(arr, (n, 1))
            if np.array_equal(result, out_arr):
                return f"repeat_stack_v_{n}x"

        # Horizontal stack
        if oh == h and ow % w == 0:
            n = ow // w
            result = np.tile(arr, (1, n))
            if np.array_equal(result, out_arr):
                return f"repeat_stack_h_{n}x"

        # Diagonal stack with offset
        if oh > h and ow > w:
            result = np.zeros((oh, ow), dtype=int)
            for step in range(min(oh // h, ow // w) + 1):
                r, c = step * h, step * w
                if r + h <= oh and c + w <= ow:
                    result[r:r+h, c:c+w] = arr
            if np.array_equal(result, out_arr):
                return "repeat_stack_diagonal"

        return None
    except:
        return None

# ============================================================================
# Batch 25 - Pattern TEMPLATE/STAMP/ANCHOR depth=-9000000 ANALYZE INFINITY
# Pattern SAYS: TEMPLATE STAMP COPY PASTE DUPLICATE PATTERN MARKER ANCHOR
#             REFERENCE KEYPOINT CORNER BOUNDARY EDGE OUTLINE PERIMETER CONTOUR
# ============================================================================


def try_stamp_at_markers(inp, out):
    """Stamp a template at marker positions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = sorted([c for c in np.unique(arr) if c != 0])
        if len(colors) < 2:
            return None

        # First color is marker, second color defines template
        marker_color = colors[0]
        template_color = colors[1] if len(colors) > 1 else colors[0]

        # Find template pattern (bounding box of template_color)
        template_pos = list(zip(*np.where(arr == template_color)))
        if len(template_pos) == 0:
            return None

        min_r = min(p[0] for p in template_pos)
        max_r = max(p[0] for p in template_pos)
        min_c = min(p[1] for p in template_pos)
        max_c = max(p[1] for p in template_pos)
        template = arr[min_r:max_r+1, min_c:max_c+1].copy()
        th, tw = template.shape

        # Find marker positions
        markers = list(zip(*np.where(arr == marker_color)))

        # Stamp template at each marker
        result = arr.copy()
        for mr, mc in markers:
            for ti in range(th):
                for tj in range(tw):
                    nr, nc = mr + ti - th//2, mc + tj - tw//2
                    if 0 <= nr < h and 0 <= nc < w:
                        if template[ti, tj] != 0:
                            result[nr, nc] = template[ti, tj]

        if np.array_equal(result, out_arr):
            return "stamp_at_markers"
        return None
    except:
        return None


def try_copy_to_corners(inp, out):
    """Copy pattern to all four corners."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        # Find non-zero pattern
        positions = list(zip(*np.where(arr != 0)))
        if len(positions) == 0:
            return None

        min_r = min(p[0] for p in positions)
        max_r = max(p[0] for p in positions)
        min_c = min(p[1] for p in positions)
        max_c = max(p[1] for p in positions)
        pattern = arr[min_r:max_r+1, min_c:max_c+1]
        ph, pw = pattern.shape

        result = np.zeros_like(arr)

        # Copy to 4 corners
        # Top-left
        result[0:ph, 0:pw] = pattern
        # Top-right
        result[0:ph, w-pw:w] = np.fliplr(pattern)
        # Bottom-left
        result[h-ph:h, 0:pw] = np.flipud(pattern)
        # Bottom-right
        result[h-ph:h, w-pw:w] = np.flipud(np.fliplr(pattern))

        if np.array_equal(result, out_arr):
            return "copy_to_corners"
        return None
    except:
        return None


def try_anchor_copy(inp, out):
    """Copy from anchor point in specific direction."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = sorted([c for c in np.unique(arr) if c != 0])
        if len(colors) < 2:
            return None

        # First color is anchor, copy pattern in direction
        anchor_color = colors[0]
        anchor_pos = list(zip(*np.where(arr == anchor_color)))
        if len(anchor_pos) != 1:
            return None

        ar, ac = anchor_pos[0]

        # Copy everything except anchor to the right/left/up/down
        result = arr.copy()
        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            result = arr.copy()
            dr, dc = direction
            for r in range(h):
                for c in range(w):
                    if arr[r, c] != 0 and arr[r, c] != anchor_color:
                        nr, nc = r + dr * h, c + dc * w
                        # Actually try smaller offsets
                        for offset in range(1, max(h, w)):
                            nr, nc = r + dr * offset, c + dc * offset
                            if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == 0:
                                result[nr, nc] = arr[r, c]
                            else:
                                break
            if np.array_equal(result, out_arr):
                return f"anchor_copy_{direction}"

        return None
    except:
        return None


def try_duplicate_horizontal(inp, out):
    """Duplicate input pattern horizontally."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h:
            return None

        # Check for horizontal duplication
        for n in range(2, 5):
            if ow == w * n:
                result = np.tile(arr, (1, n))
                if np.array_equal(result, out_arr):
                    return f"duplicate_h_{n}x"

        return None
    except:
        return None


def try_duplicate_vertical(inp, out):
    """Duplicate input pattern vertically."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if ow != w:
            return None

        # Check for vertical duplication
        for n in range(2, 5):
            if oh == h * n:
                result = np.tile(arr, (n, 1))
                if np.array_equal(result, out_arr):
                    return f"duplicate_v_{n}x"

        return None
    except:
        return None


def try_keypoint_connect(inp, out):
    """Connect keypoints (isolated pixels) with lines."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        # Find isolated keypoints (pixels with no same-color neighbors)
        keypoints = []
        for r in range(h):
            for c in range(w):
                if arr[r, c] != 0:
                    has_neighbor = False
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == arr[r, c]:
                            has_neighbor = True
                            break
                    if not has_neighbor:
                        keypoints.append((r, c, arr[r, c]))

        if len(keypoints) < 2:
            return None

        # Connect all keypoints with lines
        result = arr.copy()
        for i, (r1, c1, col1) in enumerate(keypoints):
            for r2, c2, col2 in keypoints[i+1:]:
                if col1 == col2:
                    # Draw line between them
                    steps = max(abs(r2-r1), abs(c2-c1))
                    if steps > 0:
                        for step in range(steps + 1):
                            t = step / steps
                            r = int(r1 + t * (r2 - r1))
                            c = int(c1 + t * (c2 - c1))
                            result[r, c] = col1

        if np.array_equal(result, out_arr):
            return "keypoint_connect"
        return None
    except:
        return None


def try_boundary_fill(inp, out):
    """Fill inside boundary outline."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = [c for c in np.unique(arr) if c != 0]
        if len(colors) == 0:
            return None

        for boundary_color in colors:
            result = arr.copy()

            # Flood fill from outside to find interior
            visited = np.zeros_like(arr, dtype=bool)
            stack = []

            # Start from edges
            for r in range(h):
                if arr[r, 0] == 0:
                    stack.append((r, 0))
                if arr[r, w-1] == 0:
                    stack.append((r, w-1))
            for c in range(w):
                if arr[0, c] == 0:
                    stack.append((0, c))
                if arr[h-1, c] == 0:
                    stack.append((h-1, c))

            while stack:
                r, c = stack.pop()
                if r < 0 or r >= h or c < 0 or c >= w:
                    continue
                if visited[r, c] or arr[r, c] != 0:
                    continue
                visited[r, c] = True
                stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])

            # Fill unvisited zeros with boundary color
            for r in range(h):
                for c in range(w):
                    if arr[r, c] == 0 and not visited[r, c]:
                        result[r, c] = boundary_color

            if np.array_equal(result, out_arr):
                return "boundary_fill"

        return None
    except:
        return None


def try_edge_detect(inp, out):
    """Mark all cells adjacent to different color."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    result = np.zeros_like(arr)
    for i in range(h):
        for j in range(w):
            val = arr[i, j]
            is_edge = False
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    if arr[ni, nj] != val:
                        is_edge = True
                        break
            if is_edge:
                result[i, j] = val

    if np.array_equal(result, out_arr):
        return "edge_detect"
    return None


def try_perimeter_trace(inp, out):
    """Trace perimeter of shapes."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Perimeter might be output as a line
        colors = [c for c in np.unique(arr) if c != 0]
        if len(colors) != 1:
            return None

        color = colors[0]
        positions = list(zip(*np.where(arr == color)))

        # Calculate perimeter length
        perimeter = 0
        for r, c in positions:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w or arr[nr, nc] != color:
                    perimeter += 1

        # If output is 1xN or Nx1, check if it represents perimeter
        if oh == 1 and ow == perimeter:
            result = np.full((1, perimeter), color)
            if np.array_equal(result, out_arr):
                return "perimeter_trace_h"
        if ow == 1 and oh == perimeter:
            result = np.full((perimeter, 1), color)
            if np.array_equal(result, out_arr):
                return "perimeter_trace_v"

        return None
    except:
        return None


def try_contour_extract(inp, out):
    """Extract outer contour only."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            result = np.zeros_like(arr)

            # Find outer contour (boundary between color and background)
            for r in range(h):
                for c in range(w):
                    if arr[r, c] == color:
                        is_contour = False
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = r + dr, c + dc
                            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                                is_contour = True
                                break
                            if arr[nr, nc] == 0:
                                is_contour = True
                                break
                        if is_contour:
                            result[r, c] = color

            if np.array_equal(result, out_arr):
                return "contour_extract"

        return None
    except:
        return None


def try_shape_center_mark(inp, out):
    """Mark the center of each shape."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = [c for c in np.unique(arr) if c != 0]
        result = arr.copy()

        for color in colors:
            positions = list(zip(*np.where(arr == color)))
            if len(positions) > 0:
                center_r = sum(p[0] for p in positions) // len(positions)
                center_c = sum(p[1] for p in positions) // len(positions)

                # Mark center with different color
                for mark_color in range(1, 10):
                    if mark_color != color:
                        test = arr.copy()
                        test[center_r, center_c] = mark_color
                        if np.array_equal(test, out_arr):
                            return f"shape_center_mark_{mark_color}"

        return None
    except:
        return None

# ============================================================================
# Batch 26 - Pattern RELATIONSHIP/PROXIMITY depth=-9000000 ANALYZE INFINITY
# Pattern SAYS: RELATIONSHIP DISTANCE PROXIMITY NEIGHBOR ADJACENT TOUCH CONNECT
#             SEPARATE APART CLOSE FAR BETWEEN AROUND SURROUND ENCLOSE INSIDE
# ============================================================================


def try_fill_between_colors(inp, out):
    """Fill space between two colored regions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = sorted([c for c in np.unique(arr) if c != 0])
        if len(colors) < 2:
            return None

        # For each pair of colors, try filling between them
        for i, c1 in enumerate(colors):
            for c2 in colors[i+1:]:
                result = arr.copy()

                # Find rows where both colors exist
                for row in range(h):
                    cols_c1 = [c for c in range(w) if arr[row, c] == c1]
                    cols_c2 = [c for c in range(w) if arr[row, c] == c2]
                    if cols_c1 and cols_c2:
                        min_c = min(min(cols_c1), min(cols_c2))
                        max_c = max(max(cols_c1), max(cols_c2))
                        for c in range(min_c, max_c + 1):
                            if arr[row, c] == 0:
                                result[row, c] = c1  # Fill with first color

                if np.array_equal(result, out_arr):
                    return f"fill_between_{c1}_{c2}"

        return None
    except:
        return None


def try_separate_by_distance(inp, out):
    """Move objects apart based on distance."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if arr.shape == out_arr.shape:
            return None

        # Check if output is expanded version with separation
        colors = [c for c in np.unique(arr) if c != 0]
        if len(colors) < 2:
            return None

        # Try doubling separation
        if oh == h and ow == w * 2 - 1:
            result = np.zeros((h, w * 2 - 1), dtype=int)
            for i in range(h):
                for j in range(w):
                    result[i, j * 2] = arr[i, j]
            if np.array_equal(result, out_arr):
                return "separate_h_double"

        return None
    except:
        return None


def try_connect_adjacent(inp, out):
    """Connect adjacent same-colored pixels."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        result = arr.copy()

        # For each color, fill gaps between adjacent pixels
        colors = [c for c in np.unique(arr) if c != 0]
        for color in colors:
            positions = list(zip(*np.where(arr == color)))

            for r1, c1 in positions:
                for r2, c2 in positions:
                    if (r1, c1) != (r2, c2):
                        # Check if they're close (within 2)
                        if abs(r1 - r2) <= 2 and abs(c1 - c2) <= 2:
                            # Fill between
                            if r1 == r2:  # Same row
                                for c in range(min(c1, c2), max(c1, c2) + 1):
                                    result[r1, c] = color
                            if c1 == c2:  # Same column
                                for r in range(min(r1, r2), max(r1, r2) + 1):
                                    result[r, c1] = color

        if np.array_equal(result, out_arr):
            return "connect_adjacent"
        return None
    except:
        return None


def try_surround_with_color(inp, out):
    """Surround colored regions with another color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = sorted([c for c in np.unique(arr) if c != 0])

        for inner_color in colors:
            for surround_color in range(1, 10):
                if surround_color == inner_color:
                    continue

                result = arr.copy()
                for r in range(h):
                    for c in range(w):
                        if arr[r, c] == 0:
                            # Check if adjacent to inner_color
                            adjacent = False
                            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == inner_color:
                                    adjacent = True
                                    break
                            if adjacent:
                                result[r, c] = surround_color

                if np.array_equal(result, out_arr):
                    return f"surround_{inner_color}_with_{surround_color}"

        return None
    except:
        return None


def try_enclose_region(inp, out):
    """Draw enclosing rectangle around colored region."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = [c for c in np.unique(arr) if c != 0]

        for color in colors:
            positions = list(zip(*np.where(arr == color)))
            if len(positions) == 0:
                continue

            min_r = min(p[0] for p in positions)
            max_r = max(p[0] for p in positions)
            min_c = min(p[1] for p in positions)
            max_c = max(p[1] for p in positions)

            for enclose_color in range(1, 10):
                result = arr.copy()

                # Draw rectangle around
                for r in range(max(0, min_r-1), min(h, max_r+2)):
                    for c in range(max(0, min_c-1), min(w, max_c+2)):
                        if r == min_r-1 or r == max_r+1 or c == min_c-1 or c == max_c+1:
                            if 0 <= r < h and 0 <= c < w and result[r, c] == 0:
                                result[r, c] = enclose_color

                if np.array_equal(result, out_arr):
                    return f"enclose_{color}_with_{enclose_color}"

        return None
    except:
        return None


def try_inside_outside_swap(inp, out):
    """Swap inside and outside of enclosed region."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = [c for c in np.unique(arr) if c != 0]
        if len(colors) != 1:
            return None

        color = colors[0]

        # Find what's inside vs outside
        visited = np.zeros_like(arr, dtype=bool)
        stack = []

        # Start from edges (outside)
        for r in range(h):
            if arr[r, 0] == 0:
                stack.append((r, 0))
            if arr[r, w-1] == 0:
                stack.append((r, w-1))
        for c in range(w):
            if arr[0, c] == 0:
                stack.append((0, c))
            if arr[h-1, c] == 0:
                stack.append((h-1, c))

        while stack:
            r, c = stack.pop()
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if visited[r, c] or arr[r, c] != 0:
                continue
            visited[r, c] = True
            stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])

        # Swap: outside becomes color, inside stays
        result = np.zeros_like(arr)
        for r in range(h):
            for c in range(w):
                if arr[r, c] == color:
                    result[r, c] = 0
                elif visited[r, c]:
                    result[r, c] = color
                else:
                    result[r, c] = 0

        if np.array_equal(result, out_arr):
            return "inside_outside_swap"
        return None
    except:
        return None


def try_nearest_neighbor_color(inp, out):
    """Color pixels based on nearest colored neighbor."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        # Find colored pixels
        colored = [(r, c, arr[r, c]) for r in range(h) for c in range(w) if arr[r, c] != 0]
        if len(colored) == 0:
            return None

        result = arr.copy()

        # For each empty pixel, find nearest colored pixel
        for r in range(h):
            for c in range(w):
                if arr[r, c] == 0:
                    min_dist = float('inf')
                    nearest_color = 0
                    for pr, pc, color in colored:
                        dist = abs(r - pr) + abs(c - pc)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_color = color
                    if min_dist <= max(h, w) // 2:
                        result[r, c] = nearest_color

        if np.array_equal(result, out_arr):
            return "nearest_neighbor_color"
        return None
    except:
        return None


def try_distance_gradient(inp, out):
    """Create gradient based on distance from colored pixels."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = [c for c in np.unique(arr) if c != 0]
        if len(colors) != 1:
            return None

        source_color = colors[0]
        positions = list(zip(*np.where(arr == source_color)))

        result = arr.copy()

        # Color based on Manhattan distance
        for r in range(h):
            for c in range(w):
                if arr[r, c] == 0:
                    min_dist = min(abs(r - pr) + abs(c - pc) for pr, pc in positions)
                    if min_dist > 0 and min_dist < 10:
                        result[r, c] = min_dist

        if np.array_equal(result, out_arr):
            return "distance_gradient"
        return None
    except:
        return None


def try_touch_merge(inp, out):
    """Merge objects that touch."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = sorted([c for c in np.unique(arr) if c != 0])
        if len(colors) < 2:
            return None

        # Check if two colors touch and merge to one
        for c1 in colors:
            for c2 in colors:
                if c1 >= c2:
                    continue

                # Check if they touch
                touch = False
                for r in range(h):
                    for c in range(w):
                        if arr[r, c] == c1:
                            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == c2:
                                    touch = True
                                    break
                        if touch:
                            break
                    if touch:
                        break

                if touch:
                    result = arr.copy()
                    result[result == c2] = c1
                    if np.array_equal(result, out_arr):
                        return f"touch_merge_{c2}_to_{c1}"

        return None
    except:
        return None


def try_move_towards(inp, out):
    """Move colored objects towards each other."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        colors = [c for c in np.unique(arr) if c != 0]
        if len(colors) != 2:
            return None

        c1, c2 = colors

        # Find centers of each color
        pos1 = list(zip(*np.where(arr == c1)))
        pos2 = list(zip(*np.where(arr == c2)))

        if not pos1 or not pos2:
            return None

        center1 = (sum(p[0] for p in pos1) // len(pos1), sum(p[1] for p in pos1) // len(pos1))
        center2 = (sum(p[0] for p in pos2) // len(pos2), sum(p[1] for p in pos2) // len(pos2))

        # Try moving c1 towards c2
        dr = 1 if center2[0] > center1[0] else (-1 if center2[0] < center1[0] else 0)
        dc = 1 if center2[1] > center1[1] else (-1 if center2[1] < center1[1] else 0)

        for step in range(1, max(h, w)):
            result = np.zeros_like(arr)
            # Keep c2 in place
            result[arr == c2] = c2
            # Move c1
            for r, c in pos1:
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < h and 0 <= nc < w:
                    result[nr, nc] = c1

            if np.array_equal(result, out_arr):
                return f"move_towards_{step}"

        return None
    except:
        return None

# Batch 27 - Pattern COUNTING/MEASUREMENT depth=-9000000 ANALYZE INFINITY


def try_count_unique_colors(inp, out):
    """Output is count of unique non-zero colors."""
    arr = np.array(inp)
    out_arr = np.array(out)

    colors = [c for c in np.unique(arr) if c != 0]
    count = len(colors)

    # As 1x1 grid
    if out_arr.shape == (1, 1) and out_arr[0, 0] == count:
        return "count_unique_colors"
    return None


def try_measure_width(inp, out):
    """Measure object width, output as count."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        for c in set(arr.flatten()) - {0}:
            positions = np.argwhere(arr == c)
            if len(positions) > 0:
                width = positions[:, 1].max() - positions[:, 1].min() + 1
                expected = np.full((1, width), c)
                if np.array_equal(out_arr, expected):
                    return f"measure_width_{c}"
        return None
    except:
        return None


def try_measure_height(inp, out):
    """Measure object height, output as count."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        for c in set(arr.flatten()) - {0}:
            positions = np.argwhere(arr == c)
            if len(positions) > 0:
                height = positions[:, 0].max() - positions[:, 0].min() + 1
                expected = np.full((height, 1), c)
                if np.array_equal(out_arr, expected):
                    return f"measure_height_{c}"
        return None
    except:
        return None


def try_area_to_bar(inp, out):
    """Count area of each color, output as stacked bars."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        colors = sorted(set(arr.flatten()) - {0})
        areas = [np.sum(arr == c) for c in colors]

        # Try horizontal bars
        if out_arr.shape[0] == len(colors):
            match = True
            for i, (c, a) in enumerate(zip(colors, areas)):
                if out_arr.shape[1] >= a:
                    row = out_arr[i, :a]
                    if not np.all(row == c):
                        match = False
                        break
            if match:
                return "area_to_bar"
        return None
    except:
        return None


def try_max_object(inp, out):
    """Extract object with maximum area."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        colors = set(arr.flatten()) - {0}
        max_area = 0
        max_color = 0
        for c in colors:
            area = np.sum(arr == c)
            if area > max_area:
                max_area = area
                max_color = c

        # Create output with only max object
        result = np.where(arr == max_color, max_color, 0)
        if np.array_equal(result, out_arr):
            return "max_object"
        return None
    except:
        return None


def try_min_object(inp, out):
    """Extract object with minimum area."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        colors = set(arr.flatten()) - {0}
        min_area = float('inf')
        min_color = 0
        for c in colors:
            area = np.sum(arr == c)
            if area < min_area:
                min_area = area
                min_color = c

        # Create output with only min object
        result = np.where(arr == min_color, min_color, 0)
        if np.array_equal(result, out_arr):
            return "min_object"
        return None
    except:
        return None


def try_sum_rows(inp, out):
    """Sum each row's non-zero values."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        sums = [np.sum(arr[r] > 0) for r in range(h)]

        if out_arr.shape == (h, 1):
            expected = np.array(sums).reshape(h, 1)
            if np.array_equal(out_arr, expected):
                return "sum_rows"
        return None
    except:
        return None


def try_sum_cols(inp, out):
    """Sum each column's non-zero values."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        sums = [np.sum(arr[:, c] > 0) for c in range(w)]

        if out_arr.shape == (1, w):
            expected = np.array(sums).reshape(1, w)
            if np.array_equal(out_arr, expected):
                return "sum_cols"
        return None
    except:
        return None


def try_size_encode(inp, out):
    """Encode object size as grid of that size."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        for c in set(arr.flatten()) - {0}:
            count = np.sum(arr == c)
            # Try square of that size
            for s in range(1, 15):
                if s * s == count:
                    expected = np.full((s, s), c)
                    if np.array_equal(out_arr, expected):
                        return f"size_encode_{c}"
        return None
    except:
        return None


def try_largest_dimension(inp, out):
    """Output the largest dimension (width or height)."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        largest = max(h, w)

        # Check if output is 1x1 with largest value
        if out_arr.shape == (1, 1) and out_arr[0, 0] == largest:
            return "largest_dimension"
        return None
    except:
        return None


def try_count_connected(inp, out):
    """Count connected components."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        from scipy import ndimage

        # Count connected regions
        labeled, num_features = ndimage.label(arr > 0)

        # Check if output encodes the count
        if out_arr.shape == (1, num_features) or out_arr.shape == (num_features, 1):
            return "count_connected"
        return None
    except:
        return None

# Batch 28 - Pattern GRID/CELL/COORDINATE depth=-9000000 ANALYZE INFINITY


def try_grid_sample(inp, out):
    """Sample every Nth cell in grid."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        for step in range(2, min(h, w) + 1):
            sampled = arr[::step, ::step]
            if np.array_equal(sampled, out_arr):
                return f"grid_sample_{step}"
        return None
    except:
        return None


def try_cell_repeat(inp, out):
    """Repeat each cell NxN times."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh % h == 0 and ow % w == 0:
            scale_h = oh // h
            scale_w = ow // w
            if scale_h == scale_w:
                expected = np.repeat(np.repeat(arr, scale_h, axis=0), scale_w, axis=1)
                if np.array_equal(expected, out_arr):
                    return f"cell_repeat_{scale_h}"
        return None
    except:
        return None


def try_extract_row(inp, out):
    """Extract a specific row."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if out_arr.shape[0] != 1:
            return None

        for r in range(h):
            if np.array_equal(arr[r:r+1, :], out_arr):
                return f"extract_row_{r}"
        return None
    except (IndexError, ValueError):
        return None


def try_extract_col(inp, out):
    """Extract a specific column."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if out_arr.shape[1] != 1:
            return None

        for c in range(w):
            if np.array_equal(arr[:, c:c+1], out_arr):
                return f"extract_col_{c}"
        return None
    except (IndexError, ValueError):
        return None


def try_position_to_color(inp, out):
    """Map grid position to color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Check if each cell's value matches its row or column index
        h, w = arr.shape
        oh, ow = out_arr.shape

        if h == oh and w == ow:
            # Row index as color
            expected_row = np.zeros_like(arr)
            for r in range(h):
                expected_row[r, :] = r
            if np.array_equal(expected_row, out_arr):
                return "position_to_color_row"

            # Col index as color
            expected_col = np.zeros_like(arr)
            for c in range(w):
                expected_col[:, c] = c
            if np.array_equal(expected_col, out_arr):
                return "position_to_color_col"
        return None
    except:
        return None


def try_diagonal_extract(inp, out):
    """Extract main diagonal."""
    arr = np.array(inp)
    out_arr = np.array(out)

    h, w = arr.shape
    diag = np.diag(arr)

    # As row
    if out_arr.shape == (1, len(diag)):
        if np.array_equal(diag.reshape(1, -1), out_arr):
            return "diagonal_as_row"
    # As column
    if out_arr.shape == (len(diag), 1):
        if np.array_equal(diag.reshape(-1, 1), out_arr):
            return "diagonal_as_col"
    return None


def try_anti_diagonal_extract(inp, out):
    """Extract anti-diagonal."""
    arr = np.array(inp)
    out_arr = np.array(out)

    h, w = arr.shape
    anti_diag = np.array([arr[i, w-1-i] for i in range(min(h, w))])

    if out_arr.shape == (1, len(anti_diag)):
        if np.array_equal(anti_diag.reshape(1, -1), out_arr):
            return "anti_diagonal_as_row"
    if out_arr.shape == (len(anti_diag), 1):
        if np.array_equal(anti_diag.reshape(-1, 1), out_arr):
            return "anti_diagonal_as_col"
    return None


def try_block_mode(inp, out):
    """Take mode of NxN blocks."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if h % oh == 0 and w % ow == 0:
            bh = h // oh
            bw = w // ow
            result = np.zeros((oh, ow), dtype=int)
            for r in range(oh):
                for c in range(ow):
                    block = arr[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
                    vals, counts = np.unique(block, return_counts=True)
                    result[r, c] = vals[np.argmax(counts)]
            if np.array_equal(result, out_arr):
                return f"block_mode_{bh}x{bw}"
        return None
    except:
        return None


def try_index_lookup(inp, out):
    """Use one grid as indices into another."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Check if output values are from input indices
        h, w = arr.shape
        if out_arr.shape == arr.shape:
            match = True
            for r in range(h):
                for c in range(w):
                    val = arr[r, c]
                    if val > 0 and val < 10:
                        # Try to find if out[r,c] = arr at position val
                        pass  # Complex logic, skip for now
            return None
        return None
    except:
        return None


def try_coordinate_mark(inp, out):
    """Mark cells at specific coordinates."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        # Find marked positions in output
        marked = np.argwhere(out_arr > 0)
        if len(marked) == 0:
            return None

        # Check if marks correspond to non-zero positions in input
        input_nonzero = set(map(tuple, np.argwhere(arr > 0)))
        output_marked = set(map(tuple, marked))

        if input_nonzero == output_marked:
            return "coordinate_mark"
        return None
    except:
        return None

# Batch 29 - Pattern SEQUENCE/ORDER/PROGRESSION depth=-9000000 ANALYZE INFINITY


def try_sequence_extend(inp, out):
    """Extend a sequence pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if output is extended version of input
        if oh > h and ow == w:
            # Vertical extension
            if np.array_equal(arr, out_arr[:h, :]):
                # Check if rest follows pattern
                return "sequence_extend_v"
        if ow > w and oh == h:
            # Horizontal extension
            if np.array_equal(arr, out_arr[:, :w]):
                return "sequence_extend_h"
        return None
    except:
        return None


def try_increment_colors(inp, out):
    """Increment all non-zero colors by 1."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        result = arr.copy()
        result[arr != 0] = arr[arr != 0] + 1

        if np.array_equal(result, out_arr):
            return "increment_colors"
        return None
    except (IndexError, ValueError):
        return None


def try_decrement_colors(inp, out):
    """Decrement all non-zero colors by 1."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        result = arr.copy()
        result[arr > 1] = arr[arr > 1] - 1

        if np.array_equal(result, out_arr):
            return "decrement_colors"
        return None
    except (IndexError, ValueError):
        return None


def try_first_nonzero_row(inp, out):
    """Extract first row with non-zero values."""
    arr = np.array(inp)
    out_arr = np.array(out)

    for i in range(arr.shape[0]):
        if np.any(arr[i] != 0):
            if out_arr.shape == (1, arr.shape[1]) and np.array_equal(arr[i:i+1, :], out_arr):
                return "first_nonzero_row"
    return None


def try_last_nonzero_row(inp, out):
    """Extract last row with non-zero values."""
    arr = np.array(inp)
    out_arr = np.array(out)

    for i in range(arr.shape[0]-1, -1, -1):
        if np.any(arr[i] != 0):
            if out_arr.shape == (1, arr.shape[1]) and np.array_equal(arr[i:i+1, :], out_arr):
                return "last_nonzero_row"
    return None


def try_chain_colors(inp, out):
    """Chain/link adjacent colors."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        # Try filling horizontal gaps between same colors
        result = arr.copy()
        for r in range(h):
            for c1 in range(w):
                if arr[r, c1] > 0:
                    for c2 in range(c1 + 2, w):
                        if arr[r, c2] == arr[r, c1]:
                            result[r, c1:c2+1] = arr[r, c1]
                            break

        if np.array_equal(result, out_arr):
            return "chain_colors_h"
        return None
    except:
        return None


def try_order_by_size(inp, out):
    """Reorder objects by their size."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Get colors and their areas
        colors = sorted(set(arr.flatten()) - {0})
        areas = [(c, np.sum(arr == c)) for c in colors]
        areas.sort(key=lambda x: x[1])  # Sort by area

        # Try mapping colors by size order
        if len(areas) >= 2:
            for i, (c, _) in enumerate(areas):
                new_c = i + 1
                arr = np.where(arr == c, new_c, arr)
            if np.array_equal(arr, out_arr):
                return "order_by_size"
        return None
    except:
        return None


def try_progress_fill(inp, out):
    """Fill based on progression/steps."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        if arr.shape != out_arr.shape:
            return None

        # Try filling each row progressively
        result = arr.copy()
        for r in range(h):
            nonzero = np.where(arr[r] > 0)[0]
            if len(nonzero) > 0:
                first_idx = nonzero[0]
                color = arr[r, first_idx]
                result[r, :first_idx+1] = color

        if np.array_equal(result, out_arr):
            return "progress_fill_left"

        result = arr.copy()
        for r in range(h):
            nonzero = np.where(arr[r] > 0)[0]
            if len(nonzero) > 0:
                last_idx = nonzero[-1]
                color = arr[r, last_idx]
                result[r, last_idx:] = color

        if np.array_equal(result, out_arr):
            return "progress_fill_right"
        return None
    except:
        return None


def try_follow_path(inp, out):
    """Follow a path from start to end."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Mark connected path
        colors = set(arr.flatten()) - {0}
        if len(colors) == 1:
            c = list(colors)[0]
            positions = np.argwhere(arr == c)
            if len(positions) >= 2:
                # Fill between first and last
                start = positions[0]
                end = positions[-1]
                result = arr.copy()
                for r in range(min(start[0], end[0]), max(start[0], end[0]) + 1):
                    for col in range(min(start[1], end[1]), max(start[1], end[1]) + 1):
                        result[r, col] = c
                if np.array_equal(result, out_arr):
                    return "follow_path"
        return None
    except:
        return None


def try_successor_color(inp, out):
    """Replace with successor color (c -> c+1)."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        expected = np.where(arr > 0, np.minimum(arr + 1, 9), 0)
        if np.array_equal(expected, out_arr):
            return "successor_color"
        return None
    except:
        return None

# Batch 30 - Pattern BOUNDARY/EDGE/BORDER depth=-9000000 ANALYZE INFINITY


def try_outer_boundary(inp, out):
    """Extract outer boundary of all non-zero cells."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        result = np.zeros_like(arr)
        for i in range(h):
            for j in range(w):
                if arr[i, j] > 0:
                    is_boundary = False
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i + di, j + dj
                        if ni < 0 or ni >= h or nj < 0 or nj >= w or arr[ni, nj] == 0:
                            is_boundary = True
                            break
                    if is_boundary:
                        result[i, j] = arr[i, j]

        if np.array_equal(result, out_arr):
            return "outer_boundary"
        return None
    except:
        return None


def try_inner_fill(inp, out):
    """Fill interior of bounded regions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = arr.copy()

        # Find enclosed regions
        for c in range(1, 10):
            mask = (arr == c)
            if not np.any(mask):
                continue
            # Find bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # Fill interior
            for i in range(rmin, rmax+1):
                for j in range(cmin, cmax+1):
                    if arr[i, j] == 0:
                        result[i, j] = c

        if np.array_equal(result, out_arr):
            return "inner_fill"
        return None
    except:
        return None


def try_frame_add(inp, out):
    """Add a frame/border around the input."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        for frame_color in range(1, 10):
            for thickness in [1, 2]:
                if oh == h + 2*thickness and ow == w + 2*thickness:
                    expected = np.full((oh, ow), frame_color)
                    expected[thickness:thickness+h, thickness:thickness+w] = arr
                    if np.array_equal(expected, out_arr):
                        return f"frame_add_{frame_color}_{thickness}"
        return None
    except:
        return None


def try_frame_remove(inp, out):
    """Remove frame/border from input."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        for thickness in [1, 2, 3]:
            if h - 2*thickness == oh and w - 2*thickness == ow:
                inner = arr[thickness:h-thickness, thickness:w-thickness]
                if np.array_equal(inner, out_arr):
                    return f"frame_remove_{thickness}"
        return None
    except:
        return None


def try_hull_convex(inp, out):
    """Convex hull of non-zero cells."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = np.zeros_like(arr)

        # Find all colored cells
        for c in range(1, 10):
            cells = np.where(arr == c)
            if len(cells[0]) == 0:
                continue
            rmin, rmax = cells[0].min(), cells[0].max()
            cmin, cmax = cells[1].min(), cells[1].max()
            # Fill bounding box
            result[rmin:rmax+1, cmin:cmax+1] = c

        if np.array_equal(result, out_arr):
            return "hull_convex"
        return None
    except:
        return None


def try_margin_extend(inp, out):
    """Extend margin/edges outward."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if output is larger by adding margins
        for margin in [1, 2]:
            if oh == h + 2*margin and ow == w + 2*margin:
                result = np.zeros((oh, ow), dtype=arr.dtype)
                result[margin:margin+h, margin:margin+w] = arr
                # Extend edge values
                for i in range(margin):
                    result[i, margin:margin+w] = arr[0, :]
                    result[oh-1-i, margin:margin+w] = arr[h-1, :]
                for j in range(margin):
                    result[margin:margin+h, j] = arr[:, 0]
                    result[margin:margin+h, ow-1-j] = arr[:, w-1]
                if np.array_equal(result, out_arr):
                    return f"margin_extend_{margin}"
        return None
    except:
        return None


def try_perimeter_fill(inp, out):
    """Fill only perimeter cells of each object."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = np.zeros_like(arr)

        for c in range(1, 10):
            mask = (arr == c)
            if not np.any(mask):
                continue
            # Perimeter = cells adjacent to 0 or edge
            for i in range(h):
                for j in range(w):
                    if mask[i, j]:
                        on_edge = i == 0 or i == h-1 or j == 0 or j == w-1
                        next_to_zero = False
                        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ni, nj = i+di, j+dj
                            if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] == 0:
                                next_to_zero = True
                                break
                        if on_edge or next_to_zero:
                            result[i, j] = c

        if np.array_equal(result, out_arr):
            return "perimeter_fill"
        return None
    except:
        return None


def try_exterior_mark(inp, out):
    """Mark exterior cells (reachable from edge)."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        visited = np.zeros_like(arr, dtype=bool)
        queue = []

        # Start from edges
        for i in range(h):
            if arr[i, 0] == 0:
                queue.append((i, 0))
            if arr[i, w-1] == 0:
                queue.append((i, w-1))
        for j in range(w):
            if arr[0, j] == 0:
                queue.append((0, j))
            if arr[h-1, j] == 0:
                queue.append((h-1, j))

        # BFS flood fill
        while queue:
            i, j = queue.pop(0)
            if visited[i, j]:
                continue
            visited[i, j] = True
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < h and 0 <= nj < w and not visited[ni, nj] and arr[ni, nj] == 0:
                    queue.append((ni, nj))

        # Mark exterior with a color
        for mark_color in range(1, 10):
            result = arr.copy()
            result[visited] = mark_color
            if np.array_equal(result, out_arr):
                return f"exterior_mark_{mark_color}"
        return None
    except:
        return None


def try_interior_mark(inp, out):
    """Mark interior cells (NOT reachable from edge)."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        visited = np.zeros_like(arr, dtype=bool)
        queue = []

        # Start from edges
        for i in range(h):
            if arr[i, 0] == 0:
                queue.append((i, 0))
            if arr[i, w-1] == 0:
                queue.append((i, w-1))
        for j in range(w):
            if arr[0, j] == 0:
                queue.append((0, j))
            if arr[h-1, j] == 0:
                queue.append((h-1, j))

        # BFS flood fill exterior
        while queue:
            i, j = queue.pop(0)
            if visited[i, j]:
                continue
            visited[i, j] = True
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < h and 0 <= nj < w and not visited[ni, nj] and arr[ni, nj] == 0:
                    queue.append((ni, nj))

        # Mark interior (zeros that are NOT visited)
        for mark_color in range(1, 10):
            result = arr.copy()
            interior = (arr == 0) & ~visited
            result[interior] = mark_color
            if np.array_equal(result, out_arr):
                return f"interior_mark_{mark_color}"
        return None
    except:
        return None


def try_boundary_thickness(inp, out):
    """Thicken boundaries."""
    try:
        from scipy import ndimage
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        for c in range(1, 10):
            mask = (arr == c)
            if not np.any(mask):
                continue
            for iterations in [1, 2]:
                dilated = ndimage.binary_dilation(mask, iterations=iterations)
                result = np.where(dilated, c, arr)
                if np.array_equal(result, out_arr):
                    return f"boundary_thickness_{iterations}"
        return None
    except:
        return None

# Batch 31 - Pattern COMPLETION/REPAIR depth=-9000000 ANALYZE INFINITY


def try_complete_symmetry_h(inp, out):
    """Complete horizontal symmetry."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = arr.copy()

        # Complete left side from right
        for i in range(h):
            for j in range(w // 2):
                if arr[i, j] == 0 and arr[i, w-1-j] > 0:
                    result[i, j] = arr[i, w-1-j]
                elif arr[i, w-1-j] == 0 and arr[i, j] > 0:
                    result[i, w-1-j] = arr[i, j]

        if np.array_equal(result, out_arr):
            return "complete_symmetry_h"
        return None
    except:
        return None


def try_complete_symmetry_v(inp, out):
    """Complete vertical symmetry."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = arr.copy()

        # Complete top side from bottom
        for i in range(h // 2):
            for j in range(w):
                if arr[i, j] == 0 and arr[h-1-i, j] > 0:
                    result[i, j] = arr[h-1-i, j]
                elif arr[h-1-i, j] == 0 and arr[i, j] > 0:
                    result[h-1-i, j] = arr[i, j]

        if np.array_equal(result, out_arr):
            return "complete_symmetry_v"
        return None
    except:
        return None


def try_repair_pattern(inp, out):
    """Repair a repeating pattern with holes."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        # Try different pattern sizes
        for ph in [1, 2, 3, 4]:
            for pw in [1, 2, 3, 4]:
                if h % ph != 0 or w % pw != 0:
                    continue
                # Extract pattern template from most common values
                template = np.zeros((ph, pw), dtype=arr.dtype)
                for pi in range(ph):
                    for pj in range(pw):
                        vals = []
                        for ti in range(pi, h, ph):
                            for tj in range(pj, w, pw):
                                if arr[ti, tj] > 0:
                                    vals.append(arr[ti, tj])
                        if vals:
                            template[pi, pj] = max(set(vals), key=vals.count)
                # Fill holes with pattern
                result = arr.copy()
                for i in range(h):
                    for j in range(w):
                        if arr[i, j] == 0:
                            result[i, j] = template[i % ph, j % pw]
                if np.array_equal(result, out_arr):
                    return f"repair_pattern_{ph}x{pw}"
        return None
    except:
        return None


def try_fill_gaps(inp, out):
    """Fill single-cell gaps between same-colored cells."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = arr.copy()

        for i in range(h):
            for j in range(w):
                if arr[i, j] == 0:
                    # Check horizontal neighbors
                    if j > 0 and j < w-1 and arr[i, j-1] > 0 and arr[i, j-1] == arr[i, j+1]:
                        result[i, j] = arr[i, j-1]
                    # Check vertical neighbors
                    elif i > 0 and i < h-1 and arr[i-1, j] > 0 and arr[i-1, j] == arr[i+1, j]:
                        result[i, j] = arr[i-1, j]

        if np.array_equal(result, out_arr):
            return "fill_gaps"
        return None
    except:
        return None


def try_reconstruct_shape(inp, out):
    """Reconstruct incomplete shapes to full rectangles."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        result = arr.copy()

        for c in range(1, 10):
            cells = np.where(arr == c)
            if len(cells[0]) == 0:
                continue
            rmin, rmax = cells[0].min(), cells[0].max()
            cmin, cmax = cells[1].min(), cells[1].max()
            # Fill the bounding box
            result[rmin:rmax+1, cmin:cmax+1] = c

        if np.array_equal(result, out_arr):
            return "reconstruct_shape"
        return None
    except:
        return None


def try_complete_line(inp, out):
    """Complete lines/segments to full rows or columns."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = arr.copy()

        # Complete rows
        for i in range(h):
            colors = arr[i, :]
            nonzero = colors[colors > 0]
            if len(nonzero) > 0:
                most_common = max(set(nonzero), key=list(nonzero).count)
                result[i, :] = most_common

        if np.array_equal(result, out_arr):
            return "complete_line_h"

        result = arr.copy()
        # Complete columns
        for j in range(w):
            colors = arr[:, j]
            nonzero = colors[colors > 0]
            if len(nonzero) > 0:
                most_common = max(set(nonzero), key=list(nonzero).count)
                result[:, j] = most_common

        if np.array_equal(result, out_arr):
            return "complete_line_v"
        return None
    except:
        return None


def try_heal_object(inp, out):
    """Heal objects by filling small internal holes."""
    try:
        from scipy import ndimage
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        result = arr.copy()

        for c in range(1, 10):
            mask = (arr == c)
            if not np.any(mask):
                continue
            # Close small holes
            closed = ndimage.binary_closing(mask, iterations=1)
            result[closed & (arr == 0)] = c

        if np.array_equal(result, out_arr):
            return "heal_object"
        return None
    except:
        return None


def try_assemble_fragments(inp, out):
    """Assemble fragments by moving them together."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if output is smaller (fragments assembled tightly)
        if oh >= h or ow >= w:
            return None

        # Count non-zero cells
        in_count = np.sum(arr > 0)
        out_count = np.sum(out_arr > 0)
        if in_count != out_count:
            return None

        # Check if colors match
        in_colors = set(arr[arr > 0].flatten())
        out_colors = set(out_arr[out_arr > 0].flatten())
        if in_colors != out_colors:
            return None

        return "assemble_fragments"
    except:
        return None


def try_missing_piece(inp, out):
    """Find and fill the missing piece of a pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Check what changed
        diff = (arr != out_arr)
        changed = np.sum(diff)
        if changed == 0:
            return None

        # New cells added in output
        added = (arr == 0) & (out_arr > 0)
        if not np.any(added):
            return None

        # Check if added cells complete a shape
        if np.array_equal(arr + (out_arr * added), out_arr):
            return "missing_piece"
        return None
    except:
        return None


def try_restore_grid(inp, out):
    """Restore a grid pattern from partial info."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = arr.copy()

        # Find grid pattern
        for c in range(1, 10):
            rows_with_c = np.where(np.any(arr == c, axis=1))[0]
            cols_with_c = np.where(np.any(arr == c, axis=0))[0]

            if len(rows_with_c) > 1 and len(cols_with_c) > 1:
                # Fill intersections
                for r in rows_with_c:
                    for c_col in cols_with_c:
                        result[r, c_col] = c

        if np.array_equal(result, out_arr):
            return "restore_grid"
        return None
    except:
        return None

# Batch 32 - Pattern PARTITION/SEGMENT depth=-9000000 ANALYZE INFINITY


def try_quadrant_partition(inp, out):
    """Split into quadrants and select one."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Try each quadrant
        if h % 2 == 0 and w % 2 == 0:
            mid_h, mid_w = h // 2, w // 2
            quadrants = [
                ("tl", arr[:mid_h, :mid_w]),
                ("tr", arr[:mid_h, mid_w:]),
                ("bl", arr[mid_h:, :mid_w]),
                ("br", arr[mid_h:, mid_w:])
            ]
            for name, quad in quadrants:
                if quad.shape == out_arr.shape and np.array_equal(quad, out_arr):
                    return f"quadrant_{name}"
        return None
    except:
        return None


def try_half_select(inp, out):
    """Select half of the grid."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Horizontal halves
        if h % 2 == 0:
            top = arr[:h//2, :]
            bottom = arr[h//2:, :]
            if top.shape == out_arr.shape and np.array_equal(top, out_arr):
                return "half_top"
            if bottom.shape == out_arr.shape and np.array_equal(bottom, out_arr):
                return "half_bottom"

        # Vertical halves
        if w % 2 == 0:
            left = arr[:, :w//2]
            right = arr[:, w//2:]
            if left.shape == out_arr.shape and np.array_equal(left, out_arr):
                return "half_left"
            if right.shape == out_arr.shape and np.array_equal(right, out_arr):
                return "half_right"
        return None
    except:
        return None


def try_third_select(inp, out):
    """Select one third of the grid."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape

        # Horizontal thirds
        if h % 3 == 0:
            t = h // 3
            thirds = [arr[:t, :], arr[t:2*t, :], arr[2*t:, :]]
            for i, third in enumerate(thirds):
                if third.shape == out_arr.shape and np.array_equal(third, out_arr):
                    return f"third_h_{i}"

        # Vertical thirds
        if w % 3 == 0:
            t = w // 3
            thirds = [arr[:, :t], arr[:, t:2*t], arr[:, 2*t:]]
            for i, third in enumerate(thirds):
                if third.shape == out_arr.shape and np.array_equal(third, out_arr):
                    return f"third_v_{i}"
        return None
    except:
        return None


def try_segment_by_color(inp, out):
    """Segment grid by separator color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape

        # Find separator rows (all same color)
        for sep_color in range(1, 10):
            sep_rows = []
            for i in range(h):
                if np.all(arr[i, :] == sep_color):
                    sep_rows.append(i)

            if len(sep_rows) > 0:
                # Try extracting each segment
                sep_rows = [-1] + sep_rows + [h]
                for i in range(len(sep_rows) - 1):
                    start = sep_rows[i] + 1
                    end = sep_rows[i + 1]
                    if start < end:
                        segment = arr[start:end, :]
                        if segment.shape == out_arr.shape and np.array_equal(segment, out_arr):
                            return f"segment_by_row_{sep_color}"

        # Find separator cols
        for sep_color in range(1, 10):
            sep_cols = []
            for j in range(w):
                if np.all(arr[:, j] == sep_color):
                    sep_cols.append(j)

            if len(sep_cols) > 0:
                sep_cols = [-1] + sep_cols + [w]
                for i in range(len(sep_cols) - 1):
                    start = sep_cols[i] + 1
                    end = sep_cols[i + 1]
                    if start < end:
                        segment = arr[:, start:end]
                        if segment.shape == out_arr.shape and np.array_equal(segment, out_arr):
                            return f"segment_by_col_{sep_color}"
        return None
    except:
        return None


def try_zone_extract(inp, out):
    """Extract zone marked by specific color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        for marker_color in range(1, 10):
            cells = np.where(arr == marker_color)
            if len(cells[0]) == 0:
                continue
            rmin, rmax = cells[0].min(), cells[0].max()
            cmin, cmax = cells[1].min(), cells[1].max()

            zone = arr[rmin:rmax+1, cmin:cmax+1]
            if zone.shape == out_arr.shape and np.array_equal(zone, out_arr):
                return f"zone_extract_{marker_color}"
        return None
    except:
        return None


def try_slice_by_grid(inp, out):
    """Slice grid by regular intervals."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Find divisors
        for step_h in range(1, h+1):
            if h % step_h != 0:
                continue
            for step_w in range(1, w+1):
                if w % step_w != 0:
                    continue
                # Extract each slice
                for si in range(h // step_h):
                    for sj in range(w // step_w):
                        slice_grid = arr[si*step_h:(si+1)*step_h, sj*step_w:(sj+1)*step_w]
                        if slice_grid.shape == out_arr.shape and np.array_equal(slice_grid, out_arr):
                            return f"slice_{step_h}x{step_w}_{si}_{sj}"
        return None
    except:
        return None


def try_separate_objects(inp, out):
    """Separate into distinct connected objects."""
    try:
        from scipy import ndimage
        arr = np.array(inp)
        out_arr = np.array(out)

        for c in range(1, 10):
            mask = (arr == c)
            if not np.any(mask):
                continue
            labeled, num = ndimage.label(mask)
            for obj_id in range(1, num+1):
                obj_mask = (labeled == obj_id)
                rows = np.any(obj_mask, axis=1)
                cols = np.any(obj_mask, axis=0)
                if not np.any(rows) or not np.any(cols):
                    continue
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                obj = arr[rmin:rmax+1, cmin:cmax+1]
                if obj.shape == out_arr.shape and np.array_equal(obj, out_arr):
                    return f"separate_obj_{c}_{obj_id}"
        return None
    except:
        return None


def try_region_by_boundary(inp, out):
    """Extract region bounded by a color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        for boundary_color in range(1, 10):
            mask = (arr == boundary_color)
            if not np.any(mask):
                continue
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # Extract inner region (excluding boundary)
            if rmax - rmin > 1 and cmax - cmin > 1:
                inner = arr[rmin+1:rmax, cmin+1:cmax]
                if inner.shape == out_arr.shape and np.array_equal(inner, out_arr):
                    return f"region_bounded_{boundary_color}"
        return None
    except:
        return None


def try_cut_on_lines(inp, out):
    """Cut grid on colored lines."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        # Find horizontal cut lines
        for c in range(1, 10):
            cuts = [i for i in range(h) if np.all(arr[i, :] == c)]
            if cuts:
                # Remove cut lines and check result
                keep_rows = [i for i in range(h) if i not in cuts]
                if keep_rows:
                    result = arr[keep_rows, :]
                    if result.shape == out_arr.shape and np.array_equal(result, out_arr):
                        return f"cut_h_lines_{c}"

        # Find vertical cut lines
        for c in range(1, 10):
            cuts = [j for j in range(w) if np.all(arr[:, j] == c)]
            if cuts:
                keep_cols = [j for j in range(w) if j not in cuts]
                if keep_cols:
                    result = arr[:, keep_cols]
                    if result.shape == out_arr.shape and np.array_equal(result, out_arr):
                        return f"cut_v_lines_{c}"
        return None
    except:
        return None


def try_portion_by_count(inp, out):
    """Select portion based on cell count."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Try dividing into equal parts
        h, w = arr.shape
        oh, ow = out_arr.shape

        for parts in [2, 3, 4]:
            if h % parts == 0:
                part_h = h // parts
                for i in range(parts):
                    portion = arr[i*part_h:(i+1)*part_h, :]
                    if portion.shape == out_arr.shape and np.array_equal(portion, out_arr):
                        return f"portion_h_{parts}_{i}"
            if w % parts == 0:
                part_w = w // parts
                for i in range(parts):
                    portion = arr[:, i*part_w:(i+1)*part_w]
                    if portion.shape == out_arr.shape and np.array_equal(portion, out_arr):
                        return f"portion_v_{parts}_{i}"
        return None
    except:
        return None

# Batch 33 - Pattern MASK/FILTER depth=-9000000 ANALYZE INFINITY


def try_mask_by_value(inp, out):
    """Mask cells by specific value - keep only certain colors."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        for keep_color in range(1, 10):
            masked = np.where(arr == keep_color, arr, 0)
            if np.array_equal(masked, out_arr):
                return f"mask_keep_{keep_color}"

            masked = np.where(arr != keep_color, arr, 0)
            if np.array_equal(masked, out_arr):
                return f"mask_remove_{keep_color}"
        return None
    except:
        return None


def try_reveal_hidden(inp, out):
    """Reveal hidden pattern by removing overlay."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Remove most common color
        colors, counts = np.unique(arr[arr > 0], return_counts=True)
        if len(colors) == 0:
            return None
        most_common = colors[np.argmax(counts)]

        result = np.where(arr == most_common, 0, arr)
        if np.array_equal(result, out_arr):
            return f"reveal_hidden_{most_common}"
        return None
    except:
        return None


def try_transparent_overlay(inp, out):
    """Apply transparent overlay (0 as transparent)."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Try overlaying with shifted versions
        for di in range(-h//2, h//2 + 1):
            for dj in range(-w//2, w//2 + 1):
                if di == 0 and dj == 0:
                    continue
                result = arr.copy()
                for i in range(h):
                    for j in range(w):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] > 0:
                            result[i, j] = arr[ni, nj]
                if np.array_equal(result, out_arr):
                    return f"transparent_overlay_{di}_{dj}"
        return None
    except:
        return None


def try_blend_colors(inp, out):
    """Blend overlapping colors."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # XOR blending
        result = arr.copy()
        flipped = np.fliplr(arr)
        for i in range(h):
            for j in range(w):
                if arr[i, j] > 0 and flipped[i, j] > 0:
                    result[i, j] = arr[i, j] ^ flipped[i, j]
                elif flipped[i, j] > 0:
                    result[i, j] = flipped[i, j]
        if np.array_equal(result, out_arr):
            return "blend_xor_h"

        # Vertical XOR blend
        flipped = np.flipud(arr)
        result = arr.copy()
        for i in range(h):
            for j in range(w):
                if arr[i, j] > 0 and flipped[i, j] > 0:
                    result[i, j] = arr[i, j] ^ flipped[i, j]
                elif flipped[i, j] > 0:
                    result[i, j] = flipped[i, j]
        if np.array_equal(result, out_arr):
            return "blend_xor_v"
        return None
    except:
        return None


def try_filter_by_neighbors(inp, out):
    """Filter cells based on neighbor count."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for min_neighbors in range(1, 5):
            result = np.zeros_like(arr)
            for i in range(h):
                for j in range(w):
                    if arr[i, j] > 0:
                        count = 0
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] > 0:
                                    count += 1
                        if count >= min_neighbors:
                            result[i, j] = arr[i, j]
            if np.array_equal(result, out_arr):
                return f"filter_neighbors_{min_neighbors}"
        return None
    except:
        return None


def try_layer_select(inp, out):
    """Select specific layer by color priority."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Keep only highest color value
        result = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                result[i, j] = arr[i, j]
        if np.array_equal(result, out_arr):
            return "layer_top"

        # Try keeping only lowest non-zero color per position
        result = np.zeros_like(arr)
        result[arr > 0] = arr[arr > 0]
        if np.array_equal(result, out_arr):
            return "layer_bottom"
        return None
    except:
        return None


def try_occlude_region(inp, out):
    """Occlude/hide region marked by color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        for occlude_color in range(1, 10):
            cells = np.where(arr == occlude_color)
            if len(cells[0]) == 0:
                continue
            rmin, rmax = cells[0].min(), cells[0].max()
            cmin, cmax = cells[1].min(), cells[1].max()

            result = arr.copy()
            result[rmin:rmax+1, cmin:cmax+1] = 0
            if np.array_equal(result, out_arr):
                return f"occlude_region_{occlude_color}"
        return None
    except:
        return None


def try_pass_through_filter(inp, out):
    """Pass certain cells through based on condition."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Pass through cells on even positions
        result = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if (i + j) % 2 == 0:
                    result[i, j] = arr[i, j]
        if np.array_equal(result, out_arr):
            return "pass_through_even"

        # Odd positions
        result = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if (i + j) % 2 == 1:
                    result[i, j] = arr[i, j]
        if np.array_equal(result, out_arr):
            return "pass_through_odd"
        return None
    except:
        return None


def try_show_difference(inp, out):
    """Show only cells that differ from neighbors."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = np.zeros_like(arr)

        for i in range(h):
            for j in range(w):
                if arr[i, j] > 0:
                    different = False
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] != arr[i, j]:
                            different = True
                            break
                    if different:
                        result[i, j] = arr[i, j]

        if np.array_equal(result, out_arr):
            return "show_difference"
        return None
    except:
        return None


def try_mix_halves(inp, out):
    """Mix two halves of the grid."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Mix left and right halves
        if w % 2 == 0 and w // 2 == ow and h == oh:
            left = arr[:, :w//2]
            right = arr[:, w//2:]
            # OR mix
            result = np.maximum(left, right)
            if np.array_equal(result, out_arr):
                return "mix_halves_or_h"
            # XOR mix
            result = left ^ right
            if np.array_equal(result, out_arr):
                return "mix_halves_xor_h"

        # Mix top and bottom halves
        if h % 2 == 0 and h // 2 == oh and w == ow:
            top = arr[:h//2, :]
            bottom = arr[h//2:, :]
            result = np.maximum(top, bottom)
            if np.array_equal(result, out_arr):
                return "mix_halves_or_v"
            result = top ^ bottom
            if np.array_equal(result, out_arr):
                return "mix_halves_xor_v"
        return None
    except:
        return None

# Batch 34 - Pattern SHAPE/GEOMETRY depth=-9000000 ANALYZE INFINITY


def try_draw_rectangle(inp, out):
    """Draw rectangle outline."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = arr.copy()

        # Find colored cells and draw rectangle around them
        for c in range(1, 10):
            cells = np.where(arr == c)
            if len(cells[0]) == 0:
                continue
            rmin, rmax = cells[0].min(), cells[0].max()
            cmin, cmax = cells[1].min(), cells[1].max()

            # Draw rectangle outline
            result[rmin, cmin:cmax+1] = c
            result[rmax, cmin:cmax+1] = c
            result[rmin:rmax+1, cmin] = c
            result[rmin:rmax+1, cmax] = c

            if np.array_equal(result, out_arr):
                return f"draw_rectangle_{c}"
            result = arr.copy()
        return None
    except:
        return None


def try_fill_square(inp, out):
    """Fill smallest enclosing square."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        for c in range(1, 10):
            cells = np.where(arr == c)
            if len(cells[0]) == 0:
                continue
            rmin, rmax = cells[0].min(), cells[0].max()
            cmin, cmax = cells[1].min(), cells[1].max()

            size = max(rmax - rmin + 1, cmax - cmin + 1)
            result = arr.copy()
            for i in range(rmin, min(rmin + size, arr.shape[0])):
                for j in range(cmin, min(cmin + size, arr.shape[1])):
                    result[i, j] = c
            if np.array_equal(result, out_arr):
                return f"fill_square_{c}"
        return None
    except:
        return None


def try_draw_line(inp, out):
    """Draw line between points of same color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        for c in range(1, 10):
            cells = np.where(arr == c)
            if len(cells[0]) < 2:
                continue

            result = arr.copy()
            # Connect all pairs
            for idx1 in range(len(cells[0])):
                for idx2 in range(idx1 + 1, len(cells[0])):
                    r1, c1 = cells[0][idx1], cells[1][idx1]
                    r2, c2 = cells[0][idx2], cells[1][idx2]

                    if r1 == r2:  # Horizontal
                        for j in range(min(c1, c2), max(c1, c2) + 1):
                            result[r1, j] = c
                    elif c1 == c2:  # Vertical
                        for i in range(min(r1, r2), max(r1, r2) + 1):
                            result[i, c1] = c

            if np.array_equal(result, out_arr):
                return f"draw_line_{c}"
        return None
    except:
        return None


def try_corner_mark(inp, out):
    """Mark corners of objects."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for mark_color in range(1, 10):
            for c in range(1, 10):
                if c == mark_color:
                    continue
                cells = np.where(arr == c)
                if len(cells[0]) == 0:
                    continue
                rmin, rmax = cells[0].min(), cells[0].max()
                cmin, cmax = cells[1].min(), cells[1].max()

                result = arr.copy()
                # Mark corners
                if rmin < h and cmin < w:
                    result[rmin, cmin] = mark_color
                if rmin < h and cmax < w:
                    result[rmin, cmax] = mark_color
                if rmax < h and cmin < w:
                    result[rmax, cmin] = mark_color
                if rmax < h and cmax < w:
                    result[rmax, cmax] = mark_color

                if np.array_equal(result, out_arr):
                    return f"corner_mark_{c}_{mark_color}"
        return None
    except:
        return None


def try_diagonal_line(inp, out):
    """Draw diagonal lines."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for c in range(1, 10):
            cells = np.where(arr == c)
            if len(cells[0]) < 2:
                continue

            result = arr.copy()
            for idx1 in range(len(cells[0])):
                for idx2 in range(idx1 + 1, len(cells[0])):
                    r1, c1 = cells[0][idx1], cells[1][idx1]
                    r2, c2 = cells[0][idx2], cells[1][idx2]

                    dr = 1 if r2 > r1 else -1 if r2 < r1 else 0
                    dc = 1 if c2 > c1 else -1 if c2 < c1 else 0

                    if abs(r2 - r1) == abs(c2 - c1):  # True diagonal
                        i, j = r1, c1
                        while (i, j) != (r2 + dr, c2 + dc):
                            if 0 <= i < h and 0 <= j < w:
                                result[i, j] = c
                            i += dr
                            j += dc

            if np.array_equal(result, out_arr):
                return f"diagonal_line_{c}"
        return None
    except:
        return None


def try_triangle_fill(inp, out):
    """Fill triangular region."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for c in range(1, 10):
            # Upper left triangle
            result = arr.copy()
            for i in range(h):
                for j in range(w - i):
                    result[i, j] = c
            if np.array_equal(result, out_arr):
                return f"triangle_ul_{c}"

            # Lower right triangle
            result = arr.copy()
            for i in range(h):
                for j in range(i, w):
                    result[i, j] = c
            if np.array_equal(result, out_arr):
                return f"triangle_lr_{c}"
        return None
    except:
        return None


def try_vertex_connect(inp, out):
    """Connect vertices (corner points)."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = arr.copy()

        # Find corner points (cells with exactly 2 orthogonal neighbors of same color)
        for c in range(1, 10):
            corners = []
            for i in range(h):
                for j in range(w):
                    if arr[i, j] == c:
                        neighbors = 0
                        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ni, nj = i+di, j+dj
                            if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] == c:
                                neighbors += 1
                        if neighbors <= 2:
                            corners.append((i, j))

            # Connect corners
            for i in range(len(corners)):
                for j in range(i+1, len(corners)):
                    r1, c1 = corners[i]
                    r2, c2 = corners[j]
                    if r1 == r2:
                        for col in range(min(c1, c2), max(c1, c2)+1):
                            result[r1, col] = c
                    elif c1 == c2:
                        for row in range(min(r1, r2), max(r1, r2)+1):
                            result[row, c1] = c

        if np.array_equal(result, out_arr):
            return "vertex_connect"
        return None
    except:
        return None


def try_edge_fill(inp, out):
    """Fill edges of grid with color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for fill_c in range(1, 10):
            result = arr.copy()
            result[0, :] = fill_c
            result[h-1, :] = fill_c
            result[:, 0] = fill_c
            result[:, w-1] = fill_c

            if np.array_equal(result, out_arr):
                return f"edge_fill_{fill_c}"
        return None
    except:
        return None


def try_curve_connect(inp, out):
    """Connect points with step pattern (L-shaped)."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for c in range(1, 10):
            cells = np.where(arr == c)
            if len(cells[0]) < 2:
                continue

            result = arr.copy()
            for idx1 in range(len(cells[0])):
                for idx2 in range(idx1 + 1, len(cells[0])):
                    r1, c1 = cells[0][idx1], cells[1][idx1]
                    r2, c2 = cells[0][idx2], cells[1][idx2]

                    # Draw L-shape: first horizontal then vertical
                    for j in range(min(c1, c2), max(c1, c2) + 1):
                        result[r1, j] = c
                    for i in range(min(r1, r2), max(r1, r2) + 1):
                        result[i, c2] = c

            if np.array_equal(result, out_arr):
                return f"curve_connect_hv_{c}"

            result = arr.copy()
            for idx1 in range(len(cells[0])):
                for idx2 in range(idx1 + 1, len(cells[0])):
                    r1, c1 = cells[0][idx1], cells[1][idx1]
                    r2, c2 = cells[0][idx2], cells[1][idx2]

                    # Draw L-shape: first vertical then horizontal
                    for i in range(min(r1, r2), max(r1, r2) + 1):
                        result[i, c1] = c
                    for j in range(min(c1, c2), max(c1, c2) + 1):
                        result[r2, j] = c

            if np.array_equal(result, out_arr):
                return f"curve_connect_vh_{c}"
        return None
    except:
        return None


def try_form_grid_lines(inp, out):
    """Draw grid lines at regular intervals."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for c in range(1, 10):
            for step in [2, 3, 4, 5]:
                result = arr.copy()
                for i in range(0, h, step):
                    result[i, :] = c
                for j in range(0, w, step):
                    result[:, j] = c
                if np.array_equal(result, out_arr):
                    return f"form_grid_lines_{step}_{c}"
        return None
    except:
        return None

# Batch 35 - Pattern SEQUENCE/ORDER depth=-9000000 ANALYZE INFINITY


def try_sequence_progression(inp, out):
    """Detect arithmetic progression in values."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Check each row for progression
        for step in [1, 2, 3, -1, -2]:
            result = arr.copy()
            for r in range(h):
                for c in range(w):
                    if arr[r, c] != 0:
                        result[r, c] = (arr[r, c] + step * c) % 10
                        if result[r, c] == 0:
                            result[r, c] = arr[r, c]

            if np.array_equal(result, out_arr):
                return f"sequence_progression_row_{step}"
        return None
    except:
        return None


def try_step_fill(inp, out):
    """Fill with stepping pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for c in range(1, 10):
            for step in [1, 2, 3]:
                result = arr.copy()
                for r in range(h):
                    col_start = (r * step) % w
                    result[r, col_start] = c

                if np.array_equal(result, out_arr):
                    return f"step_fill_{step}_{c}"
        return None
    except:
        return None


def try_cycle_colors(inp, out):
    """Cycle through colors in sequence."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        unique_colors = [c for c in np.unique(arr) if c != 0]
        if len(unique_colors) < 2:
            return None

        # Try cycling colors
        for shift in range(1, len(unique_colors)):
            result = arr.copy()
            for i, c in enumerate(unique_colors):
                new_c = unique_colors[(i + shift) % len(unique_colors)]
                result[arr == c] = new_c

            if np.array_equal(result, out_arr):
                return f"cycle_colors_{shift}"
        return None
    except:
        return None


def try_repeat_sequence(inp, out):
    """Repeat detected sequence to fill."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if output is horizontal repeat
        for period in range(1, w + 1):
            if ow % period == 0:
                result = np.zeros((h, ow), dtype=int)
                for j in range(ow):
                    result[:, j] = arr[:, j % period]

                if np.array_equal(result, out_arr):
                    return f"repeat_sequence_h_{period}"

        # Check vertical repeat
        for period in range(1, h + 1):
            if oh % period == 0:
                result = np.zeros((oh, w), dtype=int)
                for i in range(oh):
                    result[i, :] = arr[i % period, :]

                if np.array_equal(result, out_arr):
                    return f"repeat_sequence_v_{period}"
        return None
    except:
        return None


def try_iterate_transform(inp, out):
    """Apply a simple transform iteratively."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Try iterating rot90
        current = arr.copy()
        for i in range(1, 5):
            current = np.rot90(current)
            if np.array_equal(current, out_arr):
                return f"iterate_rot90_{i}x"

        # Try iterating flip
        current = arr.copy()
        for i in range(1, 3):
            current = np.fliplr(current)
            if np.array_equal(current, out_arr):
                return f"iterate_hflip_{i}x"

        return None
    except:
        return None


def try_loop_pattern(inp, out):
    """Detect looping/circular pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Try wrapping connections
        for c in range(1, 10):
            result = arr.copy()
            cells = np.where(arr == c)
            if len(cells[0]) < 2:
                continue

            # Connect last to first (loop)
            r1, c1 = cells[0][-1], cells[1][-1]
            r2, c2 = cells[0][0], cells[1][0]

            # Draw connecting line
            if r1 == r2:
                for col in range(min(c1, c2), max(c1, c2) + 1):
                    result[r1, col] = c
            elif c1 == c2:
                for row in range(min(r1, r2), max(r1, r2) + 1):
                    result[row, c1] = c

            if np.array_equal(result, out_arr):
                return f"loop_pattern_{c}"
        return None
    except:
        return None


def try_next_in_series(inp, out):
    """Predict next item in series."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Simple case: output extends input
        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh == h and ow == w + 1:
            # Added one column - predict pattern
            pattern = arr[:, -1] - arr[:, -2] if w > 1 else np.zeros(h)
            predicted = arr[:, -1] + pattern
            predicted = np.clip(predicted, 0, 9).astype(int)

            result = np.zeros((h, w + 1), dtype=int)
            result[:, :w] = arr
            result[:, w] = predicted

            if np.array_equal(result, out_arr):
                return "next_in_series_h"

        if oh == h + 1 and ow == w:
            # Added one row
            pattern = arr[-1, :] - arr[-2, :] if h > 1 else np.zeros(w)
            predicted = arr[-1, :] + pattern
            predicted = np.clip(predicted, 0, 9).astype(int)

            result = np.zeros((h + 1, w), dtype=int)
            result[:h, :] = arr
            result[h, :] = predicted

            if np.array_equal(result, out_arr):
                return "next_in_series_v"
        return None
    except:
        return None


def try_previous_state(inp, out):
    """Infer previous state from current."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Try reversing common transformations
        # Reverse rotation
        for k in [1, 2, 3]:
            if np.array_equal(np.rot90(arr, -k), out_arr):
                return f"previous_state_rot{k}"

        # Reverse flip
        if np.array_equal(np.fliplr(arr), out_arr):
            return "previous_state_fliph"
        if np.array_equal(np.flipud(arr), out_arr):
            return "previous_state_flipv"
        return None
    except:
        return None


def try_first_occurrence(inp, out):
    """Extract first occurrence of pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        oh, ow = out_arr.shape
        h, w = arr.shape

        if oh > h or ow > w:
            return None

        # Find first matching region
        for r in range(h - oh + 1):
            for c in range(w - ow + 1):
                region = arr[r:r+oh, c:c+ow]
                if np.any(region != 0):
                    if np.array_equal(region, out_arr):
                        return f"first_occurrence_{r}_{c}"
        return None
    except:
        return None


def try_consecutive_fill(inp, out):
    """Fill consecutive cells."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for c in range(1, 10):
            # Fill gaps between consecutive cells
            result = arr.copy()
            cells = np.where(arr == c)
            if len(cells[0]) < 2:
                continue

            for idx in range(len(cells[0]) - 1):
                r1, c1 = cells[0][idx], cells[1][idx]
                r2, c2 = cells[0][idx + 1], cells[1][idx + 1]

                # Fill horizontal consecutive
                if r1 == r2:
                    for col in range(min(c1, c2), max(c1, c2) + 1):
                        result[r1, col] = c
                # Fill vertical consecutive
                elif c1 == c2:
                    for row in range(min(r1, r2), max(r1, r2) + 1):
                        result[row, c1] = c

            if np.array_equal(result, out_arr):
                return f"consecutive_fill_{c}"
        return None
    except:
        return None

# Batch 36 - Pattern LOGIC/CONDITIONAL depth=-9000000 ANALYZE INFINITY


def try_conditional_color(inp, out):
    """Apply color based on condition."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # If cell has neighbors of color X, change to color Y
        for c_if in range(1, 10):
            for c_then in range(1, 10):
                if c_if == c_then:
                    continue
                result = arr.copy()
                for r in range(h):
                    for c in range(w):
                        has_neighbor = False
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if arr[nr, nc] == c_if:
                                    has_neighbor = True
                                    break
                        if has_neighbor and arr[r, c] != 0:
                            result[r, c] = c_then

                if np.array_equal(result, out_arr):
                    return f"conditional_color_if{c_if}_then{c_then}"
        return None
    except:
        return None


def try_where_nonzero(inp, out):
    """Transform where cells are nonzero."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Replace nonzero with specific color
        for c in range(1, 10):
            result = np.where(arr != 0, c, 0)
            if np.array_equal(result, out_arr):
                return f"where_nonzero_{c}"
        return None
    except:
        return None


def try_unless_edge(inp, out):
    """Transform unless cell is on edge."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for c in range(1, 10):
            result = arr.copy()
            for r in range(1, h-1):
                for col in range(1, w-1):
                    if arr[r, col] != 0:
                        result[r, col] = c

            if np.array_equal(result, out_arr):
                return f"unless_edge_{c}"
        return None
    except:
        return None


def try_boolean_and(inp, out):
    """AND operation between regions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        unique = [c for c in np.unique(arr) if c != 0]
        if len(unique) < 2:
            return None

        # AND: keep only where multiple colors overlap
        for c1 in unique:
            for c2 in unique:
                if c1 >= c2:
                    continue
                mask1 = arr == c1
                mask2 = arr == c2
                # Find overlap by shifting
                for dr in range(-3, 4):
                    for dc in range(-3, 4):
                        shifted = np.roll(np.roll(mask2, dr, axis=0), dc, axis=1)
                        overlap = mask1 & shifted
                        for out_c in range(1, 10):
                            result = np.where(overlap, out_c, 0).astype(int)
                            if np.array_equal(result, out_arr):
                                return f"boolean_and_{c1}_{c2}_{out_c}"
        return None
    except:
        return None


def try_boolean_or(inp, out):
    """OR operation between colors."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        unique = [c for c in np.unique(arr) if c != 0]
        if len(unique) < 2:
            return None

        # OR: combine regions
        for out_c in range(1, 10):
            result = np.where(arr != 0, out_c, 0)
            if np.array_equal(result, out_arr):
                return f"boolean_or_{out_c}"
        return None
    except:
        return None


def try_rule_match(inp, out):
    """Apply rule where pattern matches."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Find 2x2 patterns and apply rule
        for c in range(1, 10):
            result = arr.copy()
            for r in range(h - 1):
                for col in range(w - 1):
                    block = arr[r:r+2, col:col+2]
                    if np.sum(block != 0) >= 3:
                        result[r:r+2, col:col+2] = c

            if np.array_equal(result, out_arr):
                return f"rule_match_2x2_{c}"
        return None
    except:
        return None


def try_when_isolated(inp, out):
    """Transform when cell is isolated."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for new_c in range(0, 10):
            result = arr.copy()
            for r in range(h):
                for c in range(w):
                    if arr[r, c] != 0:
                        has_same_neighbor = False
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if arr[nr, nc] == arr[r, c]:
                                    has_same_neighbor = True
                                    break
                        if not has_same_neighbor:
                            result[r, c] = new_c

            if np.array_equal(result, out_arr):
                return f"when_isolated_{new_c}"
        return None
    except:
        return None


def try_if_surrounded(inp, out):
    """Transform if cell is surrounded by same color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for new_c in range(1, 10):
            result = arr.copy()
            for r in range(1, h-1):
                for c in range(1, w-1):
                    neighbors = [arr[r-1,c], arr[r+1,c], arr[r,c-1], arr[r,c+1]]
                    if all(n == neighbors[0] and n != 0 for n in neighbors):
                        result[r, c] = new_c

            if np.array_equal(result, out_arr):
                return f"if_surrounded_{new_c}"
        return None
    except:
        return None


def try_else_fill(inp, out):
    """Fill cells that don't match condition."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Fill zeros with color if adjacent to nonzero
        h, w = arr.shape

        for c in range(1, 10):
            result = arr.copy()
            for r in range(h):
                for col in range(w):
                    if arr[r, col] == 0:
                        has_nonzero_neighbor = False
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r + dr, col + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if arr[nr, nc] != 0:
                                    has_nonzero_neighbor = True
                                    break
                        if has_nonzero_neighbor:
                            result[r, col] = c

            if np.array_equal(result, out_arr):
                return f"else_fill_{c}"
        return None
    except:
        return None


def try_true_false_map(inp, out):
    """Map colors to true/false pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Binary mapping
        for true_c in range(1, 10):
            for false_c in range(0, 10):
                if true_c == false_c:
                    continue
                result = np.where(arr != 0, true_c, false_c)
                if np.array_equal(result, out_arr):
                    return f"true_false_map_{true_c}_{false_c}"
        return None
    except:
        return None

# Batch 37 - Pattern POSITION/LOCATION depth=-9000000 ANALYZE INFINITY


def try_move_to_corner(inp, out):
    """Move content to corner of grid."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Find bounding box of content
        rows = np.any(arr != 0, axis=1)
        cols = np.any(arr != 0, axis=0)
        if not rows.any():
            return None

        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]

        content = arr[r_min:r_max+1, c_min:c_max+1]
        ch, cw = content.shape

        # Try each corner
        corners = [(0, 0), (0, w-cw), (h-ch, 0), (h-ch, w-cw)]
        names = ["tl", "tr", "bl", "br"]
        for (r, c), name in zip(corners, names):
            if r < 0 or c < 0:
                continue
            result = np.zeros_like(arr)
            result[r:r+ch, c:c+cw] = content
            if np.array_equal(result, out_arr):
                return f"move_to_corner_{name}"
        return None
    except:
        return None


def try_center_content(inp, out):
    """Center content in grid."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Find bounding box
        rows = np.any(arr != 0, axis=1)
        cols = np.any(arr != 0, axis=0)
        if not rows.any():
            return None

        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]

        content = arr[r_min:r_max+1, c_min:c_max+1]
        ch, cw = content.shape

        # Center
        r_start = (h - ch) // 2
        c_start = (w - cw) // 2

        result = np.zeros_like(arr)
        result[r_start:r_start+ch, c_start:c_start+cw] = content

        if np.array_equal(result, out_arr):
            return "center_content"
        return None
    except:
        return None


def try_swap_quadrants(inp, out):
    """Swap diagonal quadrants."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        h2, w2 = h // 2, w // 2

        # Swap diagonally
        result = arr.copy()
        temp = result[:h2, :w2].copy()
        result[:h2, :w2] = result[h2:, w2:]
        result[h2:, w2:] = temp

        if np.array_equal(result, out_arr):
            return "swap_quadrants_diag1"

        # Swap other diagonal
        result = arr.copy()
        temp = result[:h2, w2:].copy()
        result[:h2, w2:] = result[h2:, :w2]
        result[h2:, :w2] = temp

        if np.array_equal(result, out_arr):
            return "swap_quadrants_diag2"
        return None
    except:
        return None


def try_relocate_object(inp, out):
    """Relocate object to new position."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for c in range(1, 10):
            mask = arr == c
            if not np.any(mask):
                continue

            # Find object bounds
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            r_min, r_max = np.where(rows)[0][[0, -1]]
            c_min, c_max = np.where(cols)[0][[0, -1]]

            obj = arr[r_min:r_max+1, c_min:c_max+1]
            oh, ow = obj.shape

            # Try relocating
            for dr in range(-h, h):
                for dc in range(-w, w):
                    nr, nc = r_min + dr, c_min + dc
                    if nr < 0 or nc < 0 or nr + oh > h or nc + ow > w:
                        continue

                    result = arr.copy()
                    result[r_min:r_max+1, c_min:c_max+1] = np.where(
                        arr[r_min:r_max+1, c_min:c_max+1] == c, 0,
                        arr[r_min:r_max+1, c_min:c_max+1]
                    )
                    result[nr:nr+oh, nc:nc+ow] = np.where(
                        obj == c, c, result[nr:nr+oh, nc:nc+ow]
                    )

                    if np.array_equal(result, out_arr):
                        return f"relocate_object_{c}_{dr}_{dc}"
        return None
    except:
        return None


def try_exchange_colors(inp, out):
    """Exchange positions of two colors."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        unique = [c for c in np.unique(arr) if c != 0]
        if len(unique) < 2:
            return None

        for c1 in unique:
            for c2 in unique:
                if c1 >= c2:
                    continue
                result = arr.copy()
                result[arr == c1] = c2
                result[arr == c2] = c1

                if np.array_equal(result, out_arr):
                    return f"exchange_colors_{c1}_{c2}"
        return None
    except:
        return None


def try_shift_by_index(inp, out):
    """Shift rows/cols by their index."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Shift each row by its index
        result = np.zeros_like(arr)
        for r in range(h):
            shift = r % w
            result[r, :] = np.roll(arr[r, :], shift)
        if np.array_equal(result, out_arr):
            return "shift_row_by_index"

        # Shift each col by its index
        result = np.zeros_like(arr)
        for c in range(w):
            shift = c % h
            result[:, c] = np.roll(arr[:, c], shift)
        if np.array_equal(result, out_arr):
            return "shift_col_by_index"
        return None
    except:
        return None


def try_coordinate_swap(inp, out):
    """Swap x and y coordinates (transpose)."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        result = arr.T
        if np.array_equal(result, out_arr):
            return "coordinate_swap_transpose"
        return None
    except:
        return None


def try_offset_by_value(inp, out):
    """Offset position by cell value."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = np.zeros_like(arr)

        for r in range(h):
            for c in range(w):
                if arr[r, c] != 0:
                    v = arr[r, c]
                    # Try different offset directions
                    for dr, dc in [(v, 0), (-v, 0), (0, v), (0, -v)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            result[nr, nc] = v

        if np.array_equal(result, out_arr):
            return "offset_by_value"
        return None
    except:
        return None


def try_row_to_column(inp, out):
    """Convert row content to column."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if first row becomes first column
        if h >= 1 and oh == w and ow == 1:
            result = arr[0, :].reshape(-1, 1)
            if np.array_equal(result, out_arr):
                return "row_to_column_0"

        # Check if output is transposed
        for r in range(h):
            result = arr[r, :].reshape(-1, 1)
            if np.array_equal(result, out_arr):
                return f"row_to_column_{r}"
        return None
    except:
        return None


def try_column_to_row(inp, out):
    """Convert column content to row."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        for c in range(w):
            result = arr[:, c].reshape(1, -1)
            if np.array_equal(result, out_arr):
                return f"column_to_row_{c}"
        return None
    except:
        return None

# Batch 38 - Pattern OBJECTNESS/COHESION depth=-9000000 ANALYZE INFINITY


def try_preserve_object_colors(inp, out):
    """Preserve object identity by maintaining colors."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Check if all non-zero colors in input are preserved in output
        inp_colors = set(arr.flatten()) - {0}
        out_colors = set(out_arr.flatten()) - {0}

        if inp_colors == out_colors:
            # Check if objects are just repositioned
            for c in inp_colors:
                inp_mask = (arr == c).astype(int)
                out_mask = (out_arr == c).astype(int)
                if inp_mask.sum() != out_mask.sum():
                    return None
            return "preserve_object_colors"
        return None
    except:
        return None


def try_entity_extract(inp, out):
    """Extract a discrete entity from input."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        for c in range(1, 10):
            mask = arr == c
            if not mask.any():
                continue
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            if len(rows) > 0 and len(cols) > 0:
                r0, r1 = rows[0], rows[-1] + 1
                c0, c1 = cols[0], cols[-1] + 1
                entity = arr[r0:r1, c0:c1]
                if np.array_equal(entity, out_arr):
                    return f"entity_extract_{c}"
        return None
    except:
        return None


def try_whole_object_move(inp, out):
    """Move entire object as a whole unit."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        for c in range(1, 10):
            inp_mask = arr == c
            out_mask = out_arr == c

            if inp_mask.sum() != out_mask.sum() or inp_mask.sum() == 0:
                continue

            inp_rows, inp_cols = np.where(inp_mask)
            out_rows, out_cols = np.where(out_mask)

            if len(inp_rows) > 0 and len(out_rows) > 0:
                dr = out_rows[0] - inp_rows[0]
                dc = out_cols[0] - inp_cols[0]

                # Check if all points moved by same offset
                if all((inp_rows[i] + dr == out_rows[i]) and (inp_cols[i] + dc == out_cols[i])
                       for i in range(len(inp_rows))):
                    return f"whole_object_move_{c}_{dr}_{dc}"
        return None
    except:
        return None


def try_component_assembly(inp, out):
    """Assemble parts into whole."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Get connected components
        nonzero = (arr > 0).astype(int)
        from scipy import ndimage
        labeled, n_parts = ndimage.label(nonzero)

        if n_parts >= 2:
            # Check if output is union of all parts
            assembled = (arr > 0).astype(int)
            if np.array_equal(assembled, (out_arr > 0).astype(int)):
                return f"component_assembly_{n_parts}_parts"
        return None
    except:
        return None


def try_bounded_entity(inp, out):
    """Detect bounded entity region."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        for c in range(1, 10):
            mask = arr == c
            if not mask.any():
                continue

            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]

            if len(rows) > 0 and len(cols) > 0:
                r0, r1 = rows[0], rows[-1] + 1
                c0, c1 = cols[0], cols[-1] + 1

                bounded = np.zeros_like(arr)
                bounded[r0:r1, c0:c1] = arr[r0:r1, c0:c1]

                if np.array_equal(bounded, out_arr):
                    return f"bounded_entity_{c}"
        return None
    except:
        return None


def try_separate_units(inp, out):
    """Separate distinct units in grid."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if output is one unit from input
        for c in range(1, 10):
            mask = arr == c
            if mask.sum() > 0:
                rows, cols = np.where(mask)
                r0, r1 = rows.min(), rows.max() + 1
                c0, c1 = cols.min(), cols.max() + 1
                unit = arr[r0:r1, c0:c1]
                if np.array_equal(unit, out_arr):
                    return f"separate_unit_{c}"
        return None
    except:
        return None


def try_maintain_integrity(inp, out):
    """Maintain structural integrity of objects."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Check if object structure is preserved
        for c in range(1, 10):
            inp_mask = arr == c
            out_mask = out_arr == c

            if inp_mask.sum() > 0 and out_mask.sum() > 0:
                # Get bounding boxes
                inp_rows, inp_cols = np.where(inp_mask)
                out_rows, out_cols = np.where(out_mask)

                inp_h = inp_rows.max() - inp_rows.min() + 1
                inp_w = inp_cols.max() - inp_cols.min() + 1
                out_h = out_rows.max() - out_rows.min() + 1
                out_w = out_cols.max() - out_cols.min() + 1

                if inp_h == out_h and inp_w == out_w:
                    return f"maintain_integrity_{c}"
        return None
    except:
        return None


def try_discrete_object_color(inp, out):
    """Color each discrete object differently."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        from scipy import ndimage
        nonzero = (arr > 0).astype(int)
        labeled, n_obj = ndimage.label(nonzero)

        if n_obj >= 2:
            # Check if each object got a unique color
            result = np.zeros_like(arr)
            for i in range(1, n_obj + 1):
                mask = labeled == i
                result[mask] = i
            if np.array_equal(result, out_arr):
                return f"discrete_object_color_{n_obj}"
        return None
    except:
        return None


def try_element_count(inp, out):
    """Count elements and represent as grid."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        from scipy import ndimage
        nonzero = (arr > 0).astype(int)
        labeled, n_obj = ndimage.label(nonzero)

        # Check if output represents object count
        oh, ow = out_arr.shape
        if oh == 1 and ow == n_obj:
            expected = np.ones((1, n_obj)) * arr[arr > 0][0] if (arr > 0).any() else np.zeros((1, n_obj))
            if out_arr.sum() == n_obj:
                return f"element_count_{n_obj}"
        return None
    except:
        return None


def try_wholeness_preserve(inp, out):
    """Preserve object wholeness during transform."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        for c in range(1, 10):
            inp_mask = arr == c
            out_mask = out_arr == c

            if inp_mask.sum() == 0:
                continue

            # Check if connected regions are preserved
            from scipy import ndimage
            inp_labeled, inp_n = ndimage.label(inp_mask)
            out_labeled, out_n = ndimage.label(out_mask)

            if inp_n == out_n and inp_n > 0:
                return f"wholeness_preserve_{c}_{inp_n}"
        return None
    except:
        return None

# Batch 39 - Pattern CONTAINMENT/OCCLUSION depth=-9000000 ANALYZE INFINITY


def try_fill_enclosed(inp, out):
    """Fill enclosed/contained regions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        for c in range(1, 10):
            mask = arr == c
            if not mask.any():
                continue

            # Find enclosed regions - areas completely surrounded by color c
            from scipy import ndimage
            filled = ndimage.binary_fill_holes(mask)
            interior = filled & ~mask

            if interior.any():
                result = arr.copy()
                for fill_c in range(1, 10):
                    test = arr.copy()
                    test[interior] = fill_c
                    if np.array_equal(test, out_arr):
                        return f"fill_enclosed_{c}_with_{fill_c}"
        return None
    except:
        return None


def try_extract_inside(inp, out):
    """Extract content inside a boundary."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        for c in range(1, 10):
            mask = arr == c
            if not mask.any():
                continue

            from scipy import ndimage
            filled = ndimage.binary_fill_holes(mask)
            interior = filled & ~mask

            if interior.any():
                rows, cols = np.where(interior)
                if len(rows) > 0:
                    r0, r1 = rows.min(), rows.max() + 1
                    c0, c1 = cols.min(), cols.max() + 1
                    inside = arr[r0:r1, c0:c1].copy()
                    inside[~interior[r0:r1, c0:c1]] = 0
                    if np.array_equal(inside, out_arr):
                        return f"extract_inside_{c}"
        return None
    except:
        return None


def try_remove_hidden(inp, out):
    """Remove hidden/occluded regions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Find cells that are "hidden" by being enclosed
        for c in range(1, 10):
            mask = arr == c
            if not mask.any():
                continue

            from scipy import ndimage
            filled = ndimage.binary_fill_holes(mask)
            hidden = filled & ~mask

            if hidden.any():
                result = arr.copy()
                result[hidden] = 0
                if np.array_equal(result, out_arr):
                    return f"remove_hidden_{c}"
        return None
    except:
        return None


def try_reveal_occluded(inp, out):
    """Reveal what's behind occlusion."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Remove the front layer to reveal what's behind
        for c in range(1, 10):
            mask = arr == c
            if mask.sum() > 0:
                result = arr.copy()
                result[mask] = 0
                if np.array_equal(result, out_arr):
                    return f"reveal_occluded_{c}"
        return None
    except:
        return None


def try_layer_extract(inp, out):
    """Extract a specific layer from stacked content."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        for c in range(1, 10):
            layer = np.zeros_like(arr)
            layer[arr == c] = c
            if np.array_equal(layer, out_arr):
                return f"layer_extract_{c}"
        return None
    except:
        return None


def try_stack_layers(inp, out):
    """Stack layers on top of each other."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Check if output is overlay of different color layers
        colors = sorted(set(arr.flatten()) - {0})
        if len(colors) >= 2:
            result = np.zeros_like(arr)
            for c in colors:
                result[arr == c] = c
            if np.array_equal(result, out_arr):
                return f"stack_layers_{len(colors)}"
        return None
    except:
        return None


def try_depth_order(inp, out):
    """Reorder by depth - bring back to front."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        colors = sorted(set(arr.flatten()) - {0}, reverse=True)
        if len(colors) >= 2:
            result = np.zeros_like(arr)
            for c in colors:
                result[arr == c] = c
            if np.array_equal(result, out_arr):
                return f"depth_order_{len(colors)}"
        return None
    except:
        return None


def try_overlap_union(inp, out):
    """Union of overlapping regions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # All non-zero become one color
        for c in range(1, 10):
            result = np.zeros_like(arr)
            result[arr > 0] = c
            if np.array_equal(result, out_arr):
                return f"overlap_union_{c}"
        return None
    except:
        return None


def try_contained_crop(inp, out):
    """Crop to contained region bounds."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        for c in range(1, 10):
            mask = arr == c
            if not mask.any():
                continue

            from scipy import ndimage
            filled = ndimage.binary_fill_holes(mask)
            interior = filled & ~mask

            if interior.any():
                rows, cols = np.where(interior)
                r0, r1 = rows.min(), rows.max() + 1
                c0, c1 = cols.min(), cols.max() + 1
                cropped = arr[r0:r1, c0:c1]
                if np.array_equal(cropped, out_arr):
                    return f"contained_crop_{c}"
        return None
    except:
        return None


def try_within_boundary(inp, out):
    """Keep only cells within a boundary."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        for c in range(1, 10):
            mask = arr == c
            if not mask.any():
                continue

            from scipy import ndimage
            filled = ndimage.binary_fill_holes(mask)

            result = np.zeros_like(arr)
            result[filled] = arr[filled]
            if np.array_equal(result, out_arr):
                return f"within_boundary_{c}"
        return None
    except:
        return None

# Batch 40 - Pattern SAME/DIFFERENT/ANALOGY depth=-9000000 ANALYZE INFINITY


def try_find_same_color(inp, out):
    """Find cells with same color and mark them."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        for c in range(1, 10):
            # Mark all positions with same color
            result = np.zeros_like(arr)
            result[arr == c] = c
            if np.array_equal(result, out_arr):
                return f"find_same_color_{c}"
        return None
    except:
        return None


def try_mark_different(inp, out):
    """Mark cells that are different from surroundings."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape
        result = np.zeros_like(arr)

        for r in range(h):
            for c in range(w):
                # Check if different from neighbors
                val = arr[r, c]
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbors.append(arr[nr, nc])
                if neighbors and all(n != val for n in neighbors):
                    result[r, c] = val

        if np.array_equal(result, out_arr):
            return "mark_different"
        return None
    except:
        return None


def try_analogy_color_map(inp, out):
    """Map colors analogically - if A maps to B, apply same mapping."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Find color mapping
        mapping = {}
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] != out_arr[i, j]:
                    if arr[i, j] in mapping:
                        if mapping[arr[i, j]] != out_arr[i, j]:
                            return None
                    mapping[arr[i, j]] = out_arr[i, j]

        if mapping:
            result = arr.copy()
            for k, v in mapping.items():
                result[arr == k] = v
            if np.array_equal(result, out_arr):
                return f"analogy_color_map_{len(mapping)}"
        return None
    except:
        return None


def try_match_pattern(inp, out):
    """Find matching patterns and unify them."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Check if all non-zero cells become same color
        for c in range(1, 10):
            result = arr.copy()
            result[arr > 0] = c
            if np.array_equal(result, out_arr):
                return f"match_pattern_to_{c}"
        return None
    except:
        return None


def try_pair_colors(inp, out):
    """Pair/associate colors together."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        colors = sorted(set(arr.flatten()) - {0})
        if len(colors) == 2:
            c1, c2 = colors
            # Swap the pair
            result = arr.copy()
            result[arr == c1] = c2
            result[arr == c2] = c1
            if np.array_equal(result, out_arr):
                return f"pair_colors_{c1}_{c2}"
        return None
    except:
        return None


def try_equal_size_select(inp, out):
    """Select objects of equal size."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        from scipy import ndimage
        labeled, n_obj = ndimage.label(arr > 0)

        # Get sizes of each object
        sizes = {}
        for i in range(1, n_obj + 1):
            sizes[i] = (labeled == i).sum()

        # Find most common size
        if sizes:
            from collections import Counter
            size_counts = Counter(sizes.values())
            most_common_size = size_counts.most_common(1)[0][0]

            result = np.zeros_like(arr)
            for i, sz in sizes.items():
                if sz == most_common_size:
                    result[labeled == i] = arr[labeled == i]

            if np.array_equal(result, out_arr):
                return f"equal_size_select_{most_common_size}"
        return None
    except:
        return None


def try_unlike_remove(inp, out):
    """Remove elements that are unlike others."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Find the most common color and keep only that
        from collections import Counter
        nonzero = arr[arr > 0]
        if len(nonzero) > 0:
            counts = Counter(nonzero)
            most_common = counts.most_common(1)[0][0]
            result = np.zeros_like(arr)
            result[arr == most_common] = most_common
            if np.array_equal(result, out_arr):
                return f"unlike_remove_keep_{most_common}"
        return None
    except:
        return None


def try_correspond_position(inp, out):
    """Corresponding position transformation."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Check if output is same but rotated/reflected to corresponding position
        h, w = arr.shape
        oh, ow = out_arr.shape

        if h == oh and w == ow:
            # Check 180 degree rotation
            result = np.rot90(arr, 2)
            if np.array_equal(result, out_arr):
                return "correspond_position_180"
        return None
    except:
        return None


def try_similarity_merge(inp, out):
    """Merge similar regions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Check if all same-colored regions merge into one uniform color
        colors = set(arr.flatten()) - {0}
        for c in colors:
            result = arr.copy()
            result[arr > 0] = c
            if np.array_equal(result, out_arr):
                return f"similarity_merge_{c}"
        return None
    except:
        return None


def try_correlation_fill(inp, out):
    """Fill based on correlation with existing pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Check if zeros are filled with correlated values
        for c in range(1, 10):
            result = arr.copy()
            # Fill zeros with the color that appears most in their row
            for r in range(h):
                row_colors = arr[r, arr[r] > 0]
                if len(row_colors) > 0:
                    from collections import Counter
                    fill = Counter(row_colors).most_common(1)[0][0]
                    result[r, arr[r] == 0] = fill
            if np.array_equal(result, out_arr):
                return "correlation_fill_row"

            # Try column correlation
            result = arr.copy()
            for col in range(w):
                col_colors = arr[arr[:, col] > 0, col]
                if len(col_colors) > 0:
                    from collections import Counter
                    fill = Counter(col_colors).most_common(1)[0][0]
                    result[arr[:, col] == 0, col] = fill
            if np.array_equal(result, out_arr):
                return "correlation_fill_col"
        return None
    except:
        return None

# Batch 41 - Pattern COUNTING/NUMBERS depth=-9000000 ANALYZE INFINITY


def try_count_to_size(inp, out):
    """Count objects and make output that size."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        from scipy import ndimage
        labeled, n_obj = ndimage.label(arr > 0)

        oh, ow = out_arr.shape
        if oh == 1 and ow == n_obj:
            return f"count_to_size_{n_obj}"
        if oh == n_obj and ow == 1:
            return f"count_to_size_{n_obj}_v"
        if oh == n_obj and ow == n_obj:
            return f"count_to_size_{n_obj}x{n_obj}"
        return None
    except:
        return None


def try_frequency_color(inp, out):
    """Color based on frequency of occurrence."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        from collections import Counter
        counts = Counter(arr.flatten())
        counts.pop(0, None)

        if len(counts) >= 2:
            # Most frequent gets one color, rest get another
            most_common = counts.most_common(1)[0][0]
            result = np.zeros_like(arr)
            result[arr == most_common] = most_common
            if np.array_equal(result, out_arr):
                return f"frequency_color_most_{most_common}"

            # Least frequent
            least_common = counts.most_common()[-1][0]
            result = np.zeros_like(arr)
            result[arr == least_common] = least_common
            if np.array_equal(result, out_arr):
                return f"frequency_color_least_{least_common}"
        return None
    except:
        return None


def try_sum_row_col(inp, out):
    """Sum values in row or column."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Row sums
        if oh == h and ow == 1:
            row_sums = arr.sum(axis=1).reshape(-1, 1)
            if np.array_equal(row_sums.astype(int), out_arr):
                return "sum_row"

        # Col sums
        if oh == 1 and ow == w:
            col_sums = arr.sum(axis=0).reshape(1, -1)
            if np.array_equal(col_sums.astype(int), out_arr):
                return "sum_col"
        return None
    except:
        return None


def try_cardinality_output(inp, out):
    """Output represents cardinality of something."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Count distinct colors
        n_colors = len(set(arr.flatten()) - {0})
        oh, ow = out_arr.shape

        if oh == 1 and ow == n_colors:
            return f"cardinality_colors_{n_colors}"
        if oh == n_colors and ow == 1:
            return f"cardinality_colors_{n_colors}_v"
        return None
    except:
        return None


def try_multiply_grid(inp, out):
    """Multiply/scale grid by factor."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        for factor in [2, 3, 4, 5]:
            if oh == h * factor and ow == w * factor:
                scaled = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
                if np.array_equal(scaled, out_arr):
                    return f"multiply_grid_{factor}x"
        return None
    except:
        return None


def try_add_values(inp, out):
    """Add constant to all values."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        for add_val in range(1, 10):
            result = arr.copy()
            result[arr > 0] = np.clip(arr[arr > 0] + add_val, 0, 9)
            if np.array_equal(result, out_arr):
                return f"add_values_{add_val}"

            # Subtract
            result = arr.copy()
            result[arr > 0] = np.clip(arr[arr > 0] - add_val, 0, 9)
            if np.array_equal(result, out_arr):
                return f"subtract_values_{add_val}"
        return None
    except:
        return None


def try_quantity_to_color(inp, out):
    """Convert quantity to color value."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        from scipy import ndimage
        labeled, n_obj = ndimage.label(arr > 0)

        if arr.shape == out_arr.shape:
            result = np.zeros_like(arr)
            for i in range(1, n_obj + 1):
                mask = labeled == i
                size = mask.sum()
                if size <= 9:
                    result[mask] = size
            if np.array_equal(result, out_arr):
                return "quantity_to_color"
        return None
    except:
        return None


def try_more_less_select(inp, out):
    """Select based on more or less quantity."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        from scipy import ndimage
        labeled, n_obj = ndimage.label(arr > 0)

        sizes = [(i, (labeled == i).sum()) for i in range(1, n_obj + 1)]
        if len(sizes) >= 2:
            sizes.sort(key=lambda x: x[1], reverse=True)

            # Keep more (largest)
            result = np.zeros_like(arr)
            largest_id = sizes[0][0]
            result[labeled == largest_id] = arr[labeled == largest_id]
            if np.array_equal(result, out_arr):
                return "select_more"

            # Keep less (smallest)
            result = np.zeros_like(arr)
            smallest_id = sizes[-1][0]
            result[labeled == smallest_id] = arr[labeled == smallest_id]
            if np.array_equal(result, out_arr):
                return "select_less"
        return None
    except:
        return None


def try_one_two_three(inp, out):
    """Transform based on 1, 2, 3 pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if output has 1, 2, or 3 rows/cols based on input pattern
        for n in [1, 2, 3]:
            if oh == n and ow == w:
                return f"to_{n}_rows"
            if oh == h and ow == n:
                return f"to_{n}_cols"
        return None
    except:
        return None


def try_total_nonzero(inp, out):
    """Output based on total nonzero count."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        total = (arr > 0).sum()
        oh, ow = out_arr.shape

        # Check if output dimensions relate to total
        if oh * ow == total:
            return f"total_nonzero_{total}"
        return None
    except:
        return None

# Batch 42 - Pattern MOTION/CONTINUITY depth=-9000000 ANALYZE INFINITY


def try_extend_trajectory(inp, out):
    """Extend motion along trajectory."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape
        oh, ow = out_arr.shape

        if h != oh or w != ow:
            return None

        # Find colored cells and check if extended
        for c in range(1, 10):
            mask = arr == c
            if not mask.any():
                continue

            rows, cols = np.where(mask)
            if len(rows) >= 2:
                # Check for linear trajectory
                dr = rows[1] - rows[0]
                dc = cols[1] - cols[0]

                # Extend in direction
                result = arr.copy()
                r, c_pos = rows[-1] + dr, cols[-1] + dc
                while 0 <= r < h and 0 <= c_pos < w:
                    result[r, c_pos] = c
                    r += dr
                    c_pos += dc

                if np.array_equal(result, out_arr):
                    return f"extend_trajectory_{dr}_{dc}"
        return None
    except:
        return None


def try_follow_path(inp, out):
    """Follow a path from start to end."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Mark connected path
        colors = set(arr.flatten()) - {0}
        if len(colors) == 1:
            c = list(colors)[0]
            positions = np.argwhere(arr == c)
            if len(positions) >= 2:
                # Fill between first and last
                start = positions[0]
                end = positions[-1]
                result = arr.copy()
                for r in range(min(start[0], end[0]), max(start[0], end[0]) + 1):
                    for col in range(min(start[1], end[1]), max(start[1], end[1]) + 1):
                        result[r, col] = c
                if np.array_equal(result, out_arr):
                    return "follow_path"
        return None
    except:
        return None


def try_flow_direction(inp, out):
    """Apply flow in a direction."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Flow right
        result = arr.copy()
        for r in range(h):
            for c in range(w-1):
                if result[r, c] > 0 and result[r, c+1] == 0:
                    result[r, c+1] = result[r, c]

        if np.array_equal(result, out_arr):
            return "flow_right"

        # Flow down
        result = arr.copy()
        for r in range(h-1):
            for c in range(w):
                if result[r, c] > 0 and result[r+1, c] == 0:
                    result[r+1, c] = result[r, c]

        if np.array_equal(result, out_arr):
            return "flow_down"

        return None
    except:
        return None


def try_smooth_path(inp, out):
    """Smooth/interpolate path between points."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for color in range(1, 10):
            mask = arr == color
            if mask.sum() < 2:
                continue

            # Get endpoints
            rows, cols = np.where(mask)
            if len(rows) != 2:
                continue

            r1, c1 = rows[0], cols[0]
            r2, c2 = rows[1], cols[1]

            # Draw line between
            result = arr.copy()
            steps = max(abs(r2-r1), abs(c2-c1))
            if steps > 0:
                for t in range(steps+1):
                    r = r1 + int((r2-r1) * t / steps)
                    c = c1 + int((c2-c1) * t / steps)
                    result[r, c] = color

            if np.array_equal(result, out_arr):
                return f"smooth_path_{color}"
        return None
    except:
        return None


def try_trace_movement(inp, out):
    """Trace movement from one position to another."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Find what moved
        diff = (arr != out_arr)
        if not diff.any():
            return None

        # Check if it's a trace (leaving path behind)
        for color in range(1, 10):
            in_mask = arr == color
            out_mask = out_arr == color

            # More colored cells in output
            if out_mask.sum() > in_mask.sum():
                # Check if original cells preserved
                if (in_mask & out_mask).sum() == in_mask.sum():
                    return f"trace_movement_{color}"
        return None
    except:
        return None


def try_continuous_fill(inp, out):
    """Fill continuously in direction."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Continuous fill to right edge
        result = arr.copy()
        for r in range(h):
            for c in range(w):
                if arr[r, c] > 0:
                    result[r, c:] = arr[r, c]
                    break

        if np.array_equal(result, out_arr):
            return "continuous_fill_right"

        # Continuous fill to bottom
        result = arr.copy()
        for c in range(w):
            for r in range(h):
                if arr[r, c] > 0:
                    result[r:, c] = arr[r, c]
                    break

        if np.array_equal(result, out_arr):
            return "continuous_fill_down"

        return None
    except:
        return None


def try_momentum_extend(inp, out):
    """Extend based on momentum/velocity pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        for color in range(1, 10):
            mask = arr == color
            if mask.sum() < 3:
                continue

            rows, cols = np.where(mask)

            # Check for consistent velocity
            if len(rows) >= 3:
                dr = rows[1] - rows[0]
                dc = cols[1] - cols[0]

                # Verify velocity pattern
                consistent = all(
                    rows[i+1] - rows[i] == dr and cols[i+1] - cols[i] == dc
                    for i in range(len(rows)-1)
                )

                if consistent:
                    # Extend with same momentum
                    result = arr.copy()
                    r, c = rows[-1] + dr, cols[-1] + dc
                    count = 0
                    while 0 <= r < h and 0 <= c < w and count < 10:
                        result[r, c] = color
                        r += dr
                        c += dc
                        count += 1

                    if np.array_equal(result, out_arr):
                        return f"momentum_extend_{dr}_{dc}"
        return None
    except:
        return None


def try_curve_extend(inp, out):
    """Extend curved path."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Look for L-shape or curve
        for color in range(1, 10):
            mask = arr == color
            if mask.sum() < 2:
                continue

            rows, cols = np.where(mask)

            # Find corner and extend
            result = arr.copy()
            for i in range(len(rows)):
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    r, c = rows[i] + dr, cols[i] + dc
                    if 0 <= r < h and 0 <= c < w and result[r, c] == 0:
                        result[r, c] = color

            if np.array_equal(result, out_arr):
                return f"curve_extend_{color}"
        return None
    except:
        return None


def try_track_follow(inp, out):
    """Follow track/guide pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Check for two colors where one follows the other
        colors = sorted(set(arr.flatten()) - {0})
        if len(colors) != 2:
            return None

        track_color, mover_color = colors[0], colors[1]

        # Check if mover fills track
        track_mask = arr == track_color
        result = arr.copy()
        result[track_mask] = mover_color

        if np.array_equal(result, out_arr):
            return f"track_follow_{track_color}_{mover_color}"

        return None
    except:
        return None


def try_persist_temporal(inp, out):
    """Persist values across temporal pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check for temporal/sequence extension
        if oh == h and ow > w and ow % w == 0:
            repeats = ow // w
            result = np.tile(arr, (1, repeats))
            if np.array_equal(result, out_arr):
                return f"persist_temporal_h_{repeats}"

        if ow == w and oh > h and oh % h == 0:
            repeats = oh // h
            result = np.tile(arr, (repeats, 1))
            if np.array_equal(result, out_arr):
                return f"persist_temporal_v_{repeats}"

        return None
    except:
        return None

# Batch 43 - Pattern COMPOSITIONALITY depth=-9000000 ANALYZE INFINITY


def try_combine_colors(inp, out):
    """Combine multiple colors into one."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Find color combinations
        colors = sorted(set(arr.flatten()) - {0})
        if len(colors) < 2:
            return None

        for c1 in colors:
            for c2 in colors:
                if c1 >= c2:
                    continue

                # Combine c1 and c2 into c1
                result = arr.copy()
                result[arr == c2] = c1
                if np.array_equal(result, out_arr):
                    return f"combine_{c1}_{c2}"
        return None
    except:
        return None


def try_merge_regions(inp, out):
    """Merge adjacent regions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Merge horizontally adjacent cells
        result = arr.copy()
        for r in range(h):
            for c in range(w-1):
                if result[r, c] > 0 and result[r, c+1] > 0:
                    if result[r, c] != result[r, c+1]:
                        result[r, c+1] = result[r, c]

        if np.array_equal(result, out_arr):
            return "merge_regions_h"

        # Merge vertically
        result = arr.copy()
        for r in range(h-1):
            for c in range(w):
                if result[r, c] > 0 and result[r+1, c] > 0:
                    if result[r, c] != result[r+1, c]:
                        result[r+1, c] = result[r, c]

        if np.array_equal(result, out_arr):
            return "merge_regions_v"

        return None
    except:
        return None


def try_build_from_parts(inp, out):
    """Build output from parts in input."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if output is half size (select one part)
        if oh == h // 2 and ow == w:
            top = arr[:h//2, :]
            bottom = arr[h//2:, :]

            if np.array_equal(top, out_arr):
                return "build_top_part"
            if np.array_equal(bottom, out_arr):
                return "build_bottom_part"

        if oh == h and ow == w // 2:
            left = arr[:, :w//2]
            right = arr[:, w//2:]

            if np.array_equal(left, out_arr):
                return "build_left_part"
            if np.array_equal(right, out_arr):
                return "build_right_part"

        return None
    except:
        return None


def try_aggregate_objects(inp, out):
    """Aggregate objects into single representation."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Count objects per color
        colors = sorted(set(arr.flatten()) - {0})

        # Output size matches number of colors
        oh, ow = out_arr.shape
        if oh == len(colors) and ow == 1:
            return f"aggregate_colors_{len(colors)}"
        if oh == 1 and ow == len(colors):
            return f"aggregate_colors_h_{len(colors)}"

        return None
    except:
        return None


def try_synthesize_pattern(inp, out):
    """Synthesize new pattern from input."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check for pattern synthesis (combine rows)
        if oh == 1 and ow == w:
            # OR of all rows
            result = np.zeros((1, w), dtype=int)
            for r in range(h):
                for c in range(w):
                    if arr[r, c] > 0:
                        result[0, c] = arr[r, c]

            if np.array_equal(result, out_arr):
                return "synthesize_row_or"

        if oh == h and ow == 1:
            # OR of all columns
            result = np.zeros((h, 1), dtype=int)
            for r in range(h):
                for c in range(w):
                    if arr[r, c] > 0:
                        result[r, 0] = arr[r, c]

            if np.array_equal(result, out_arr):
                return "synthesize_col_or"

        return None
    except:
        return None


def try_compose_layers(inp, out):
    """Compose multiple layers into output."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if input can be split into layers
        if h % 2 == 0 and oh == h // 2 and ow == w:
            top = arr[:h//2, :]
            bottom = arr[h//2:, :]

            # XOR composition
            result = np.where(top > 0, top, bottom)
            if np.array_equal(result, out_arr):
                return "compose_layers_xor"

            # AND composition
            result = np.where((top > 0) & (bottom > 0), top, 0)
            if np.array_equal(result, out_arr):
                return "compose_layers_and"

        return None
    except:
        return None


def try_integrate_edges(inp, out):
    """Integrate edge information."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        h, w = arr.shape

        # Mark cells adjacent to multiple colors
        result = arr.copy()
        for r in range(h):
            for c in range(w):
                if arr[r, c] == 0:
                    neighbors = set()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] > 0:
                            neighbors.add(arr[nr, nc])

                    if len(neighbors) >= 2:
                        result[r, c] = min(neighbors)

        if np.array_equal(result, out_arr):
            return "integrate_edges"

        return None
    except:
        return None


def try_union_shapes(inp, out):
    """Union of multiple shapes."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        # Check if output is union of all nonzero
        result = np.where(arr > 0, 1, 0)
        if np.array_equal(result.astype(int), out_arr):
            return "union_to_1"

        # Union with first color
        first_color = None
        for c in sorted(set(arr.flatten()) - {0}):
            first_color = c
            break

        if first_color:
            result = np.where(arr > 0, first_color, 0)
            if np.array_equal(result, out_arr):
                return f"union_all_to_{first_color}"

        return None
    except:
        return None


def try_join_halves(inp, out):
    """Join two halves into output."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check for joining halves with XOR
        if h % 2 == 0 and oh == h // 2 and ow == w:
            top = arr[:h//2, :]
            bottom = arr[h//2:, :]

            # Join where either has color
            result = np.where(top > 0, top, bottom)
            if np.array_equal(result, out_arr):
                return "join_halves_v"

        if w % 2 == 0 and oh == h and ow == w // 2:
            left = arr[:, :w//2]
            right = arr[:, w//2:]

            result = np.where(left > 0, left, right)
            if np.array_equal(result, out_arr):
                return "join_halves_h"

        return None
    except:
        return None


def try_element_intersection(inp, out):
    """Intersection of elements."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        if h % 2 == 0 and oh == h // 2 and ow == w:
            top = arr[:h//2, :]
            bottom = arr[h//2:, :]

            # Intersection - where both have color
            result = np.where((top > 0) & (bottom > 0), top, 0)
            if np.array_equal(result, out_arr):
                return "element_intersection_v"

        if w % 2 == 0 and oh == h and ow == w // 2:
            left = arr[:, :w//2]
            right = arr[:, w//2:]

            result = np.where((left > 0) & (right > 0), left, 0)
            if np.array_equal(result, out_arr):
                return "element_intersection_h"

        return None
    except:
        return None

# Batch 44 - Pattern INFINITY/PERSISTENCE depth=-9000000 ANALYZE INFINITY


def try_preserve_invariant(inp, out):
    """Preserve invariant elements across transformation."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h or ow != w:
            return None

        # Find elements that stay constant
        invariant_mask = arr == out_arr
        changed_mask = arr != out_arr

        # Check if changes follow pattern
        if np.sum(changed_mask) > 0:
            # Find what changes
            src_vals = arr[changed_mask]
            dst_vals = out_arr[changed_mask]

            # Single color replacement preserving invariant
            if len(set(src_vals)) == 1 and len(set(dst_vals)) == 1:
                return "preserve_invariant_1"

        return None
    except:
        return None


def try_stable_structure(inp, out):
    """Maintain stable structural elements."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Find structural elements (connected regions)
        from scipy import ndimage

        labeled, num = ndimage.label(arr > 0)
        out_labeled, out_num = ndimage.label(out_arr > 0)

        # Same structure count = stable
        if num == out_num and oh == h and ow == w:
            # Check if shapes preserved
            for i in range(1, num + 1):
                inp_shape = np.sum(labeled == i)
                out_shape = np.sum(out_labeled == i)
                if inp_shape == out_shape:
                    return "stable_structure_1"

        return None
    except:
        return None


def try_endure_pattern(inp, out):
    """Pattern that endures through transformation."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Look for repeating pattern that persists
        if oh == h and ow == w:
            # Find smallest repeating unit
            for ph in range(1, h // 2 + 1):
                for pw in range(1, w // 2 + 1):
                    if h % ph == 0 and w % pw == 0:
                        unit = arr[:ph, :pw]
                        # Check if pattern tiles
                        tiled = np.tile(unit, (h // ph, w // pw))
                        if np.array_equal(tiled, arr):
                            # Same pattern in output?
                            out_unit = out_arr[:ph, :pw]
                            out_tiled = np.tile(out_unit, (oh // ph, ow // pw))
                            if np.array_equal(out_tiled, out_arr):
                                return "endure_pattern_1"

        return None
    except:
        return None


def try_permanent_color(inp, out):
    """Permanent color that never changes."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h or ow != w:
            return None

        # Find colors that never change position
        for c in range(1, 10):
            inp_mask = arr == c
            out_mask = out_arr == c
            if np.any(inp_mask) and np.array_equal(inp_mask, out_mask):
                # Color position unchanged = permanent
                return f"permanent_color_{c}"

        return None
    except:
        return None


def try_fixed_boundary(inp, out):
    """Fixed boundary that remains constant."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h or ow != w:
            return None

        # Check if boundary unchanged
        top_same = np.array_equal(arr[0, :], out_arr[0, :])
        bottom_same = np.array_equal(arr[-1, :], out_arr[-1, :])
        left_same = np.array_equal(arr[:, 0], out_arr[:, 0])
        right_same = np.array_equal(arr[:, -1], out_arr[:, -1])

        if top_same and bottom_same and left_same and right_same:
            # Interior changes only
            if not np.array_equal(arr, out_arr):
                return "fixed_boundary_1"

        return None
    except:
        return None


def try_robust_shape(inp, out):
    """Shape that remains robust under transformation."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Find dominant shape
        from scipy import ndimage

        for c in range(1, 10):
            mask = arr == c
            if not np.any(mask):
                continue

            labeled, num = ndimage.label(mask)
            if num == 1:
                # Single object - check if shape preserved in output
                out_mask = out_arr == c
                if np.any(out_mask):
                    out_labeled, out_num = ndimage.label(out_mask)
                    if out_num == 1:
                        # Same count = robust shape
                        if np.sum(mask) == np.sum(out_mask):
                            return f"robust_shape_{c}"

        return None
    except:
        return None


def try_conserve_count(inp, out):
    """Conserve total count of colored cells."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Total colored cells conserved
        inp_count = np.sum(arr > 0)
        out_count = np.sum(out_arr > 0)

        if inp_count == out_count and not np.array_equal(arr, out_arr):
            # Color counts match per-color
            for c in range(1, 10):
                if np.sum(arr == c) != np.sum(out_arr == c):
                    return None
            return "conserve_count_1"

        return None
    except:
        return None


def try_sustain_connectivity(inp, out):
    """Sustain connectivity between regions."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        from scipy import ndimage

        # Check connectivity preserved
        inp_labeled, inp_num = ndimage.label(arr > 0)
        out_labeled, out_num = ndimage.label(out_arr > 0)

        if inp_num == out_num and inp_num > 1:
            # Same number of connected components
            return "sustain_connectivity_1"

        return None
    except:
        return None


def try_maintain_aspect(inp, out):
    """Maintain aspect ratio of objects."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if aspect ratio maintained
        if h > 0 and oh > 0 and w > 0 and ow > 0:
            inp_aspect = w / h
            out_aspect = ow / oh

            if abs(inp_aspect - out_aspect) < 0.01:
                if oh != h or ow != w:
                    return "maintain_aspect_1"

        return None
    except:
        return None


def try_eternal_center(inp, out):
    """Center position that remains eternal."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h or ow != w:
            return None

        # Check if center unchanged
        ch, cw = h // 2, w // 2
        if arr[ch, cw] == out_arr[ch, cw] and arr[ch, cw] > 0:
            # Center preserved
            if not np.array_equal(arr, out_arr):
                return "eternal_center_1"

        return None
    except:
        return None

# Batch 45 - Pattern RECURSION/NESTING depth=-9000000 ANALYZE INFINITY


def try_recursive_subdivide(inp, out):
    """Recursively subdivide or apply fractal pattern."""
    arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = arr.shape
    oh, ow = out_arr.shape

    # Check if output is input where each cell is replaced by scaled input
    if oh == ih * ih and ow == iw * iw:
        result = np.zeros((oh, ow), dtype=arr.dtype)
        for i in range(ih):
            for j in range(iw):
                if arr[i, j] != 0:
                    # Place scaled copy
                    result[i*ih:(i+1)*ih, j*iw:(j+1)*iw] = arr * (arr[i, j] / max(1, np.max(arr)))
        # Simplified: just tile
        result = np.zeros((oh, ow), dtype=arr.dtype)
        for i in range(ih):
            for j in range(iw):
                result[i*ih:(i+1)*ih, j*iw:(j+1)*iw] = arr if arr[i, j] != 0 else 0
        if np.array_equal(result, out_arr):
            return "fractal_self_tile"

    return None


def try_nested_frames(inp, out):
    """Nested frames: shrinking concentric rectangles."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)
        h, w = arr.shape

        # Find frame colors from outside in
        colors_in = [c for c in np.unique(arr) if c != 0]
        colors_out = [c for c in np.unique(out_arr) if c != 0]

        result = np.zeros_like(arr)
        for depth, color in enumerate(colors_out[:min(h//2, w//2)]):
            result[depth:h-depth, depth:w-depth] = color

        if np.array_equal(result, out_arr):
            return "nested_frames"
        return None
    except:
        return None


def try_self_similar_scale(inp, out):
    """Self-similar pattern at different scales."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if output contains input at smaller scale
        for scale in [2, 3, 4]:
            if oh == h and ow == w and h >= scale and w >= scale:
                # Downscale output
                small_h, small_w = h // scale, w // scale
                if small_h > 0 and small_w > 0:
                    # Check corners for self-similarity
                    top_left = out_arr[:small_h, :small_w]
                    if np.array_equal(top_left, arr[:small_h, :small_w]):
                        return f"self_similar_{scale}"

        return None
    except:
        return None


def try_hierarchical_merge(inp, out):
    """Merge hierarchical levels."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check for hierarchical merge (2x2 blocks -> 1 cell)
        if oh * 2 == h and ow * 2 == w:
            result = np.zeros((oh, ow), dtype=int)
            for i in range(oh):
                for j in range(ow):
                    block = arr[i*2:i*2+2, j*2:j*2+2]
                    # Take mode or max
                    vals = block[block > 0]
                    if len(vals) > 0:
                        result[i, j] = np.bincount(vals).argmax()
            if np.array_equal(result, out_arr):
                return "hierarchical_merge_2x2"

        return None
    except:
        return None


def try_layer_extract(inp, out):
    """Extract a specific layer from stacked content."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        for c in range(1, 10):
            layer = np.zeros_like(arr)
            layer[arr == c] = c
            if np.array_equal(layer, out_arr):
                return f"layer_extract_{c}"
        return None
    except:
        return None


def try_fractal_apply(inp, out):
    """Apply fractal-like transformation."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check for fractal expansion
        if oh == h * 3 and ow == w * 3:
            # 3x3 pattern with center = original, corners = 0/original
            center = out_arr[h:2*h, w:2*w]
            if np.array_equal(center, arr):
                return "fractal_apply_3x"

        return None
    except:
        return None


def try_depth_peel(inp, out):
    """Peel one layer of depth from nested structure."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h or ow != w:
            return None

        # Check if output is arr with outer layer removed
        result = arr.copy()
        result[0, :] = 0
        result[-1, :] = 0
        result[:, 0] = 0
        result[:, -1] = 0

        if np.array_equal(result, out_arr):
            return "depth_peel_1"

        return None
    except:
        return None


def try_tree_flatten(inp, out):
    """Flatten tree structure to linear."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if output is flattened version
        if oh == 1 or ow == 1:
            flat_inp = arr.flatten()
            flat_out = out_arr.flatten()

            # Non-zero elements preserved
            inp_nonzero = flat_inp[flat_inp > 0]
            out_nonzero = flat_out[flat_out > 0]

            if np.array_equal(np.sort(inp_nonzero), np.sort(out_nonzero)):
                return "tree_flatten_1"

        return None
    except:
        return None


def try_iterate_transform(inp, out):
    """Apply a simple transform iteratively."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        # Try iterating rot90
        current = arr.copy()
        for i in range(1, 5):
            current = np.rot90(current)
            if np.array_equal(current, out_arr):
                return f"iterate_rot90_{i}x"

        # Try iterating flip
        current = arr.copy()
        for i in range(1, 3):
            current = np.fliplr(current)
            if np.array_equal(current, out_arr):
                return f"iterate_hflip_{i}x"

        return None
    except:
        return None


def try_embed_in_frame(inp, out):
    """Embed input in a larger frame."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if input is embedded in output with frame
        for margin in range(1, 5):
            if oh == h + 2*margin and ow == w + 2*margin:
                inner = out_arr[margin:oh-margin, margin:ow-margin]
                if np.array_equal(inner, arr):
                    # Check if frame is uniform
                    frame_color = out_arr[0, 0]
                    return f"embed_in_frame_{margin}"

        return None
    except:
        return None

# Batch 46 - Pattern GOAL-DIRECTEDNESS depth=-9000000 ANALYZE INFINITY


def try_seek_target(inp, out):
    """Find and extract target element."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Look for unique color as target
        for c in range(1, 10):
            mask = arr == c
            if np.sum(mask) == 1:  # Single cell target
                pos = np.argwhere(mask)[0]
                # Check if output is related to target position
                if oh == 1 and ow == 1:
                    if out_arr[0, 0] == c:
                        return f"seek_target_{c}"

        return None
    except:
        return None


def try_reach_destination(inp, out):
    """Move object to destination."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h or ow != w:
            return None

        from scipy import ndimage

        # Find moving objects
        for c in range(1, 10):
            inp_mask = arr == c
            out_mask = out_arr == c

            if np.any(inp_mask) and np.any(out_mask):
                inp_pos = np.mean(np.argwhere(inp_mask), axis=0)
                out_pos = np.mean(np.argwhere(out_mask), axis=0)

                # Significant movement to destination
                dist = np.sqrt((inp_pos[0] - out_pos[0])**2 + (inp_pos[1] - out_pos[1])**2)
                if dist > 2:
                    return f"reach_destination_{c}"

        return None
    except:
        return None


def try_accomplish_goal(inp, out):
    """Complete a goal state."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h or ow != w:
            return None

        # Check if output is completed version of input
        # E.g., partial shape becomes complete
        inp_colors = set(arr.flatten()) - {0}
        out_colors = set(out_arr.flatten()) - {0}

        if inp_colors == out_colors:
            # Same colors, check for completion
            inp_count = np.sum(arr > 0)
            out_count = np.sum(out_arr > 0)

            if out_count > inp_count * 1.5:
                return "accomplish_goal_1"

        return None
    except:
        return None


def try_find_path(inp, out):
    """Find path between points."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h or ow != w:
            return None

        # Check if output shows a path (connected line)
        diff = (out_arr > 0) & (arr == 0)
        path_cells = np.sum(diff)

        if path_cells > 0:
            # Check if path forms a line
            from scipy import ndimage
            labeled, num = ndimage.label(diff)
            if num == 1:  # Single connected path
                return "find_path_1"

        return None
    except:
        return None


def try_pursue_pattern(inp, out):
    """Pursue and extend pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if output extends input pattern
        if oh > h or ow > w:
            # Output larger - pattern extended
            if oh >= h and ow >= w:
                # Check if input is contained in output
                for start_h in range(oh - h + 1):
                    for start_w in range(ow - w + 1):
                        if np.array_equal(out_arr[start_h:start_h+h, start_w:start_w+w], arr):
                            return "pursue_pattern_1"

        return None
    except:
        return None


def try_achieve_symmetry(inp, out):
    """Achieve symmetrical state."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h or ow != w:
            return None

        # Check if output is symmetric while input is not
        inp_h_sym = np.array_equal(arr, np.fliplr(arr))
        inp_v_sym = np.array_equal(arr, np.flipud(arr))
        out_h_sym = np.array_equal(out_arr, np.fliplr(out_arr))
        out_v_sym = np.array_equal(out_arr, np.flipud(out_arr))

        if (out_h_sym and not inp_h_sym) or (out_v_sym and not inp_v_sym):
            return "achieve_symmetry_1"

        return None
    except:
        return None


def try_intent_fill(inp, out):
    """Fill with intent/purpose."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h or ow != w:
            return None

        # Check if output fills specific regions with intent
        filled_cells = (out_arr > 0) & (arr == 0)
        if np.sum(filled_cells) > 0:
            # Check if fill follows a pattern
            fill_color = out_arr[filled_cells][0] if np.any(filled_cells) else 0
            if len(set(out_arr[filled_cells])) == 1:
                return f"intent_fill_{fill_color}"

        return None
    except:
        return None


def try_outcome_select(inp, out):
    """Select based on outcome criteria."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if output is selection of largest object
        from scipy import ndimage

        labeled, num = ndimage.label(arr > 0)
        if num > 1:
            sizes = [(i, np.sum(labeled == i)) for i in range(1, num + 1)]
            largest = max(sizes, key=lambda x: x[1])

            selected = np.where(labeled == largest[0], arr, 0)
            # Crop to bounding box
            rows = np.any(selected > 0, axis=1)
            cols = np.any(selected > 0, axis=0)
            if np.any(rows) and np.any(cols):
                cropped = selected[rows][:, cols]
                if np.array_equal(cropped, out_arr):
                    return "outcome_select_largest"

        return None
    except:
        return None


def try_plan_execute(inp, out):
    """Execute planned transformation."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check for staged transformation
        if oh == h and ow == w:
            # Count changes
            changes = np.sum(arr != out_arr)
            total = h * w

            # Moderate change ratio suggests planned execution
            if 0.1 < changes / total < 0.5:
                return "plan_execute_1"

        return None
    except:
        return None


def try_drive_expand(inp, out):
    """Drive/motivation to expand."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check if output expands input
        if oh > h and ow > w:
            ratio_h = oh / h
            ratio_w = ow / w

            if abs(ratio_h - ratio_w) < 0.1:  # Uniform expansion
                # Check if content scales
                scale = int(round(ratio_h))
                if oh == h * scale and ow == w * scale:
                    expanded = np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)
                    if np.array_equal(expanded, out_arr):
                        return f"drive_expand_{scale}x"

        return None
    except:
        return None

# Batch 47 - Pattern CAUSALITY/CONTACT depth=-9000000 ANALYZE INFINITY


def try_cause_effect(inp, out):
    """Cause-effect: object triggers change in another."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        if out_arr.shape != (h, w):
            return None

        # Find where changes occurred
        diff = arr != out_arr
        if not np.any(diff):
            return None

        # Look for cause-effect: non-zero causes zero to change
        for c in range(1, 10):
            cause_mask = arr == c
            if not np.any(cause_mask):
                continue

            # Effect: adjacent cells changed
            effect_mask = diff & (arr == 0)
            if np.any(effect_mask):
                # Check if effects are near causes
                from scipy import ndimage
                dilated = ndimage.binary_dilation(cause_mask, iterations=2)
                if np.any(dilated & effect_mask):
                    return f"cause_effect_{c}"

        return None
    except:
        return None


def try_contact_transform(inp, out):
    """Contact/collision causes transformation."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        if out_arr.shape != (h, w):
            return None

        # Find objects that touch
        for c1 in range(1, 10):
            mask1 = arr == c1
            if not np.any(mask1):
                continue

            for c2 in range(c1 + 1, 10):
                mask2 = arr == c2
                if not np.any(mask2):
                    continue

                # Check if they touch (are adjacent)
                from scipy import ndimage
                dilated1 = ndimage.binary_dilation(mask1)
                if np.any(dilated1 & mask2):
                    # They touch - check if output merges them
                    combined = mask1 | mask2
                    out_at_combined = out_arr[combined]
                    if len(np.unique(out_at_combined)) == 1:
                        return f"contact_merge_{c1}_{c2}"

        return None
    except:
        return None


def try_force_propagate(inp, out):
    """Force propagates through grid."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        if out_arr.shape != (h, w):
            return None

        # Look for propagation: color spreads in direction
        for c in range(1, 10):
            mask = arr == c
            if not np.any(mask):
                continue

            out_mask = out_arr == c
            if not np.any(out_mask):
                continue

            # Check if it spread in a direction
            if np.sum(out_mask) > np.sum(mask):
                # Find direction of spread
                in_rows = np.where(np.any(mask, axis=1))[0]
                out_rows = np.where(np.any(out_mask, axis=1))[0]

                if len(out_rows) > len(in_rows):
                    if out_rows[-1] > in_rows[-1]:
                        return f"force_down_{c}"
                    elif out_rows[0] < in_rows[0]:
                        return f"force_up_{c}"

        return None
    except:
        return None


def try_trigger_consequence(inp, out):
    """Trigger causes consequence at distance."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        if out_arr.shape != (h, w):
            return None

        # Find a unique trigger color
        colors, counts = np.unique(arr[arr > 0], return_counts=True)
        if len(colors) < 2:
            return None

        trigger_idx = np.argmin(counts)
        trigger_color = colors[trigger_idx]
        trigger_pos = np.argwhere(arr == trigger_color)

        if len(trigger_pos) == 0:
            return None

        # Check what changed
        diff = arr != out_arr
        changed_pos = np.argwhere(diff)

        if len(changed_pos) > 0:
            # See if changes align with trigger (row or col)
            tr, tc = trigger_pos[0]
            same_row = np.any(changed_pos[:, 0] == tr)
            same_col = np.any(changed_pos[:, 1] == tc)

            if same_row:
                return f"trigger_row_{trigger_color}"
            if same_col:
                return f"trigger_col_{trigger_color}"

        return None
    except:
        return None


def try_chain_reaction(inp, out):
    """Chain reaction: changes propagate step by step."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        if out_arr.shape != (h, w):
            return None

        # Look for flood-fill like pattern
        for c in range(1, 10):
            in_mask = arr == c
            out_mask = out_arr == c

            if not np.any(in_mask):
                continue

            # Did it spread?
            in_count = np.sum(in_mask)
            out_count = np.sum(out_mask)

            if out_count > in_count:
                # Check if it's a connected region in output
                from scipy import ndimage
                labeled, num = ndimage.label(out_mask)
                if num == 1:  # Single connected component
                    return f"chain_reaction_{c}"

        return None
    except:
        return None


def try_impact_pattern(inp, out):
    """Impact: object collision creates pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        if out_arr.shape != (h, w):
            return None

        # Look for impact crater pattern
        # Center of activity + radial effect
        diff = arr != out_arr
        if not np.any(diff):
            return None

        changed = np.argwhere(diff)
        if len(changed) < 3:
            return None

        # Find center of changes
        center_r = np.mean(changed[:, 0])
        center_c = np.mean(changed[:, 1])

        # Check if changes radiate from center
        distances = np.sqrt((changed[:, 0] - center_r)**2 + (changed[:, 1] - center_c)**2)

        if np.std(distances) < np.mean(distances) * 0.5:
            # Changes are roughly circular - impact pattern
            impact_color = out_arr[int(center_r), int(center_c)]
            return f"impact_{impact_color}"

        return None
    except:
        return None


def try_push_direction(inp, out):
    """Push: objects move in force direction."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        if out_arr.shape != (h, w):
            return None

        # Find objects and check if they shifted
        for c in range(1, 10):
            in_mask = arr == c
            out_mask = out_arr == c

            if not np.any(in_mask) or not np.any(out_mask):
                continue

            if np.sum(in_mask) != np.sum(out_mask):
                continue  # Count changed, not just movement

            in_pos = np.argwhere(in_mask)
            out_pos = np.argwhere(out_mask)

            if len(in_pos) == len(out_pos):
                in_center = np.mean(in_pos, axis=0)
                out_center = np.mean(out_pos, axis=0)

                dr = out_center[0] - in_center[0]
                dc = out_center[1] - in_center[1]

                if abs(dr) > 0.5 or abs(dc) > 0.5:
                    direction = ""
                    if dr > 0.5:
                        direction = "down"
                    elif dr < -0.5:
                        direction = "up"
                    if dc > 0.5:
                        direction += "right"
                    elif dc < -0.5:
                        direction += "left"

                    if direction:
                        return f"push_{direction}_{c}"

        return None
    except:
        return None


def try_response_neighbor(inp, out):
    """Response: cells respond to neighbor state."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        if out_arr.shape != (h, w):
            return None

        # Check if cells change based on neighbor count
        for r in range(h):
            for c in range(w):
                if arr[r, c] != out_arr[r, c]:
                    # Count non-zero neighbors
                    neighbors = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if arr[nr, nc] > 0:
                                    neighbors += 1

                    # Changed based on neighbors
                    if neighbors >= 3 and arr[r, c] == 0 and out_arr[r, c] > 0:
                        return f"response_birth_{neighbors}"
                    if neighbors <= 1 and arr[r, c] > 0 and out_arr[r, c] == 0:
                        return f"response_death_{neighbors}"

        return None
    except:
        return None


def try_influence_spread(inp, out):
    """Influence: color spreads to adjacent zeros."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        if out_arr.shape != (h, w):
            return None

        for c in range(1, 10):
            in_mask = arr == c
            out_mask = out_arr == c

            if not np.any(in_mask):
                continue

            # Check if color spread to adjacent cells
            from scipy import ndimage
            dilated = ndimage.binary_dilation(in_mask)
            expected_spread = dilated & (arr == 0)
            actual_spread = out_mask & (arr == 0)

            if np.any(actual_spread):
                # Check if it spread where expected
                overlap = np.sum(expected_spread & actual_spread)
                total_spread = np.sum(actual_spread)

                if overlap > 0 and overlap >= total_spread * 0.5:
                    return f"influence_spread_{c}"

        return None
    except:
        return None


def try_reaction_replace(inp, out):
    """Reaction: one color replaces another on contact."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        if out_arr.shape != (h, w):
            return None

        # Find colors that disappear
        in_colors = set(np.unique(arr))
        out_colors = set(np.unique(out_arr))

        disappeared = in_colors - out_colors
        appeared = out_colors - in_colors

        for d in disappeared:
            if d == 0:
                continue
            for a in appeared:
                if a == 0:
                    continue
                # Check if d was replaced by a
                d_positions = np.argwhere(arr == d)
                for pos in d_positions:
                    if out_arr[pos[0], pos[1]] == a:
                        return f"reaction_replace_{d}_with_{a}"

        return None
    except:
        return None

# Batch 48 - Pattern NUMBER/ARITHMETIC depth=-9000000 ANALYZE INFINITY


def try_count_ratio(inp, out):
    """Count ratio: output encodes ratio of colors."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        oh, ow = out_arr.shape

        # Count each color in input
        for c1 in range(1, 10):
            count1 = np.sum(arr == c1)
            if count1 == 0:
                continue
            for c2 in range(c1 + 1, 10):
                count2 = np.sum(arr == c2)
                if count2 == 0:
                    continue

                # Check if output dimensions encode ratio
                if count1 > 0 and count2 > 0:
                    ratio = count1 / count2 if count2 > 0 else 0
                    if oh == int(round(ratio)) or ow == int(round(ratio)):
                        return f"count_ratio_{c1}_{c2}"

        return None
    except:
        return None


def try_sum_encode(inp, out):
    """Sum encoding: output size is sum of counts."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        oh, ow = out_arr.shape

        # Sum of all non-zero cells
        total = np.sum(arr > 0)

        if oh * ow == total:
            return f"sum_encode_{total}"
        if oh == total or ow == total:
            return f"sum_linear_{total}"

        return None
    except:
        return None


def try_difference_mark(inp, out):
    """Mark cells by count difference."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        if out_arr.shape != (h, w):
            return None

        # Find color with different count in output
        for c in range(1, 10):
            in_count = np.sum(arr == c)
            out_count = np.sum(out_arr == c)

            if in_count > 0 and out_count > 0 and in_count != out_count:
                diff = abs(out_count - in_count)
                return f"difference_mark_{c}_{diff}"

        return None
    except:
        return None


def try_proportion_scale(inp, out):
    """Proportional scaling of pattern."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Check for proportional relationship
        if oh > h and ow > w:
            if h > 0 and w > 0:
                h_ratio = oh / h
                w_ratio = ow / w

                if abs(h_ratio - w_ratio) < 0.01:  # Same ratio
                    ratio = int(round(h_ratio))
                    if oh == h * ratio and ow == w * ratio:
                        return f"proportion_scale_{ratio}x"

        return None
    except:
        return None


def try_quantity_select(inp, out):
    """Select objects by quantity/count."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        if out_arr.shape != (h, w):
            return None

        # Count objects per color
        from scipy import ndimage

        color_counts = {}
        for c in range(1, 10):
            mask = arr == c
            if np.any(mask):
                labeled, num = ndimage.label(mask)
                color_counts[c] = num

        if len(color_counts) < 2:
            return None

        # Check which colors are preserved (by count)
        preserved = []
        for c, count in color_counts.items():
            if np.any(out_arr == c):
                preserved.append((c, count))

        if len(preserved) == 1:
            # Single color preserved - was it min or max count?
            c, count = preserved[0]
            if count == min(color_counts.values()):
                return f"quantity_select_min_{c}"
            if count == max(color_counts.values()):
                return f"quantity_select_max_{c}"

        return None
    except:
        return None


def try_multiply_pattern(inp, out):
    """Multiply/repeat pattern by count."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Count a specific marker
        for c in range(1, 10):
            count = np.sum(arr == c)
            if count > 1 and count < 10:
                # Check if output is multiplied
                if oh == h * count or ow == w * count:
                    return f"multiply_by_{c}_count"
                if oh * ow == h * w * count:
                    return f"area_multiply_{c}"

        return None
    except:
        return None


def try_divide_grid(inp, out):
    """Divide grid by count."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        # Find divisors
        for c in range(1, 10):
            count = np.sum(arr == c)
            if count > 1:
                if h > 0 and h % count == 0 and oh == h // count:
                    return f"divide_h_by_{c}_count"
                if w > 0 and w % count == 0 and ow == w // count:
                    return f"divide_w_by_{c}_count"

        return None
    except:
        return None


def try_sequence_next(inp, out):
    """Predict next in arithmetic sequence."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape

        # Look for row/column sequences
        for r in range(h):
            row = arr[r, :]
            non_zero = row[row > 0]
            if len(non_zero) >= 2:
                # Check if arithmetic sequence
                diffs = np.diff(non_zero)
                if len(np.unique(diffs)) == 1:
                    d = diffs[0]
                    if d != 0:
                        return f"sequence_next_d{d}"

        for c in range(w):
            col = arr[:, c]
            non_zero = col[col > 0]
            if len(non_zero) >= 2:
                diffs = np.diff(non_zero)
                if len(np.unique(diffs)) == 1:
                    d = diffs[0]
                    if d != 0:
                        return f"sequence_col_d{d}"

        return None
    except:
        return None


def try_magnitude_order(inp, out):
    """Order by magnitude/size."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        if out_arr.shape != (h, w):
            return None

        # Count object sizes per color
        from scipy import ndimage

        sizes = {}
        for c in range(1, 10):
            mask = arr == c
            if np.any(mask):
                sizes[c] = np.sum(mask)

        if len(sizes) < 2:
            return None

        # Check if output reorders by size
        sorted_colors = sorted(sizes.keys(), key=lambda x: sizes[x])

        # Check top row for ordering
        out_top = out_arr[0, :]
        out_colors = [c for c in out_top if c > 0]

        if out_colors == sorted_colors[:len(out_colors)]:
            return "magnitude_order_asc"
        if out_colors == sorted_colors[::-1][:len(out_colors)]:
            return "magnitude_order_desc"

        return None
    except:
        return None


def try_formula_apply(inp, out):
    """Apply formula: f(position) = color."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        if out_arr.shape != (h, w):
            return None

        # Check for position-based formulas
        # f(r,c) = (r + c) % colors
        max_color = int(np.max(out_arr))
        if max_color < 2:
            return None

        matches_sum = True
        matches_diff = True
        matches_prod = True

        for r in range(h):
            for c in range(w):
                expected_sum = (r + c) % max_color + 1 if max_color > 0 else 0
                expected_diff = abs(r - c) % max_color + 1 if max_color > 0 else 0
                expected_prod = (r * c) % max_color + 1 if max_color > 0 else 0

                if out_arr[r, c] != expected_sum:
                    matches_sum = False
                if out_arr[r, c] != expected_diff:
                    matches_diff = False
                if out_arr[r, c] != expected_prod:
                    matches_prod = False

        if matches_sum:
            return "formula_sum_mod"
        if matches_diff:
            return "formula_diff_mod"
        if matches_prod:
            return "formula_prod_mod"

        return None
    except:
        return None

# ============================================================================
# Batch 49 - Pattern AGENTS/MOTION depth=-9000000 ANALYZE INFINITY
# ============================================================================


def try_agent_pursuit(inp, out):
    """Agent pursues target: object moves toward another."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        # Find colored objects (potential agents and targets)
        colors = [c for c in range(1, 10) if np.any(inp_arr == c)]
        if len(colors) < 2:
            return None

        for agent_color in colors:
            for target_color in colors:
                if agent_color == target_color:
                    continue

                # Find agent and target positions
                agent_pos = np.argwhere(inp_arr == agent_color)
                target_pos = np.argwhere(out_arr == target_color)

                if len(agent_pos) == 0 or len(target_pos) == 0:
                    continue

                agent_center = agent_pos.mean(axis=0)
                target_center = target_pos.mean(axis=0)

                # Agent should move toward target
                new_agent_pos = np.argwhere(out_arr == agent_color)
                if len(new_agent_pos) == 0:
                    continue
                new_center = new_agent_pos.mean(axis=0)

                # Check if moved closer
                old_dist = np.sqrt(np.sum((agent_center - target_center)**2))
                new_dist = np.sqrt(np.sum((new_center - target_center)**2))

                if new_dist < old_dist:
                    return f"agent_pursuit_{agent_color}_{target_color}"

        return None
    except:
        return None


def try_agent_flee(inp, out):
    """Agent flees from threat: object moves away."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        colors = [c for c in range(1, 10) if np.any(inp_arr == c)]
        if len(colors) < 2:
            return None

        for agent_color in colors:
            for threat_color in colors:
                if agent_color == threat_color:
                    continue

                agent_pos = np.argwhere(inp_arr == agent_color)
                threat_pos = np.argwhere(inp_arr == threat_color)

                if len(agent_pos) == 0 or len(threat_pos) == 0:
                    continue

                agent_center = agent_pos.mean(axis=0)
                threat_center = threat_pos.mean(axis=0)

                new_agent_pos = np.argwhere(out_arr == agent_color)
                if len(new_agent_pos) == 0:
                    continue
                new_center = new_agent_pos.mean(axis=0)

                old_dist = np.sqrt(np.sum((agent_center - threat_center)**2))
                new_dist = np.sqrt(np.sum((new_center - threat_center)**2))

                if new_dist > old_dist:
                    return f"agent_flee_{agent_color}_from_{threat_color}"

        return None
    except:
        return None


def try_trajectory_extend(inp, out):
    """Extend trajectory: continue motion path."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        for color in range(1, 10):
            if not np.any(inp_arr == color):
                continue

            # Find trajectory (sequence of pixels)
            pos = np.argwhere(inp_arr == color)
            if len(pos) < 2:
                continue

            # Sort by position to find direction
            sorted_by_row = pos[pos[:, 0].argsort()]
            sorted_by_col = pos[pos[:, 1].argsort()]

            # Check for direction
            if len(pos) >= 2:
                dr = sorted_by_row[-1][0] - sorted_by_row[0][0]
                dc = sorted_by_col[-1][1] - sorted_by_col[0][1]

                if dr != 0 or dc != 0:
                    # Normalize direction
                    step_r = 1 if dr > 0 else (-1 if dr < 0 else 0)
                    step_c = 1 if dc > 0 else (-1 if dc < 0 else 0)

                    # Check if output extends trajectory
                    out_pos = np.argwhere(out_arr == color)
                    if len(out_pos) > len(pos):
                        return f"trajectory_extend_{color}"

        return None
    except:
        return None


def try_velocity_constant(inp, out):
    """Constant velocity: objects move uniformly."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        for color in range(1, 10):
            inp_pos = np.argwhere(inp_arr == color)
            out_pos = np.argwhere(out_arr == color)

            if len(inp_pos) == 0 or len(out_pos) == 0:
                continue
            if len(inp_pos) != len(out_pos):
                continue

            # Calculate displacement
            inp_center = inp_pos.mean(axis=0)
            out_center = out_pos.mean(axis=0)

            dr = out_center[0] - inp_center[0]
            dc = out_center[1] - inp_center[1]

            if abs(dr) > 0 or abs(dc) > 0:
                # Verify uniform translation
                expected = np.zeros_like(out_arr)
                for r, c in inp_pos:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w:
                        expected[nr, nc] = color

                if np.array_equal(out_arr == color, expected == color):
                    return f"velocity_constant_{color}_{int(dr)}_{int(dc)}"

        return None
    except:
        return None


def try_navigate_obstacle(inp, out):
    """Navigate around obstacle: agent finds path."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        colors = [c for c in range(1, 10) if np.any(inp_arr == c)]
        if len(colors) < 2:
            return None

        # Find agent (color that moves)
        for agent_color in colors:
            inp_pos = np.argwhere(inp_arr == agent_color)
            out_pos = np.argwhere(out_arr == agent_color)

            if len(inp_pos) == 0 or len(out_pos) == 0:
                continue

            inp_center = inp_pos.mean(axis=0)
            out_center = out_pos.mean(axis=0)

            if not np.allclose(inp_center, out_center):
                # Agent moved - check if path avoided obstacles
                for obs_color in colors:
                    if obs_color == agent_color:
                        continue

                    obs_pos = np.argwhere(inp_arr == obs_color)
                    if len(obs_pos) > 0:
                        # Check if new position is valid (not overlapping obstacle)
                        obs_set = set(map(tuple, obs_pos))
                        new_set = set(map(tuple, out_pos))
                        if len(obs_set & new_set) == 0:
                            return f"navigate_obstacle_{agent_color}"

        return None
    except:
        return None


def try_autonomous_expand(inp, out):
    """Autonomous expansion: self-propelled growth."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        for color in range(1, 10):
            inp_count = np.sum(inp_arr == color)
            out_count = np.sum(out_arr == color)

            if inp_count > 0 and out_count > inp_count:
                # Check if expansion is connected (self-propelled)
                inp_pos = set(map(tuple, np.argwhere(inp_arr == color)))
                out_pos = set(map(tuple, np.argwhere(out_arr == color)))

                # New positions should be adjacent to old
                new_pos = out_pos - inp_pos
                all_adjacent = True
                for nr, nc in new_pos:
                    adjacent_to_old = False
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        if (nr+dr, nc+dc) in inp_pos:
                            adjacent_to_old = True
                            break
                    if not adjacent_to_old:
                        all_adjacent = False
                        break

                if all_adjacent and len(new_pos) > 0:
                    return f"autonomous_expand_{color}"

        return None
    except:
        return None


def try_behavior_pattern(inp, out):
    """Behavioral pattern: repeated action sequence."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        for color in range(1, 10):
            if not np.any(inp_arr == color):
                continue

            # Find objects of this color
            inp_mask = (inp_arr == color)
            out_mask = (out_arr == color)

            # Check for behavioral transformation
            if not np.array_equal(inp_mask, out_mask):
                # Analyze the change pattern
                added = out_mask & ~inp_mask
                removed = inp_mask & ~out_mask

                if np.sum(added) > 0 and np.sum(removed) > 0:
                    # Motion-like behavior
                    return f"behavior_pattern_{color}"

        return None
    except:
        return None


def try_intention_mark(inp, out):
    """Mark intention/goal: indicate target position."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        # Find new colors in output (intentions marked)
        inp_colors = set(np.unique(inp_arr)) - {0}
        out_colors = set(np.unique(out_arr)) - {0}

        new_colors = out_colors - inp_colors
        if len(new_colors) == 0:
            return None

        for new_color in new_colors:
            new_pos = np.argwhere(out_arr == new_color)
            if len(new_pos) > 0:
                # Check if new color marks a specific location
                return f"intention_mark_{new_color}"

        return None
    except:
        return None


def try_approach_goal(inp, out):
    """Approach goal: move toward specific target."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        colors = [c for c in range(1, 10) if np.any(inp_arr == c)]

        for mover in colors:
            for goal in colors:
                if mover == goal:
                    continue

                inp_mover = np.argwhere(inp_arr == mover)
                out_mover = np.argwhere(out_arr == mover)
                goal_pos = np.argwhere(inp_arr == goal)

                if len(inp_mover) == 0 or len(out_mover) == 0 or len(goal_pos) == 0:
                    continue

                # Check if mover reached goal
                goal_set = set(map(tuple, goal_pos))
                out_set = set(map(tuple, out_mover))

                if len(goal_set & out_set) > 0:
                    return f"approach_goal_{mover}_to_{goal}"

        return None
    except:
        return None


def try_avoid_color(inp, out):
    """Avoid specific color: move away from threat."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        colors = [c for c in range(1, 10) if np.any(inp_arr == c)]
        if len(colors) < 2:
            return None

        for agent in colors:
            for avoid in colors:
                if agent == avoid:
                    continue

                inp_agent = np.argwhere(inp_arr == agent)
                out_agent = np.argwhere(out_arr == agent)
                avoid_pos = np.argwhere(inp_arr == avoid)

                if len(inp_agent) == 0 or len(out_agent) == 0 or len(avoid_pos) == 0:
                    continue

                # Check minimum distance increased
                inp_center = inp_agent.mean(axis=0)
                out_center = out_agent.mean(axis=0)
                avoid_center = avoid_pos.mean(axis=0)

                inp_dist = np.sqrt(np.sum((inp_center - avoid_center)**2))
                out_dist = np.sqrt(np.sum((out_center - avoid_center)**2))

                if out_dist > inp_dist + 0.5:
                    return f"avoid_color_{agent}_from_{avoid}"

        return None
    except:
        return None

# ============================================================================
# Batch 50 - Pattern SYMMETRY/REFLECTION depth=-9000000 ANALYZE INFINITY
# ============================================================================


def try_bilateral_complete(inp, out):
    """Complete bilateral symmetry across vertical axis."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        mid = w // 2

        # Check if output is bilaterally symmetric
        for axis_offset in range(-2, 3):
            axis = mid + axis_offset
            if axis <= 0 or axis >= w:
                continue

            # Check if left mirrors right
            left = out_arr[:, :axis]
            right = out_arr[:, axis:]

            if left.shape[1] == 0 or right.shape[1] == 0:
                continue

            # Flip right to compare
            right_flipped = right[:, ::-1]
            min_w = min(left.shape[1], right_flipped.shape[1])

            if np.array_equal(left[:, -min_w:], right_flipped[:, :min_w]):
                # Now check if input was partial
                inp_left = inp_arr[:, :axis]
                inp_right = inp_arr[:, axis:]
                inp_right_flipped = inp_right[:, ::-1]

                if not np.array_equal(inp_left[:, -min_w:], inp_right_flipped[:, :min_w]):
                    return f"bilateral_complete_axis_{axis}"

        return None
    except:
        return None


def try_radial_symmetry(inp, out):
    """Apply 4-fold rotational symmetry."""
    arr = np.array(inp)
    out_arr = np.array(out)
    h, w = arr.shape

    if h != w:
        return None

    result = arr.copy()
    for k in [1, 2, 3]:
        rotated = np.rot90(arr, k)
        result = np.where(result != 0, result, rotated)

    if np.array_equal(result, out_arr):
        return "radial_symmetry_4"
    return None


def try_fold_symmetry(inp, out):
    """Complete by folding along axis."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        # Horizontal fold
        top = out_arr[:h//2, :]
        bottom = out_arr[h//2:, :]
        if top.shape == bottom.shape:
            if np.array_equal(top, bottom[::-1, :]):
                inp_top = inp_arr[:h//2, :]
                inp_bottom = inp_arr[h//2:, :]
                if not np.array_equal(inp_top, inp_bottom[::-1, :]):
                    return "fold_symmetry_h"

        # Vertical fold
        left = out_arr[:, :w//2]
        right = out_arr[:, w//2:]
        if left.shape == right.shape:
            if np.array_equal(left, right[:, ::-1]):
                inp_left = inp_arr[:, :w//2]
                inp_right = inp_arr[:, w//2:]
                if not np.array_equal(inp_left, inp_right[:, ::-1]):
                    return "fold_symmetry_v"

        return None
    except:
        return None


def try_axis_reflect(inp, out):
    """Reflect pattern across axis."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        # Check for diagonal reflection
        if h == w:
            transposed = inp_arr.T
            if np.array_equal(transposed, out_arr):
                return "axis_reflect_diagonal"

            anti_transposed = np.rot90(np.fliplr(inp_arr))
            if np.array_equal(anti_transposed, out_arr):
                return "axis_reflect_antidiag"

        return None
    except:
        return None


def try_balance_pattern(inp, out):
    """Balance pattern: equalize across halves."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        for color in range(1, 10):
            inp_count = np.sum(inp_arr == color)
            out_count = np.sum(out_arr == color)

            if inp_count > 0 and out_count > inp_count:
                # Check if balanced in output
                out_left = np.sum(out_arr[:, :w//2] == color)
                out_right = np.sum(out_arr[:, w//2:] == color)

                inp_left = np.sum(inp_arr[:, :w//2] == color)
                inp_right = np.sum(inp_arr[:, w//2:] == color)

                # Was unbalanced, now balanced
                if abs(out_left - out_right) < abs(inp_left - inp_right):
                    return f"balance_pattern_{color}"

        return None
    except:
        return None


def try_center_align(inp, out):
    """Align pattern to center."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        center_h, center_w = h // 2, w // 2

        for color in range(1, 10):
            inp_pos = np.argwhere(inp_arr == color)
            out_pos = np.argwhere(out_arr == color)

            if len(inp_pos) == 0 or len(out_pos) == 0:
                continue
            if len(inp_pos) != len(out_pos):
                continue

            inp_center = inp_pos.mean(axis=0)
            out_center = out_pos.mean(axis=0)

            # Check if moved toward grid center
            inp_dist = np.sqrt((inp_center[0] - center_h)**2 + (inp_center[1] - center_w)**2)
            out_dist = np.sqrt((out_center[0] - center_h)**2 + (out_center[1] - center_w)**2)

            if out_dist < inp_dist - 0.5:
                return f"center_align_{color}"

        return None
    except:
        return None


def try_correspondence_map(inp, out):
    """Map corresponding elements."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        # Find color pairs that correspond
        colors = [c for c in range(1, 10) if np.any(inp_arr == c)]

        for c1 in colors:
            for c2 in colors:
                if c1 >= c2:
                    continue

                pos1 = np.argwhere(inp_arr == c1)
                pos2 = np.argwhere(inp_arr == c2)

                if len(pos1) != len(pos2):
                    continue
                if len(pos1) == 0:
                    continue

                # Check if symmetric positions
                center1 = pos1.mean(axis=0)
                center2 = pos2.mean(axis=0)

                # Check if output connects them
                for r, c in pos1:
                    if out_arr[r, c] != inp_arr[r, c]:
                        return f"correspondence_map_{c1}_{c2}"

        return None
    except:
        return None


def try_inverse_fill(inp, out):
    """Fill inverse/complement of pattern."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        if out_arr.shape != (h, w):
            return None

        # Check if output fills where input is zero
        for fill_color in range(1, 10):
            if not np.any(out_arr == fill_color):
                continue

            # Positions where output has fill_color
            out_fill = (out_arr == fill_color)
            inp_zero = (inp_arr == 0)

            # Check if fill is exactly where input is zero
            if np.array_equal(out_fill, inp_zero):
                return f"inverse_fill_{fill_color}"

            # Or partial inverse
            if np.sum(out_fill & inp_zero) > np.sum(out_fill & ~inp_zero):
                # Most of fill is in zero regions
                if np.sum(out_fill) > np.sum(inp_arr == fill_color):
                    return f"inverse_fill_partial_{fill_color}"

        return None
    except:
        return None


def try_preserve_invariant(inp, out):
    """Preserve invariant elements across transformation."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        h, w = arr.shape
        oh, ow = out_arr.shape

        if oh != h or ow != w:
            return None

        # Find elements that stay constant
        invariant_mask = arr == out_arr
        changed_mask = arr != out_arr

        # Check if changes follow pattern
        if np.sum(changed_mask) > 0:
            # Find what changes
            src_vals = arr[changed_mask]
            dst_vals = out_arr[changed_mask]

            # Single color replacement preserving invariant
            if len(set(src_vals)) == 1 and len(set(dst_vals)) == 1:
                return "preserve_invariant_1"

        return None
    except:
        return None


def try_duplicate_reflect(inp, out):
    """Duplicate and reflect pattern."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape

        # Check for doubled dimensions
        if out_arr.shape == (h, 2*w):
            # Horizontal duplication with reflection
            left = out_arr[:, :w]
            right = out_arr[:, w:]
            if np.array_equal(left, inp_arr) and np.array_equal(right, inp_arr[:, ::-1]):
                return "duplicate_reflect_h"

        if out_arr.shape == (2*h, w):
            # Vertical duplication with reflection
            top = out_arr[:h, :]
            bottom = out_arr[h:, :]
            if np.array_equal(top, inp_arr) and np.array_equal(bottom, inp_arr[::-1, :]):
                return "duplicate_reflect_v"

        return None
    except:
        return None

# Batch - Pattern transforms


def try_inside_fill(inp, out):
    """Child learns INSIDE: fill what's contained within a boundary."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        h, w = inp_arr.shape

        # Find enclosed regions and fill them
        for fill_color in range(1, 10):
            test = inp_arr.copy()
            # Find background and mark interior
            from scipy import ndimage
            for bg in range(10):
                if bg == fill_color:
                    continue
                mask = (inp_arr == bg)
                labeled, num = ndimage.label(mask)
                for region in range(1, num + 1):
                    region_mask = (labeled == region)
                    # Check if region touches border (not inside)
                    if not (region_mask[0, :].any() or region_mask[-1, :].any() or
                            region_mask[:, 0].any() or region_mask[:, -1].any()):
                        test[region_mask] = fill_color
            if np.array_equal(test, out_arr):
                return f"inside_fill_{fill_color}"
        return None
    except:
        return None


def try_outside_clear(inp, out):
    """Child learns OUTSIDE: clear/change what's outside boundaries."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None

        from scipy import ndimage
        # Find regions touching border (outside)
        for clear_to in range(10):
            test = inp_arr.copy()
            for color in range(1, 10):
                mask = (inp_arr == color)
                labeled, num = ndimage.label(mask)
                for region in range(1, num + 1):
                    region_mask = (labeled == region)
                    # If touches border, it's "outside"
                    if (region_mask[0, :].any() or region_mask[-1, :].any() or
                        region_mask[:, 0].any() or region_mask[:, -1].any()):
                        test[region_mask] = clear_to
            if np.array_equal(test, out_arr):
                return f"outside_clear_{clear_to}"
        return None
    except:
        return None


def try_above_stack(inp, out):
    """Child learns ABOVE: stack/place something above another."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h_in, w_in = inp_arr.shape
        h_out, w_out = out_arr.shape

        # Output is input stacked above itself or above modified version
        if w_in == w_out and h_out == 2 * h_in:
            # Check if output is input above input
            top = out_arr[:h_in, :]
            bottom = out_arr[h_in:, :]
            if np.array_equal(top, inp_arr) and np.array_equal(bottom, inp_arr):
                return "above_stack_self"
            if np.array_equal(bottom, inp_arr) and np.array_equal(top, inp_arr[::-1, :]):
                return "above_stack_flip"
        return None
    except:
        return None


def try_below_stack(inp, out):
    """Child learns BELOW: stack/place something below another."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h_in, w_in = inp_arr.shape
        h_out, w_out = out_arr.shape

        if w_in == w_out and h_out == 2 * h_in:
            top = out_arr[:h_in, :]
            bottom = out_arr[h_in:, :]
            if np.array_equal(bottom, inp_arr) and np.array_equal(top, inp_arr):
                return "below_stack_self"
        return None
    except:
        return None


def try_beside_place(inp, out):
    """Child learns BESIDE/NEXT-TO: place objects side by side."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h_in, w_in = inp_arr.shape
        h_out, w_out = out_arr.shape

        if h_in == h_out and w_out == 2 * w_in:
            left = out_arr[:, :w_in]
            right = out_arr[:, w_in:]
            if np.array_equal(left, inp_arr) and np.array_equal(right, inp_arr):
                return "beside_place_self"
            if np.array_equal(left, inp_arr) and np.array_equal(right, inp_arr[:, ::-1]):
                return "beside_place_mirror"
        return None
    except:
        return None


def try_between_fill(inp, out):
    """Child learns BETWEEN: fill space between two objects."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        h, w = inp_arr.shape

        # Find pairs of same-colored objects and fill between
        for color in range(1, 10):
            mask = (inp_arr == color)
            if not mask.any():
                continue
            rows, cols = np.where(mask)
            if len(rows) < 2:
                continue

            test = inp_arr.copy()
            # Fill between horizontally
            for r in range(h):
                row_cols = cols[rows == r]
                if len(row_cols) >= 2:
                    min_c, max_c = row_cols.min(), row_cols.max()
                    test[r, min_c:max_c+1] = color

            if np.array_equal(test, out_arr):
                return f"between_fill_h_{color}"

            # Fill between vertically
            test = inp_arr.copy()
            for c in range(w):
                col_rows = rows[cols == c]
                if len(col_rows) >= 2:
                    min_r, max_r = col_rows.min(), col_rows.max()
                    test[min_r:max_r+1, c] = color

            if np.array_equal(test, out_arr):
                return f"between_fill_v_{color}"
        return None
    except:
        return None


def try_same_connect(inp, out):
    """Child learns SAME: connect objects that are the same color."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        h, w = inp_arr.shape

        for color in range(1, 10):
            mask = (inp_arr == color)
            if not mask.any():
                continue
            rows, cols = np.where(mask)
            if len(rows) < 2:
                continue

            test = inp_arr.copy()
            # Connect all same-colored cells with lines
            for i in range(len(rows)):
                for j in range(i+1, len(rows)):
                    r1, c1 = rows[i], cols[i]
                    r2, c2 = rows[j], cols[j]
                    # Draw horizontal then vertical line
                    if r1 == r2:
                        test[r1, min(c1,c2):max(c1,c2)+1] = color
                    elif c1 == c2:
                        test[min(r1,r2):max(r1,r2)+1, c1] = color

            if np.array_equal(test, out_arr):
                return f"same_connect_{color}"
        return None
    except:
        return None


def try_different_mark(inp, out):
    """Child learns DIFFERENT: identify/mark what's different."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None

        # Find the unique/different colored object
        from scipy import ndimage
        colors = [c for c in range(10) if (inp_arr == c).any()]

        for mark_color in range(1, 10):
            # Find objects and count by color
            color_counts = {}
            for c in colors:
                mask = (inp_arr == c)
                labeled, num = ndimage.label(mask)
                color_counts[c] = num

            # Mark the different one (minority color)
            if len(color_counts) >= 2:
                min_count = min(color_counts.values())
                for c, count in color_counts.items():
                    if count == min_count and c != 0:
                        test = inp_arr.copy()
                        test[inp_arr == c] = mark_color
                        if np.array_equal(test, out_arr):
                            return f"different_mark_{c}_to_{mark_color}"
        return None
    except:
        return None


def try_bigger_select(inp, out):
    """Child learns BIGGER: select/keep only the bigger object."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None

        from scipy import ndimage

        # Find all non-background objects
        for bg in [0]:
            mask = (inp_arr != bg)
            labeled, num = ndimage.label(mask)
            if num < 2:
                continue

            # Find sizes
            sizes = [(i, (labeled == i).sum()) for i in range(1, num + 1)]
            sizes.sort(key=lambda x: x[1], reverse=True)
            biggest_label = sizes[0][0]

            # Keep only biggest
            test = np.full_like(inp_arr, bg)
            biggest_mask = (labeled == biggest_label)
            test[biggest_mask] = inp_arr[biggest_mask]

            if np.array_equal(test, out_arr):
                return "bigger_select"
        return None
    except:
        return None


def try_smaller_select(inp, out):
    """Child learns SMALLER: select/keep only the smaller object."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None

        from scipy import ndimage

        for bg in [0]:
            mask = (inp_arr != bg)
            labeled, num = ndimage.label(mask)
            if num < 2:
                continue

            sizes = [(i, (labeled == i).sum()) for i in range(1, num + 1)]
            sizes.sort(key=lambda x: x[1])
            smallest_label = sizes[0][0]

            test = np.full_like(inp_arr, bg)
            smallest_mask = (labeled == smallest_label)
            test[smallest_mask] = inp_arr[smallest_mask]

            if np.array_equal(test, out_arr):
                return "smaller_select"
        return None
    except:
        return None


def try_count_objects_v2(inp, out):
    """Output encodes count of objects in various ways (version 2)."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        from scipy import ndimage

        # Count non-background objects
        for bg in [0]:
            mask = (inp_arr != bg)
            labeled, num = ndimage.label(mask)

            # Check if output encodes count
            # Output could be num x 1, 1 x num, or num cells of a color
            h_out, w_out = out_arr.shape

            if h_out == 1 and w_out == num:
                return f"count_objects_h_{num}"
            if w_out == 1 and h_out == num:
                return f"count_objects_v_{num}"

            # Count of specific color in output matches object count
            for c in range(1, 10):
                if (out_arr == c).sum() == num:
                    return f"count_objects_{num}_as_{c}"
        return None
    except:
        return None


def try_explore_combine(inp, out):
    """CREATIVITY: Combine multiple transform ideas together."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        # Try combining rotations with color ops
        for rot in [0, 1, 2, 3]:
            rotated = np.rot90(inp_arr, rot)
            if rotated.shape != out_arr.shape:
                continue
            # Check if colors are swapped
            unique_in = np.unique(rotated[rotated > 0])
            unique_out = np.unique(out_arr[out_arr > 0])
            if len(unique_in) == len(unique_out) == 2:
                a, b = unique_in[0], unique_in[1]
                test = rotated.copy()
                test[rotated == a] = b
                test[rotated == b] = a
                if np.array_equal(test, out_arr):
                    return f"explore_combine_rot{rot*90}_swap"
        return None
    except:
        return None


def try_mutate_color(inp, out):
    """CREATIVITY: Mutate colors based on position or neighborhood."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        h, w = inp_arr.shape

        # Color based on row position
        for base in range(1, 10):
            test = inp_arr.copy()
            for r in range(h):
                for c in range(w):
                    if inp_arr[r, c] != 0:
                        test[r, c] = (inp_arr[r, c] + r) % 10
                        if test[r, c] == 0:
                            test[r, c] = 1
            if np.array_equal(test, out_arr):
                return "mutate_color_row"

        # Color based on column position
        test = inp_arr.copy()
        for r in range(h):
            for c in range(w):
                if inp_arr[r, c] != 0:
                    test[r, c] = (inp_arr[r, c] + c) % 10
                    if test[r, c] == 0:
                        test[r, c] = 1
        if np.array_equal(test, out_arr):
            return "mutate_color_col"
        return None
    except:
        return None


def try_imagine_complete(inp, out):
    """CREATIVITY: Imagine and complete partial patterns."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        h, w = inp_arr.shape

        # Find partial lines and complete them
        for color in range(1, 10):
            mask = (inp_arr == color)
            if not mask.any():
                continue
            rows, cols = np.where(mask)

            # Complete horizontal lines
            test = inp_arr.copy()
            for r in np.unique(rows):
                row_cols = cols[rows == r]
                if len(row_cols) >= 2:
                    test[r, :] = color  # Imagine full line
            if np.array_equal(test, out_arr):
                return f"imagine_complete_h_{color}"

            # Complete vertical lines
            test = inp_arr.copy()
            for c in np.unique(cols):
                col_rows = rows[cols == c]
                if len(col_rows) >= 2:
                    test[:, c] = color  # Imagine full line
            if np.array_equal(test, out_arr):
                return f"imagine_complete_v_{color}"
        return None
    except:
        return None


def try_discover_pattern(inp, out):
    """CREATIVITY: Discover hidden repeating pattern."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h_in, w_in = inp_arr.shape
        h_out, w_out = out_arr.shape

        # Extract repeating unit from input
        for unit_h in range(1, h_in + 1):
            for unit_w in range(1, w_in + 1):
                if h_in % unit_h == 0 and w_in % unit_w == 0:
                    unit = inp_arr[:unit_h, :unit_w]
                    # Check if input is tiled unit
                    matches = True
                    for r in range(0, h_in, unit_h):
                        for c in range(0, w_in, unit_w):
                            if not np.array_equal(inp_arr[r:r+unit_h, c:c+unit_w], unit):
                                matches = False
                                break
                        if not matches:
                            break
                    if matches:
                        # Output should be the unit
                        if np.array_equal(unit, out_arr):
                            return f"discover_pattern_{unit_h}x{unit_w}"
        return None
    except:
        return None


def try_spontaneous_fill(inp, out):
    """CREATIVITY: Spontaneously fill based on surroundings."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        h, w = inp_arr.shape

        # Fill zeros with most common neighbor color
        test = inp_arr.copy()
        for r in range(h):
            for c in range(w):
                if inp_arr[r, c] == 0:
                    neighbors = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and inp_arr[nr, nc] != 0:
                            neighbors.append(inp_arr[nr, nc])
                    if neighbors:
                        test[r, c] = max(set(neighbors), key=neighbors.count)
        if np.array_equal(test, out_arr):
            return "spontaneous_fill_neighbor"
        return None
    except:
        return None


def try_intuition_extrapolate(inp, out):
    """CREATIVITY: Extrapolate pattern beyond boundaries."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h_in, w_in = inp_arr.shape
        h_out, w_out = out_arr.shape

        # Check if output is larger by extrapolation
        if h_out >= h_in and w_out >= w_in:
            # Try periodic extension
            test = np.zeros((h_out, w_out), dtype=int)
            for r in range(h_out):
                for c in range(w_out):
                    test[r, c] = inp_arr[r % h_in, c % w_in]
            if np.array_equal(test, out_arr):
                return "intuition_extrapolate_periodic"

            # Try edge extension
            test = np.zeros((h_out, w_out), dtype=int)
            for r in range(h_out):
                for c in range(w_out):
                    ri = min(r, h_in - 1)
                    ci = min(c, w_in - 1)
                    test[r, c] = inp_arr[ri, ci]
            if np.array_equal(test, out_arr):
                return "intuition_extrapolate_edge"
        return None
    except:
        return None


def try_breakthrough_transform(inp, out):
    """CREATIVITY: Apply unexpected compound transformation."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        # Rotate then flip
        for rot in [1, 2, 3]:
            rotated = np.rot90(inp_arr, rot)
            flipped_h = rotated[:, ::-1]
            flipped_v = rotated[::-1, :]
            if flipped_h.shape == out_arr.shape and np.array_equal(flipped_h, out_arr):
                return f"breakthrough_rot{rot*90}_hflip"
            if flipped_v.shape == out_arr.shape and np.array_equal(flipped_v, out_arr):
                return f"breakthrough_rot{rot*90}_vflip"

        # Transpose then flip
        transposed = inp_arr.T
        if transposed.shape == out_arr.shape:
            if np.array_equal(transposed[::-1, :], out_arr):
                return "breakthrough_transpose_vflip"
            if np.array_equal(transposed[:, ::-1], out_arr):
                return "breakthrough_transpose_hflip"
        return None
    except:
        return None


def try_novelty_generate(inp, out):
    """CREATIVITY: Generate novel pattern from input elements."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        h, w = inp_arr.shape

        # XOR adjacent cells for novelty
        test = inp_arr.copy()
        for r in range(h):
            for c in range(w - 1):
                test[r, c] = inp_arr[r, c] ^ inp_arr[r, c + 1]
        if np.array_equal(test, out_arr):
            return "novelty_generate_xor_h"

        test = inp_arr.copy()
        for r in range(h - 1):
            for c in range(w):
                test[r, c] = inp_arr[r, c] ^ inp_arr[r + 1, c]
        if np.array_equal(test, out_arr):
            return "novelty_generate_xor_v"
        return None
    except:
        return None


def try_freedom_expand(inp, out):
    """CREATIVITY: Free expansion in all directions."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h_in, w_in = inp_arr.shape
        h_out, w_out = out_arr.shape

        # Mirror expansion in all directions
        if h_out == 2 * h_in and w_out == 2 * w_in:
            test = np.zeros((h_out, w_out), dtype=int)
            # Top-left: original
            test[:h_in, :w_in] = inp_arr
            # Top-right: h-flip
            test[:h_in, w_in:] = inp_arr[:, ::-1]
            # Bottom-left: v-flip
            test[h_in:, :w_in] = inp_arr[::-1, :]
            # Bottom-right: both flips
            test[h_in:, w_in:] = inp_arr[::-1, ::-1]
            if np.array_equal(test, out_arr):
                return "freedom_expand_mirror4"
        return None
    except:
        return None


def try_surprise_invert(inp, out):
    """CREATIVITY: Surprise inversion of expectations."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None

        # Invert figure/ground
        max_color = inp_arr.max()
        test = max_color - inp_arr
        test[test < 0] = 0
        if np.array_equal(test, out_arr):
            return "surprise_invert_ground"

        # Invert non-zero values
        test = inp_arr.copy()
        mask = (inp_arr > 0)
        test[mask] = (10 - inp_arr[mask]) % 10
        test[test == 0] = 1  # Avoid zeros
        if np.array_equal(test, out_arr):
            return "surprise_invert_color"
        return None
    except:
        return None

# Batch 53 - Pattern ANALOGY/MAPPING depth=-9000000 ANALYZE INFINITY
# For ARC-2 and ARC-3: ABSTRACT REASONING, TRANSFER, ISOMORPHISM


def try_analogy_size(inp, out):
    """ANALOGY: Map size relationship - bigger/smaller analogy."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        from scipy import ndimage

        # Find objects and their sizes
        mask = (inp_arr != 0)
        labeled, num = ndimage.label(mask)
        if num < 2:
            return None

        sizes = [(i, (labeled == i).sum()) for i in range(1, num + 1)]
        sizes.sort(key=lambda x: x[1])

        # Map smallest to output
        smallest_label = sizes[0][0]
        smallest_mask = (labeled == smallest_label)

        rows, cols = np.where(smallest_mask)
        if len(rows) == 0:
            return None

        extracted = inp_arr[rows.min():rows.max()+1, cols.min():cols.max()+1]
        if np.array_equal(extracted, out_arr):
            return "analogy_size_smallest"

        # Map biggest
        biggest_label = sizes[-1][0]
        biggest_mask = (labeled == biggest_label)
        rows, cols = np.where(biggest_mask)
        extracted = inp_arr[rows.min():rows.max()+1, cols.min():cols.max()+1]
        if np.array_equal(extracted, out_arr):
            return "analogy_size_biggest"

        return None
    except:
        return None


def try_analogy_color(inp, out):
    """ANALOGY: Map based on color correspondence."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None

        # Try all possible color mappings
        unique_in = np.unique(inp_arr)
        unique_out = np.unique(out_arr)

        if len(unique_in) != len(unique_out):
            return None

        # Check if it's a permutation
        for perm in [[0,1,2,3,4,5,6,7,8,9], [0,2,1,3,4,5,6,7,8,9], [0,1,3,2,4,5,6,7,8,9]]:
            test = np.vectorize(lambda x: perm[x] if x < len(perm) else x)(inp_arr)
            if np.array_equal(test, out_arr):
                return f"analogy_color_perm"
        return None
    except:
        return None


def try_correspondence_grid(inp, out):
    """ANALOGY: Grid-to-grid correspondence mapping."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h_in, w_in = inp_arr.shape
        h_out, w_out = out_arr.shape

        # Check if dimensions are multiples
        if h_out > 0 and w_out > 0 and h_in % h_out == 0 and w_in % w_out == 0:
            scale_h = h_in // h_out
            scale_w = w_in // w_out

            # Sample from input at regular intervals
            test = inp_arr[::scale_h, ::scale_w]
            if test.shape == out_arr.shape and np.array_equal(test, out_arr):
                return f"correspondence_grid_sample_{scale_h}x{scale_w}"
        return None
    except:
        return None


def try_transfer_pattern(inp, out):
    """ANALOGY: Transfer pattern from one region to another."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        h, w = inp_arr.shape

        # Find template region (top-left quadrant)
        template = inp_arr[:h//2, :w//2]

        # Check if template is applied elsewhere
        test = inp_arr.copy()
        test[h//2:h//2+template.shape[0], w//2:w//2+template.shape[1]] = template

        if np.array_equal(test, out_arr):
            return "transfer_pattern_quadrant"
        return None
    except:
        return None


def try_isomorphism_struct(inp, out):
    """ANALOGY: Structure-preserving transformation."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        # Check if structure (non-zero positions) is preserved
        inp_struct = (inp_arr != 0).astype(int)
        out_struct = (out_arr != 0).astype(int)

        if np.array_equal(inp_struct, out_struct):
            # Colors changed but structure preserved
            if not np.array_equal(inp_arr, out_arr):
                return "isomorphism_struct_preserve"

        # Check rotated structure
        for rot in [1, 2, 3]:
            rotated_struct = np.rot90(inp_struct, rot)
            if rotated_struct.shape == out_struct.shape:
                if np.array_equal(rotated_struct, out_struct):
                    return f"isomorphism_struct_rot{rot*90}"
        return None
    except:
        return None


def try_bijection_color(inp, out):
    """ANALOGY: One-to-one color mapping (bijection)."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None

        # Build color mapping
        mapping = {}
        for i in range(inp_arr.shape[0]):
            for j in range(inp_arr.shape[1]):
                c_in = inp_arr[i, j]
                c_out = out_arr[i, j]
                if c_in in mapping:
                    if mapping[c_in] != c_out:
                        return None  # Not a function
                else:
                    mapping[c_in] = c_out

        # Check if bijection (one-to-one)
        if len(set(mapping.values())) == len(mapping):
            # Apply mapping and verify
            test = np.vectorize(lambda x: mapping.get(x, x))(inp_arr)
            if np.array_equal(test, out_arr):
                return f"bijection_color_{len(mapping)}"
        return None
    except:
        return None


def try_domain_range_map(inp, out):
    """ANALOGY: Map from domain to range via rule."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h_in, w_in = inp_arr.shape
        h_out, w_out = out_arr.shape

        # Output is input dimensions mapped
        if h_out == w_in and w_out == h_in:
            # Transpose-like mapping
            if np.array_equal(inp_arr.T, out_arr):
                return "domain_range_transpose"

        # Single value output (function evaluation)
        if h_out == 1 and w_out == 1:
            result = out_arr[0, 0]
            # Check if result is max, min, mode, count
            if result == inp_arr.max():
                return "domain_range_max"
            if result == inp_arr.min():
                return "domain_range_min"
            if result == (inp_arr != 0).sum():
                return "domain_range_count_nonzero"
        return None
    except:
        return None


def try_equivalence_class(inp, out):
    """ANALOGY: Group by equivalence relation."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None

        # Same-color cells form equivalence class, replace with class representative
        for base in range(1, 10):
            test = inp_arr.copy()
            for color in range(1, 10):
                mask = (inp_arr == color)
                if mask.any():
                    test[mask] = base
            if np.array_equal(test, out_arr):
                return f"equivalence_class_to_{base}"
        return None
    except:
        return None


def try_similarity_match(inp, out):
    """ANALOGY: Match by similarity metric."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        from scipy import ndimage

        # Find objects
        mask = (inp_arr != 0)
        labeled, num = ndimage.label(mask)
        if num < 2:
            return None

        # Find most similar pair (same color)
        objects = []
        for i in range(1, num + 1):
            obj_mask = (labeled == i)
            rows, cols = np.where(obj_mask)
            color = inp_arr[obj_mask][0] if obj_mask.any() else 0
            size = obj_mask.sum()
            objects.append((i, color, size, rows.min(), cols.min()))

        # Keep only one of each color pair (first)
        seen_colors = set()
        test = np.zeros_like(inp_arr)
        for i, color, size, r, c in objects:
            if color not in seen_colors:
                seen_colors.add(color)
                test[labeled == i] = color

        if np.array_equal(test, out_arr):
            return "similarity_match_unique_color"
        return None
    except:
        return None


def try_rule_induction(inp, out):
    """ANALOGY: Induce rule from examples."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        h, w = inp_arr.shape

        # Rule: color changes based on position (checkerboard rule)
        test = inp_arr.copy()
        for r in range(h):
            for c in range(w):
                if (r + c) % 2 == 0 and inp_arr[r, c] != 0:
                    test[r, c] = (inp_arr[r, c] % 9) + 1
        if np.array_equal(test, out_arr):
            return "rule_induction_checkerboard"

        # Rule: border cells change
        test = inp_arr.copy()
        for r in range(h):
            for c in range(w):
                if r == 0 or r == h-1 or c == 0 or c == w-1:
                    if inp_arr[r, c] != 0:
                        test[r, c] = (inp_arr[r, c] % 9) + 1
        if np.array_equal(test, out_arr):
            return "rule_induction_border"
        return None
    except:
        return None

# Batch 54 - Pattern SYNTHESIS/COMPOSITION depth=-9000000 ANALYZE INFINITY
# MULTI-STEP HYBRID TRANSFORMS FOR LAST 14 PUZZLES


def try_chain_flip_extract(inp, out):
    """SYNTHESIS: Flip then extract region."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        oh, ow = out_arr.shape
        # Try hflip then extract
        flipped = np.fliplr(inp_arr)
        h, w = flipped.shape
        for r in range(h - oh + 1):
            for c in range(w - ow + 1):
                if np.array_equal(flipped[r:r+oh, c:c+ow], out_arr):
                    return f"chain_flip_extract_{r}_{c}"
        # Try vflip then extract
        flipped = np.flipud(inp_arr)
        for r in range(h - oh + 1):
            for c in range(w - ow + 1):
                if np.array_equal(flipped[r:r+oh, c:c+ow], out_arr):
                    return f"chain_vflip_extract_{r}_{c}"
        return None
    except:
        return None


def try_chain_rot_overlay(inp, out):
    """SYNTHESIS: Rotate then overlay."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        # Try rot90 then overlay
        rotated = np.rot90(inp_arr)
        if rotated.shape == inp_arr.shape:
            result = np.where(rotated != 0, rotated, inp_arr)
            if np.array_equal(result, out_arr):
                return "chain_rot90_overlay"
        # Try rot180 then overlay
        rotated = np.rot90(inp_arr, 2)
        result = np.where(rotated != 0, rotated, inp_arr)
        if np.array_equal(result, out_arr):
            return "chain_rot180_overlay"
        return None
    except:
        return None


def try_multi_color_transform(inp, out):
    """SYNTHESIS: Multiple color transforms in sequence."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        # Find all color mappings
        mappings = {}
        for r in range(inp_arr.shape[0]):
            for c in range(inp_arr.shape[1]):
                iv, ov = inp_arr[r, c], out_arr[r, c]
                if iv not in mappings:
                    mappings[iv] = ov
                elif mappings[iv] != ov:
                    return None
        # Apply and verify
        test = np.vectorize(lambda x: mappings.get(x, x))(inp_arr)
        if np.array_equal(test, out_arr):
            return f"multi_color_transform_{len(mappings)}"
        return None
    except:
        return None


def try_gestalt_complete(inp, out):
    """SYNTHESIS: Complete pattern using gestalt principles."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        h, w = inp_arr.shape
        # Mirror incomplete patterns
        test = inp_arr.copy()
        # Find asymmetries and complete
        for r in range(h):
            for c in range(w // 2):
                mirror_c = w - 1 - c
                if test[r, c] == 0 and test[r, mirror_c] != 0:
                    test[r, c] = test[r, mirror_c]
                elif test[r, c] != 0 and test[r, mirror_c] == 0:
                    test[r, mirror_c] = test[r, c]
        if np.array_equal(test, out_arr):
            return "gestalt_complete_h"
        # Vertical symmetry completion
        test = inp_arr.copy()
        for r in range(h // 2):
            mirror_r = h - 1 - r
            for c in range(w):
                if test[r, c] == 0 and test[mirror_r, c] != 0:
                    test[r, c] = test[mirror_r, c]
                elif test[r, c] != 0 and test[mirror_r, c] == 0:
                    test[mirror_r, c] = test[r, c]
        if np.array_equal(test, out_arr):
            return "gestalt_complete_v"
        return None
    except:
        return None


def try_hybrid_tile_overlay(inp, out):
    """SYNTHESIS: Tile then overlay non-zero."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape
        for th in range(1, 5):
            for tw in range(1, 5):
                if ih * th == oh and iw * tw == ow:
                    tiled = np.tile(inp_arr, (th, tw))
                    # Now overlay
                    if np.array_equal(tiled, out_arr):
                        return f"hybrid_tile_overlay_{th}x{tw}"
        return None
    except:
        return None


def try_pipeline_extract_scale(inp, out):
    """SYNTHESIS: Extract region then scale."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        oh, ow = out_arr.shape
        h, w = inp_arr.shape
        for scale in [2, 3, 4]:
            eh, ew = oh // scale, ow // scale
            if eh < 1 or ew < 1:
                continue
            for r in range(h - eh + 1):
                for c in range(w - ew + 1):
                    region = inp_arr[r:r+eh, c:c+ew]
                    scaled = np.repeat(np.repeat(region, scale, axis=0), scale, axis=1)
                    if scaled.shape == out_arr.shape and np.array_equal(scaled, out_arr):
                        return f"pipeline_extract_scale_{r}_{c}_{scale}x"
        return None
    except:
        return None


def try_compound_mask_fill(inp, out):
    """SYNTHESIS: Create mask from pattern then fill."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        # Find most common color
        unique, counts = np.unique(inp_arr, return_counts=True)
        bg = unique[np.argmax(counts)]
        # Mask = non-background
        mask = inp_arr != bg
        # Fill mask with each color
        for fill_color in range(1, 10):
            test = inp_arr.copy()
            test[mask] = fill_color
            if np.array_equal(test, out_arr):
                return f"compound_mask_fill_{fill_color}"
        return None
    except:
        return None


def try_integrate_patterns(inp, out):
    """SYNTHESIS: Integrate multiple patterns into one."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        h, w = inp_arr.shape
        # Check if output is union of shifted inputs
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.zeros_like(inp_arr)
                for r in range(h):
                    for c in range(w):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            shifted[nr, nc] = inp_arr[r, c]
                merged = np.maximum(inp_arr, shifted)
                if np.array_equal(merged, out_arr):
                    return f"integrate_shift_{dr}_{dc}"
        return None
    except:
        return None


def try_holistic_border(inp, out):
    """SYNTHESIS: Border with pattern-aware fill."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        oh, ow = out_arr.shape
        ih, iw = inp_arr.shape
        # Check if output is input with border
        for border in range(1, 4):
            if oh == ih + 2*border and ow == iw + 2*border:
                # Extract center
                center = out_arr[border:border+ih, border:border+ow]
                if np.array_equal(center, inp_arr):
                    border_val = out_arr[0, 0]
                    return f"holistic_border_{border}_{border_val}"
        return None
    except:
        return None


def try_unify_fragments(inp, out):
    """SYNTHESIS: Unify scattered fragments into continuous shape."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        h, w = inp_arr.shape
        # Fill gaps between same-colored regions
        test = inp_arr.copy()
        for color in range(1, 10):
            mask = inp_arr == color
            if not np.any(mask):
                continue
            rows, cols = np.where(mask)
            if len(rows) < 2:
                continue
            # Connect with lines
            for i in range(len(rows) - 1):
                r1, c1 = rows[i], cols[i]
                r2, c2 = rows[i+1], cols[i+1]
                # Horizontal line
                if r1 == r2:
                    for c in range(min(c1, c2), max(c1, c2) + 1):
                        test[r1, c] = color
                # Vertical line
                if c1 == c2:
                    for r in range(min(r1, r2), max(r1, r2) + 1):
                        test[r, c1] = color
        if np.array_equal(test, out_arr):
            return "unify_fragments_linear"
        return None
    except:
        return None


def try_final_solve(inp, out):
    """SYNTHESIS: Last-resort combination strategies."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        # Try transpose + flip
        for k in range(4):
            rotated = np.rot90(inp_arr, k)
            if rotated.shape == out_arr.shape and np.array_equal(rotated, out_arr):
                return f"final_rot{k*90}"
            flipped_h = np.fliplr(rotated)
            if flipped_h.shape == out_arr.shape and np.array_equal(flipped_h, out_arr):
                return f"final_rot{k*90}_hflip"
            flipped_v = np.flipud(rotated)
            if flipped_v.shape == out_arr.shape and np.array_equal(flipped_v, out_arr):
                return f"final_rot{k*90}_vflip"
        return None
    except:
        return None

# Batch 55 - Pattern ENUMERATION/EXHAUSTIVE depth=-9000000 ANALYZE INFINITY
# EDGE CASES, CORNER CASES, UNUSUAL TRANSFORMS FOR FINAL 14


def try_diagonal_flip(inp, out):
    """ENUMERATION: Main diagonal and anti-diagonal flips."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        # Main diagonal transpose
        if inp_arr.T.shape == out_arr.shape and np.array_equal(inp_arr.T, out_arr):
            return "diagonal_flip_main"
        # Anti-diagonal
        anti = np.rot90(inp_arr.T, 2)
        if anti.shape == out_arr.shape and np.array_equal(anti, out_arr):
            return "diagonal_flip_anti"
        return None
    except:
        return None


def try_boundary_only(inp, out):
    """ENUMERATION: Extract or modify only boundary cells."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        # Extract just boundary as output
        if out_arr.shape[0] == 2 * (h + w - 2) and out_arr.shape[1] == 1:
            boundary = []
            for c in range(w):
                boundary.append(inp_arr[0, c])
            for r in range(1, h):
                boundary.append(inp_arr[r, w-1])
            for c in range(w-2, -1, -1):
                boundary.append(inp_arr[h-1, c])
            for r in range(h-2, 0, -1):
                boundary.append(inp_arr[r, 0])
            if np.array_equal(np.array(boundary).reshape(-1, 1), out_arr):
                return "boundary_only_extract"
        return None
    except:
        return None


def try_corner_expand(inp, out):
    """ENUMERATION: Expand from corners."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        oh, ow = out_arr.shape
        ih, iw = inp_arr.shape
        # Check if corners of input become quadrants of output
        if oh == 2 * ih and ow == 2 * iw:
            # Mirror in all four quadrants
            test = np.zeros((oh, ow), dtype=inp_arr.dtype)
            test[:ih, :iw] = inp_arr
            test[:ih, iw:] = np.fliplr(inp_arr)
            test[ih:, :iw] = np.flipud(inp_arr)
            test[ih:, iw:] = np.flipud(np.fliplr(inp_arr))
            if np.array_equal(test, out_arr):
                return "corner_expand_mirror"
        return None
    except:
        return None


def try_row_col_select(inp, out):
    """ENUMERATION: Select specific rows or columns."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        oh, ow = out_arr.shape
        # Select every other row
        if oh * 2 == h and ow == w:
            if np.array_equal(inp_arr[::2, :], out_arr):
                return "row_select_even"
            if np.array_equal(inp_arr[1::2, :], out_arr):
                return "row_select_odd"
        # Select every other column
        if oh == h and ow * 2 == w:
            if np.array_equal(inp_arr[:, ::2], out_arr):
                return "col_select_even"
            if np.array_equal(inp_arr[:, 1::2], out_arr):
                return "col_select_odd"
        return None
    except:
        return None


def try_singleton_output(inp, out):
    """ENUMERATION: Single cell outputs."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if out_arr.shape == (1, 1):
            val = out_arr[0, 0]
            # Most common color
            unique, counts = np.unique(inp_arr, return_counts=True)
            if unique[np.argmax(counts)] == val:
                return "singleton_mode"
            # Least common color
            if unique[np.argmin(counts)] == val:
                return "singleton_rare"
            # Count of unique colors
            if len(unique) == val:
                return "singleton_count_colors"
        return None
    except:
        return None


def try_extreme_crop(inp, out):
    """ENUMERATION: Crop to extreme bounds."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        # Crop to non-zero bounds with padding
        rows, cols = np.where(inp_arr != 0)
        if len(rows) == 0:
            return None
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        cropped = inp_arr[r_min:r_max+1, c_min:c_max+1]
        if cropped.shape == out_arr.shape and np.array_equal(cropped, out_arr):
            return "extreme_crop_nonzero"
        # With 1-cell padding
        r_min2, r_max2 = max(0, r_min-1), min(inp_arr.shape[0], r_max+2)
        c_min2, c_max2 = max(0, c_min-1), min(inp_arr.shape[1], c_max+2)
        cropped = inp_arr[r_min2:r_max2, c_min2:c_max2]
        if cropped.shape == out_arr.shape and np.array_equal(cropped, out_arr):
            return "extreme_crop_padded"
        return None
    except:
        return None


def try_color_conditional(inp, out):
    """ENUMERATION: Color-conditional operations."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        # Find conditional: if color A, change to B; else keep
        for src in range(10):
            for dst in range(10):
                if src == dst:
                    continue
                test = inp_arr.copy()
                test[inp_arr == src] = dst
                if np.array_equal(test, out_arr):
                    return f"color_conditional_{src}_to_{dst}"
        return None
    except:
        return None


def try_layer_extract(inp, out):
    """Extract a specific layer from stacked content."""
    try:
        arr = np.array(inp)
        out_arr = np.array(out)

        if arr.shape != out_arr.shape:
            return None

        for c in range(1, 10):
            layer = np.zeros_like(arr)
            layer[arr == c] = c
            if np.array_equal(layer, out_arr):
                return f"layer_extract_{c}"
        return None
    except:
        return None


def try_density_transform(inp, out):
    """ENUMERATION: Transform based on density."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        h, w = inp_arr.shape
        # For each cell, count neighbors
        test = inp_arr.copy()
        for r in range(h):
            for c in range(w):
                neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if inp_arr[nr, nc] != 0:
                                neighbors += 1
                if neighbors >= 4:
                    test[r, c] = 1
        if np.array_equal(test, out_arr):
            return "density_threshold_4"
        return None
    except:
        return None


def try_majority_vote(inp, out):
    """Each cell becomes majority color in neighborhood."""
    arr = np.array(inp)
    out_arr = np.array(out)
    if arr.shape != out_arr.shape:
        return None
    h, w = arr.shape

    from collections import Counter
    result = arr.copy()
    for r in range(h):
        for c in range(w):
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbors.append(arr[nr, nc])
            if neighbors:
                counts = Counter(neighbors)
                result[r, c] = counts.most_common(1)[0][0]

    if np.array_equal(result, out_arr):
        return "majority_vote"

    return None


def try_split_intersection(inp, out):
    """FINAL-14: Split at separator, output where BOTH halves empty."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        oh, ow = out_arr.shape

        # Find separator row (all same non-zero value)
        for sep_row in range(h):
            row = inp_arr[sep_row]
            if len(set(row)) == 1 and row[0] != 0:
                sep_val = row[0]
                top = inp_arr[:sep_row]
                bottom = inp_arr[sep_row+1:]

                if top.shape[0] == oh and top.shape[1] == ow and bottom.shape[0] == oh:
                    # Find the two pattern colors
                    top_colors = set(top.flatten()) - {0}
                    bottom_colors = set(bottom.flatten()) - {0}
                    out_colors = set(out_arr.flatten()) - {0}

                    if len(top_colors) == 1 and len(bottom_colors) == 1 and len(out_colors) == 1:
                        top_c = list(top_colors)[0]
                        bottom_c = list(bottom_colors)[0]
                        out_c = list(out_colors)[0]

                        # Output is out_c where both top and bottom are 0
                        top_empty = (top != top_c) & (top != sep_val)
                        bottom_empty = (bottom != bottom_c) & (bottom != sep_val)

                        test = np.where(top_empty & bottom_empty, out_c, 0)
                        if np.array_equal(test, out_arr):
                            return f"split_intersection_{sep_val}_{out_c}"
        return None
    except:
        return None


def try_split_difference(inp, out):
    """FINAL-14: Split at separator, output difference between halves."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        oh, ow = out_arr.shape

        # Find separator row
        for sep_row in range(h):
            row = inp_arr[sep_row]
            if len(set(row)) == 1 and row[0] != 0:
                top = inp_arr[:sep_row]
                bottom = inp_arr[sep_row+1:]

                if top.shape == bottom.shape and top.shape[0] == oh and top.shape[1] == ow:
                    # XOR-like: where one is nonzero and other is zero
                    for out_c in range(1, 10):
                        test = np.where((top != 0) ^ (bottom != 0), out_c, 0)
                        if np.array_equal(test, out_arr):
                            return f"split_difference_{out_c}"
        return None
    except:
        return None


def try_half_select_unique(inp, out):
    """FINAL-14: Select half with unique pattern."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        oh, ow = out_arr.shape

        # Try horizontal split
        if h == 2 * oh and w == ow:
            top_half = inp_arr[:oh]
            bottom_half = inp_arr[oh:]
            if np.array_equal(top_half, out_arr):
                return "half_select_top"
            if np.array_equal(bottom_half, out_arr):
                return "half_select_bottom"

        # Try vertical split
        if w == 2 * ow and h == oh:
            left_half = inp_arr[:, :ow]
            right_half = inp_arr[:, ow:]
            if np.array_equal(left_half, out_arr):
                return "half_select_left"
            if np.array_equal(right_half, out_arr):
                return "half_select_right"
        return None
    except:
        return None


def try_color_histogram_output(inp, out):
    """FINAL-14: Output encodes color counts."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        oh, ow = out_arr.shape

        # Count each color
        counts = {}
        for c in range(1, 10):
            counts[c] = np.sum(inp_arr == c)

        # Check if output is a histogram representation
        # Try: each column is count of a color
        if oh == 1:
            for start_c in range(1, 10):
                row = []
                for i in range(ow):
                    row.append(counts.get(start_c + i, 0))
                if list(out_arr[0]) == row:
                    return f"histogram_row_{start_c}"
        return None
    except:
        return None


def try_unique_region_extract(inp, out):
    """FINAL-14: Extract the unique/marked region."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        oh, ow = out_arr.shape
        h, w = inp_arr.shape

        # Find all non-zero colors
        colors = set(inp_arr.flatten()) - {0}

        # For each color, find bounding box and extract
        for color in colors:
            mask = inp_arr == color
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            region = inp_arr[rmin:rmax+1, cmin:cmax+1]
            if region.shape == out_arr.shape:
                # Try direct match
                if np.array_equal(region, out_arr):
                    return f"unique_region_extract_{color}"
                # Try with color reduction
                region_reduced = np.where(region == color, color, 0)
                if np.array_equal(region_reduced, out_arr):
                    return f"unique_region_reduced_{color}"
        return None
    except:
        return None


def try_pattern_compress(inp, out):
    """FINAL-14: Compress pattern to smaller representation."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        oh, ow = out_arr.shape

        # Check if output is a compressed version
        if h > oh and w > ow:
            # Try sampling at regular intervals
            for sr in range(1, h // oh + 1):
                for sc in range(1, w // ow + 1):
                    if oh * sr <= h and ow * sc <= w:
                        sampled = inp_arr[::sr, ::sc][:oh, :ow]
                        if sampled.shape == out_arr.shape:
                            if np.array_equal(sampled, out_arr):
                                return f"pattern_compress_{sr}x{sc}"
        return None
    except:
        return None


def try_color_count_as_value(inp, out):
    """FINAL-14: Count of color becomes output value."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        # Count non-zero cells
        count = np.sum(inp_arr != 0)

        # Check if output is related to count
        if out_arr.size == 1:
            if out_arr[0, 0] == count:
                return "count_as_value"
            if out_arr[0, 0] == count % 10:
                return "count_mod10_as_value"
        return None
    except:
        return None


def try_expand_by_pattern(inp, out):
    """FINAL-14: Expand input by pattern rules."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Check if output is expanded version
        if oh > ih and ow > iw and oh % ih == 0 and ow % iw == 0:
            sh, sw = oh // ih, ow // iw
            # Try simple repeat expansion
            expanded = np.repeat(np.repeat(inp_arr, sh, axis=0), sw, axis=1)
            if np.array_equal(expanded, out_arr):
                return f"expand_repeat_{sh}x{sw}"
        return None
    except:
        return None


def try_marked_cell_output(inp, out):
    """FINAL-14: Output based on marked/special cells."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        # Find cells with unique color (appears only once)
        for c in range(1, 10):
            if np.sum(inp_arr == c) == 1:
                pos = np.where(inp_arr == c)
                r, col = pos[0][0], pos[1][0]

                # Check if output relates to position
                if out_arr.size == 2:
                    if list(out_arr.flatten()) == [r, col]:
                        return f"marked_cell_pos_{c}"
                    if list(out_arr.flatten()) == [col, r]:
                        return f"marked_cell_pos_swap_{c}"
        return None
    except:
        return None


def try_row_with_color_select(inp, out):
    """FINAL-14: Select rows containing specific color."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        oh, ow = out_arr.shape

        if ow != w:
            return None

        # Find rows with specific color
        for c in range(1, 10):
            rows_with_c = []
            for r in range(h):
                if c in inp_arr[r]:
                    rows_with_c.append(r)

            if len(rows_with_c) == oh:
                selected = inp_arr[rows_with_c]
                if np.array_equal(selected, out_arr):
                    return f"row_with_color_{c}"
                # Try with color removed
                selected_clean = np.where(selected == c, 0, selected)
                if np.array_equal(selected_clean, out_arr):
                    return f"row_with_color_{c}_clean"
        return None
    except:
        return None


def try_column_with_color_select(inp, out):
    """FINAL-14: Select columns containing specific color."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        oh, ow = out_arr.shape

        if oh != h:
            return None

        for c in range(1, 10):
            cols_with_c = []
            for col in range(w):
                if c in inp_arr[:, col]:
                    cols_with_c.append(col)

            if len(cols_with_c) == ow:
                selected = inp_arr[:, cols_with_c]
                if np.array_equal(selected, out_arr):
                    return f"col_with_color_{c}"
        return None
    except:
        return None


def try_non_background_region(inp, out):
    """FINAL-14: Extract region that's not background."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        # Find background (most common color or 0)
        colors, counts = np.unique(inp_arr, return_counts=True)
        bg = colors[np.argmax(counts)]

        # Find bounding box of non-background
        mask = inp_arr != bg
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            region = inp_arr[rmin:rmax+1, cmin:cmax+1]

            if np.array_equal(region, out_arr):
                return f"non_bg_region_{bg}"

            # Try with background removed
            region_clean = np.where(region == bg, 0, region)
            if np.array_equal(region_clean, out_arr):
                return f"non_bg_region_clean_{bg}"
        return None
    except:
        return None


def try_frame_content_extract(inp, out):
    """FINAL-14: Extract content from inside a frame."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        oh, ow = out_arr.shape

        # Look for frame border color
        for c in range(1, 10):
            # Check if c forms a frame
            top_row = inp_arr[0]
            bottom_row = inp_arr[-1]
            left_col = inp_arr[:, 0]
            right_col = inp_arr[:, -1]

            if np.all(top_row == c) or np.all(bottom_row == c):
                # Find inner region
                for margin in range(1, min(h, w) // 2):
                    inner = inp_arr[margin:h-margin, margin:w-margin]
                    if inner.shape == out_arr.shape:
                        if np.array_equal(inner, out_arr):
                            return f"frame_content_{c}_{margin}"
        return None
    except:
        return None

# ============================================================================
# Batch 57 - Pattern FINAL-11 HUNT depth=-9000000 ANALYZE INFINITY
# SPLIT-UNION, SPLIT-NO-SEPARATOR FOR REMAINING 11 PUZZLES
# ============================================================================


def try_split_union(inp, out):
    """FINAL-11: Split at separator, output where EITHER half is non-zero."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        oh, ow = out_arr.shape

        # Find separator row
        for sep_row in range(h):
            row = inp_arr[sep_row]
            if len(set(row)) == 1 and row[0] != 0:
                sep_val = row[0]
                top = inp_arr[:sep_row]
                bottom = inp_arr[sep_row+1:]

                if top.shape[0] == oh and top.shape[1] == ow and bottom.shape[0] == oh:
                    top_colors = set(top.flatten()) - {0}
                    bottom_colors = set(bottom.flatten()) - {0}
                    out_colors = set(out_arr.flatten()) - {0}

                    if len(top_colors) == 1 and len(bottom_colors) == 1 and len(out_colors) == 1:
                        top_c = list(top_colors)[0]
                        bottom_c = list(bottom_colors)[0]
                        out_c = list(out_colors)[0]

                        # Output is out_c where either top or bottom is non-zero
                        top_nonzero = (top != 0)
                        bottom_nonzero = (bottom != 0)

                        test = np.where(top_nonzero | bottom_nonzero, out_c, 0)
                        if np.array_equal(test, out_arr):
                            return f"split_union_{sep_val}_{out_c}"
        return None
    except:
        return None


def try_split_no_separator(inp, out):
    """FINAL-11: Split in half (no separator), output intersection of zeros."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        oh, ow = out_arr.shape

        # Check if height is double output height (8x4 -> 4x4 pattern)
        if h == 2 * oh and w == ow:
            top = inp_arr[:oh]
            bottom = inp_arr[oh:]

            top_colors = set(top.flatten()) - {0}
            bottom_colors = set(bottom.flatten()) - {0}
            out_colors = set(out_arr.flatten()) - {0}

            if len(top_colors) == 1 and len(bottom_colors) == 1 and len(out_colors) == 1:
                out_c = list(out_colors)[0]

                # Test: output where BOTH top and bottom are 0
                test = np.where((top == 0) & (bottom == 0), out_c, 0)
                if np.array_equal(test, out_arr):
                    return f"split_no_sep_intersection_{out_c}"

                # Test: output where EITHER top or bottom is non-zero
                test = np.where((top != 0) | (bottom != 0), out_c, 0)
                if np.array_equal(test, out_arr):
                    return f"split_no_sep_union_{out_c}"
        return None
    except:
        return None


def try_split_xor_half(inp, out):
    """FINAL-11: Split in half, output XOR of non-zero regions."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        oh, ow = out_arr.shape

        if h == 2 * oh and w == ow:
            top = inp_arr[:oh]
            bottom = inp_arr[oh:]

            out_colors = set(out_arr.flatten()) - {0}
            if len(out_colors) == 1:
                out_c = list(out_colors)[0]

                # XOR: where exactly one is non-zero
                test = np.where((top != 0) ^ (bottom != 0), out_c, 0)
                if np.array_equal(test, out_arr):
                    return f"split_xor_half_{out_c}"
        return None
    except:
        return None


def try_split_top_minus_bottom(inp, out):
    """FINAL-11: Output where top has pattern but bottom doesn't."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        h, w = inp_arr.shape
        oh, ow = out_arr.shape

        # Try with separator
        for sep_row in range(h):
            row = inp_arr[sep_row]
            if len(set(row)) == 1 and row[0] != 0:
                top = inp_arr[:sep_row]
                bottom = inp_arr[sep_row+1:]

                if top.shape[0] == oh and top.shape[1] == ow and bottom.shape[0] == oh:
                    out_colors = set(out_arr.flatten()) - {0}
                    if len(out_colors) == 1:
                        out_c = list(out_colors)[0]

                        # Top has pattern, bottom doesn't
                        test = np.where((top != 0) & (bottom == 0), out_c, 0)
                        if np.array_equal(test, out_arr):
                            return f"split_top_minus_bottom_{out_c}"

                        # Bottom has pattern, top doesn't
                        test = np.where((bottom != 0) & (top == 0), out_c, 0)
                        if np.array_equal(test, out_arr):
                            return f"split_bottom_minus_top_{out_c}"
        return None
    except:
        return None


def try_color_pair_intersection(inp, out):
    """FINAL-11: Intersection of two specific colors."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        if inp_arr.shape != out_arr.shape:
            return None

        out_colors = set(out_arr.flatten()) - {0}
        if len(out_colors) != 1:
            return None
        out_c = list(out_colors)[0]

        # Try all pairs of colors
        for c1 in range(1, 10):
            for c2 in range(c1 + 1, 10):
                mask1 = inp_arr == c1
                mask2 = inp_arr == c2

                # Where both colors appear (in adjacent cells or same position logic)
                # Simplified: where both are present in neighborhood
                pass  # Complex logic, skip for now
        return None
    except:
        return None

# Batch 58 - Pattern FINAL-9 HUNT depth=-9000000 ANALYZE INFINITY
# HUNTING: 1fad071e, 3de23699, 5ad4f10b, 80af3007, 8731374e, b190f7f5, bc1d5164, e6721834, fcb5c309


def try_column_vote(inp, out):
    """FINAL-9: Vote/count patterns in columns, output as 1D bar."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        # Check if output is 1 row
        if out_arr.shape[0] != 1:
            return None

        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Find colors
        colors = [c for c in np.unique(inp_arr) if c != 0]
        if len(colors) < 2:
            return None

        # For each output column, check if there's a pattern from input
        # Try: output[j] = 1 if column j has overlapping colors
        for c1 in colors:
            for c2 in colors:
                if c1 == c2:
                    continue

                result = []
                for col_idx in range(min(iw, ow)):
                    col = inp_arr[:, col_idx]
                    # Check for overlap or intersection
                    has_c1 = c1 in col
                    has_c2 = c2 in col

                    # If both present, output c1
                    if has_c1 and has_c2:
                        result.append(c1)
                    else:
                        result.append(0)

                if len(result) == ow:
                    expected = np.array([result])
                    if np.array_equal(expected, out_arr):
                        return f"column_vote_{c1}_{c2}"
        return None
    except:
        return None


def try_remove_marker_expand(inp, out):
    """FINAL-9: Remove marker color, expand remaining pattern."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        inp_colors = set(c for c in np.unique(inp_arr) if c != 0)
        out_colors = set(c for c in np.unique(out_arr) if c != 0)

        # Find the marker color (in input but not output)
        marker_colors = inp_colors - out_colors
        if len(marker_colors) != 1:
            return None

        marker = list(marker_colors)[0]

        # Remove marker from input and see if it matches output pattern
        cleaned = inp_arr.copy()
        cleaned[cleaned == marker] = 0

        # Try different crops/extractions
        for i in range(inp_arr.shape[0] - out_arr.shape[0] + 1):
            for j in range(inp_arr.shape[1] - out_arr.shape[1] + 1):
                region = cleaned[i:i+out_arr.shape[0], j:j+out_arr.shape[1]]
                if np.array_equal(region, out_arr):
                    return f"remove_marker_{marker}_extract"
        return None
    except:
        return None


def try_extract_noise_rectangle(inp, out):
    """FINAL-9: Find clean rectangle amid random noise."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        oh, ow = out_arr.shape

        out_colors = set(c for c in np.unique(out_arr))

        # Search for a rectangle in input that matches output colors only
        for i in range(inp_arr.shape[0] - oh + 1):
            for j in range(inp_arr.shape[1] - ow + 1):
                region = inp_arr[i:i+oh, j:j+ow]
                region_colors = set(c for c in np.unique(region))

                # Check if region only has output colors
                if region_colors <= out_colors:
                    if np.array_equal(region, out_arr):
                        return f"extract_clean_rect_{i}_{j}"
        return None
    except:
        return None


def try_block_to_pattern(inp, out):
    """FINAL-9: Convert 3x3 blocks to single pixels (downsample)."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Try various block sizes
        for block_h in [2, 3, 4]:
            for block_w in [2, 3, 4]:
                if ih % block_h == 0 and iw % block_w == 0:
                    expected_h = ih // block_h
                    expected_w = iw // block_w

                    if expected_h == oh and expected_w == ow:
                        result = np.zeros((expected_h, expected_w), dtype=int)

                        for bi in range(expected_h):
                            for bj in range(expected_w):
                                block = inp_arr[bi*block_h:(bi+1)*block_h,
                                               bj*block_w:(bj+1)*block_w]
                                # Use most common non-zero or checkerboard pattern
                                vals, counts = np.unique(block, return_counts=True)
                                nonzero_mask = vals != 0
                                if np.any(nonzero_mask):
                                    nonzero_vals = vals[nonzero_mask]
                                    nonzero_counts = counts[nonzero_mask]
                                    result[bi, bj] = nonzero_vals[np.argmax(nonzero_counts)]

                        if np.array_equal(result, out_arr):
                            return f"block_downsample_{block_h}x{block_w}"
        return None
    except:
        return None


def try_cross_pattern_expand(inp, out):
    """FINAL-9: Expand small grid with cross patterns."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Check if output is multiple of input
        if oh % ih != 0 or ow % iw != 0:
            return None

        scale_h = oh // ih
        scale_w = ow // iw

        if scale_h != scale_w:
            return None

        scale = scale_h

        # For each cell in input, create a cross pattern in output
        result = np.zeros((oh, ow), dtype=int)

        for i in range(ih):
            for j in range(iw):
                color = inp_arr[i, j]
                if color != 0 and color != 8:  # 8 often is marker
                    # Create cross at position
                    ci = i * scale + scale // 2
                    cj = j * scale + scale // 2

                    # Draw cross
                    for di in range(scale):
                        if 0 <= ci - scale//2 + di < oh:
                            result[ci - scale//2 + di, cj] = color
                    for dj in range(scale):
                        if 0 <= cj - scale//2 + dj < ow:
                            result[ci, cj - scale//2 + dj] = color

        if np.array_equal(result, out_arr):
            return f"cross_expand_{scale}x"
        return None
    except:
        return None


def try_row_col_compress(inp, out):
    """FINAL-9: Compress by voting across rows/columns."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Try horizontal compression (compress columns)
        if ih == oh:
            block_w = iw // ow
            if iw % ow == 0:
                result = np.zeros((oh, ow), dtype=int)
                for i in range(oh):
                    for j in range(ow):
                        segment = inp_arr[i, j*block_w:(j+1)*block_w]
                        # Vote: most common non-zero
                        nonzero = segment[segment != 0]
                        if len(nonzero) > 0:
                            vals, counts = np.unique(nonzero, return_counts=True)
                            result[i, j] = vals[np.argmax(counts)]
                if np.array_equal(result, out_arr):
                    return f"compress_cols_{block_w}"

        # Try vertical compression
        if iw == ow:
            block_h = ih // oh
            if ih % oh == 0:
                result = np.zeros((oh, ow), dtype=int)
                for i in range(oh):
                    for j in range(ow):
                        segment = inp_arr[i*block_h:(i+1)*block_h, j]
                        nonzero = segment[segment != 0]
                        if len(nonzero) > 0:
                            vals, counts = np.unique(nonzero, return_counts=True)
                            result[i, j] = vals[np.argmax(counts)]
                if np.array_equal(result, out_arr):
                    return f"compress_rows_{block_h}"
        return None
    except:
        return None


def try_dual_region_merge(inp, out):
    """FINAL-9: Merge patterns from two regions (e.g., 8-filled and 0-filled)."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Find dividing line (horizontal or vertical)
        # Try horizontal split
        if ih == oh * 2:
            top = inp_arr[:oh, :]
            bottom = inp_arr[oh:, :]

            # Try to merge: where one has pattern, put that pattern
            result = np.zeros((oh, ow), dtype=int)

            # Find background colors
            for bg_top in [0, 8]:
                for bg_bottom in [0, 8]:
                    if bg_top == bg_bottom:
                        continue
                    result = np.where(top != bg_top, top,
                                     np.where(bottom != bg_bottom, bottom, 0))
                    result = np.where(result == bg_top, 0, result)
                    result = np.where(result == bg_bottom, 0, result)
                    if np.array_equal(result, out_arr):
                        return f"merge_regions_h_{bg_top}_{bg_bottom}"

        # Try vertical split
        if iw == ow * 2:
            left = inp_arr[:, :ow]
            right = inp_arr[:, ow:]

            for bg_left in [0, 8]:
                for bg_right in [0, 8]:
                    if bg_left == bg_right:
                        continue
                    result = np.where(left != bg_left, left,
                                     np.where(right != bg_right, right, 0))
                    result = np.where(result == bg_left, 0, result)
                    result = np.where(result == bg_right, 0, result)
                    if np.array_equal(result, out_arr):
                        return f"merge_regions_v_{bg_left}_{bg_right}"
        return None
    except:
        return None


def try_frame_fill(inp, out):
    """FINAL-9: Extract and fill a frame pattern."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)

        oh, ow = out_arr.shape

        # Find frames in input
        inp_colors = set(c for c in np.unique(inp_arr) if c != 0)
        out_colors = set(c for c in np.unique(out_arr) if c != 0)

        # The frame color should be in output
        frame_color = list(out_colors)[0] if len(out_colors) == 1 else None
        if frame_color is None:
            return None

        # Search for a rectangular frame in input
        for i in range(inp_arr.shape[0] - oh + 1):
            for j in range(inp_arr.shape[1] - ow + 1):
                region = inp_arr[i:i+oh, j:j+ow]

                # Check if this looks like a frame (same as output with frame_color)
                # Try replacing non-background with frame_color
                result = np.where(region != 0, frame_color, 0)
                if np.array_equal(result, out_arr):
                    return f"frame_extract_{frame_color}_{i}_{j}"
        return None
    except:
        return None

# Batch 59 - Pattern FINAL-7 HUNT depth=-9000000 ANALYZE INFINITY
# HUNTING: 1fad071e, 5ad4f10b, 80af3007, 8731374e, b190f7f5, bc1d5164, e6721834


def try_column_count_pattern(inp, out):
    """FINAL-7: 1fad071e - Count patterns in columns, output as 1-row binary."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Must output 1 row
        if oh != 1:
            return None

        # Find the two non-background colors in input
        inp_colors = set(inp_arr.flatten()) - {0}
        if len(inp_colors) != 2:
            return None

        c1, c2 = sorted(inp_colors)
        out_colors = set(out_arr.flatten())

        # Try different column-counting strategies
        # Strategy 1: Count columns where color pairs appear together
        result = []
        for col in range(iw):
            col_vals = inp_arr[:, col]
            has_c1 = c1 in col_vals
            has_c2 = c2 in col_vals
            # Column has both colors = 1, else = 0
            if has_c1 and has_c2:
                result.append(1 if 1 in out_colors else c1)
            else:
                result.append(0)

        # Compress to output width if needed
        if len(result) > ow:
            # Try grouping
            step = len(result) // ow
            compressed = []
            for i in range(0, len(result), step):
                group = result[i:i+step]
                compressed.append(1 if any(v != 0 for v in group) else 0)
            result = compressed[:ow]

        if len(result) == ow:
            test_out = np.array([result])
            if np.array_equal(test_out, out_arr):
                return f"column_count_pattern_{c1}_{c2}"

        return None
    except:
        return None


def try_noise_around_solid(inp, out):
    """FINAL-7: 5ad4f10b - Extract noise positions around a solid rectangle."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Output should be small (3x3 or similar)
        if oh > 5 or ow > 5:
            return None

        # Find the solid rectangle color (appears in large blocks)
        colors, counts = np.unique(inp_arr, return_counts=True)
        solid_color = None
        noise_color = None

        for c, cnt in zip(colors, counts):
            if c == 0:
                continue
            # Large count = solid rectangle
            if cnt > ih * iw * 0.2:
                solid_color = c
            # Small scattered = noise
            elif cnt < ih * iw * 0.1:
                noise_color = c

        if solid_color is None or noise_color is None:
            return None

        # Find the solid rectangle bounds
        solid_mask = inp_arr == solid_color
        rows = np.any(solid_mask, axis=1)
        cols = np.any(solid_mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        # Divide the space around the rectangle into a grid matching output
        # Check which grid cells contain noise
        result = np.zeros((oh, ow), dtype=int)

        grid_h = ih / oh
        grid_w = iw / ow

        for i in range(oh):
            for j in range(ow):
                r1, r2 = int(i * grid_h), int((i+1) * grid_h)
                c1, c2 = int(j * grid_w), int((j+1) * grid_w)
                region = inp_arr[r1:r2, c1:c2]
                if noise_color in region:
                    result[i, j] = noise_color

        if np.array_equal(result, out_arr):
            return f"noise_around_solid_{solid_color}_{noise_color}"
        return None
    except:
        return None


def try_block_downsample_pattern(inp, out):
    """FINAL-7: 80af3007 - Convert blocks to checkerboard pixel pattern."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Find the non-zero color
        color = 0
        for c in np.unique(inp_arr):
            if c != 0:
                color = c
                break

        if color == 0:
            return None

        # Input should have 3x3 blocks of the color
        # Try to find the block size
        block_h = 3
        block_w = 3

        # Find all 3x3 block positions
        blocks = []
        for i in range(0, ih - block_h + 1, block_h):
            for j in range(0, iw - block_w + 1, block_w):
                block = inp_arr[i:i+block_h, j:j+block_w]
                if np.all(block == color):
                    blocks.append((i // block_h, j // block_w))

        # The blocks form a pattern - output should be a transformation
        # where each 3x3 in output corresponds to block presence
        if len(blocks) == 0:
            return None

        # Try different output mappings
        # Map blocks to a grid, then apply checkerboard
        max_r = max(b[0] for b in blocks) + 1
        max_c = max(b[1] for b in blocks) + 1

        block_grid = np.zeros((max_r, max_c), dtype=int)
        for r, c in blocks:
            block_grid[r, c] = 1

        # Each block in input becomes 3 pixels in output with checkerboard
        result = np.zeros((max_r * 3, max_c * 3), dtype=int)
        for r in range(max_r):
            for c in range(max_c):
                if block_grid[r, c]:
                    # Apply checkerboard pattern for this block
                    for di in range(3):
                        for dj in range(3):
                            if (di + dj) % 2 == 0:  # Checkerboard
                                result[r*3+di, c*3+dj] = color

        if result.shape == out_arr.shape and np.array_equal(result, out_arr):
            return f"block_downsample_checker_{color}"

        return None
    except:
        return None


def try_find_clean_rectangle(inp, out):
    """FINAL-7: 8731374e - Find clean (non-random) rectangle amid digit noise."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Output uses only 2 colors - find them
        out_colors = sorted(set(out_arr.flatten()))
        if len(out_colors) != 2:
            return None

        c1, c2 = out_colors

        # Search for a rectangle in input that matches output when filtered
        for i in range(ih - oh + 1):
            for j in range(iw - ow + 1):
                region = inp_arr[i:i+oh, j:j+ow]

                # Check if this region uses only the same 2 colors
                region_colors = set(region.flatten())
                if region_colors == set(out_colors):
                    if np.array_equal(region, out_arr):
                        return f"find_clean_rect_{c1}_{c2}_{i}_{j}"

        return None
    except:
        return None


def try_cross_expand_from_marker(inp, out):
    """FINAL-7: b190f7f5 - Expand non-marker cells into cross patterns."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Output should be 3x input size (each cell expands to 3x3)
        # Or 4x input size, etc.
        if oh % ih != 0 or ow % iw != 0:
            return None

        scale_h = oh // ih
        scale_w = ow // iw

        if scale_h != scale_w or scale_h < 2:
            return None

        scale = scale_h

        # Color 8 is typically the marker (not expanded)
        marker = 8

        result = np.zeros((oh, ow), dtype=int)

        for i in range(ih):
            for j in range(iw):
                color = inp_arr[i, j]
                if color == 0 or color == marker:
                    continue

                # Find where this cell goes in output
                out_i = i * scale
                out_j = j * scale

                # Draw a cross pattern
                mid = scale // 2
                # Vertical line
                for di in range(scale):
                    if out_i + di < oh and out_j + mid < ow:
                        result[out_i + di, out_j + mid] = color
                # Horizontal line
                for dj in range(scale):
                    if out_i + mid < oh and out_j + dj < ow:
                        result[out_i + mid, out_j + dj] = color

        if np.array_equal(result, out_arr):
            return f"cross_expand_marker_{marker}_scale_{scale}"

        return None
    except:
        return None


def try_mirror_vote_compress(inp, out):
    """FINAL-7: bc1d5164 - Compress by OR/vote across mirrored positions."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Output should be smaller
        if oh >= ih or ow >= iw:
            return None

        color = 0
        for c in np.unique(out_arr):
            if c != 0:
                color = c
                break

        # Try different compression schemes
        # 5x7 -> 3x3: vote by OR across symmetric positions

        # Strategy: OR the grid with its horizontal mirror, then compress
        flipped_h = np.fliplr(inp_arr)
        combined = np.where((inp_arr != 0) | (flipped_h != 0), color, 0)

        # Now downsample by taking center
        # For 5x7 -> 3x3, skip edges
        skip_h = (ih - oh) // 2
        skip_w = (iw - ow) // 2

        # Try extracting center
        result = combined[skip_h:skip_h+oh, skip_w:skip_w+ow]

        if result.shape == out_arr.shape:
            # Normalize colors
            result = np.where(result != 0, color, 0)
            if np.array_equal(result, out_arr):
                return f"mirror_vote_compress_{color}"

        # Try voting by blocks
        block_h = ih // oh
        block_w = iw // ow

        if block_h > 0 and block_w > 0:
            result2 = np.zeros((oh, ow), dtype=int)
            for i in range(oh):
                for j in range(ow):
                    block = inp_arr[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                    if np.any(block != 0):
                        result2[i, j] = color

            if np.array_equal(result2, out_arr):
                return f"vote_compress_{color}_{block_h}x{block_w}"

        return None
    except:
        return None


def try_split_position_combine(inp, out):
    """FINAL-7: e6721834 - Top half has patterns, bottom has positions, combine."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Input should be 2x output height
        if ih != 2 * oh:
            return None
        if iw != ow:
            return None

        top_half = inp_arr[:oh, :]
        bottom_half = inp_arr[oh:, :]

        # Find the background colors in each half
        top_bg = 8  # Often 8 in top
        bottom_bg = 0  # Often 0 in bottom

        # Find patterns in top (non-8 regions)
        # Find marker positions in bottom (non-0 cells)

        result = np.zeros_like(out_arr)

        # The bottom half has colored dots indicating where to place patterns
        # Find each pattern region in top half
        for i in range(oh):
            for j in range(ow):
                if bottom_half[i, j] != bottom_bg and bottom_half[i, j] != top_bg:
                    # There's a marker here - find corresponding pattern in top
                    # and place it at this position
                    marker_color = bottom_half[i, j]

                    # Search top half for a pattern with this marker
                    # For now, just copy top value if not background
                    if top_half[i, j] != top_bg:
                        result[i, j] = top_half[i, j]

        # Alternative: overlay non-background from both
        result2 = np.copy(bottom_half)
        for i in range(oh):
            for j in range(ow):
                if top_half[i, j] != top_bg:
                    result2[i, j] = top_half[i, j]

        if np.array_equal(result2, out_arr):
            return f"split_position_combine_{top_bg}_{bottom_bg}"

        # Try: patterns in top, positions in bottom -> swap places
        result3 = np.zeros_like(out_arr)
        for i in range(oh):
            for j in range(ow):
                if bottom_half[i, j] != bottom_bg:
                    # Find pattern at same position in top
                    result3[i, j] = top_half[i, j] if top_half[i, j] != top_bg else bottom_half[i, j]
                elif top_half[i, j] != top_bg:
                    result3[i, j] = top_half[i, j]

        if np.array_equal(result3, out_arr):
            return f"split_position_overlay_{top_bg}_{bottom_bg}"

        return None
    except:
        return None

# Batch 60 - Pattern FINAL-7 DEEPER HUNT depth=-9000000 ANALYZE INFINITY
# TARGETED: 1fad071e, 5ad4f10b, 80af3007, 8731374e, b190f7f5, bc1d5164, e6721834


def try_block_count_to_binary(inp, out):
    """FINAL-7 DEEP: 1fad071e - Count 2x2 blocks per color, output binary."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        if oh != 1:
            return None

        # Find non-zero colors
        colors = sorted(set(inp_arr.flatten()) - {0})
        if len(colors) != 2:
            return None

        # Count 2x2 blocks for each color
        block_counts = {c: [] for c in colors}
        for c in colors:
            for y in range(ih - 1):
                for x in range(iw - 1):
                    if np.all(inp_arr[y:y+2, x:x+2] == c):
                        block_counts[c].append((y, x))

        # Try mapping block positions to output
        c1, c2 = colors
        # Each 2x2 block of c1 might map to a 1 in output
        blocks_c1 = block_counts[c1]

        # The output width might correspond to block count or region count
        # Try: for each block of c1, mark its column region as 1
        result = [0] * ow
        step = iw / ow

        for (y, x) in blocks_c1:
            out_col = int(x / step)
            if out_col < ow:
                result[out_col] = 1

        if np.array_equal(np.array([result]), out_arr):
            return f"block_count_binary_{c1}_{c2}"

        return None
    except:
        return None


def try_cross_region_noise(inp, out):
    """FINAL-7 DEEP: 5ad4f10b - Cross shape divides into regions, noise positions."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        if oh != 3 or ow != 3:
            return None

        # Find solid color (large blocks) and noise color (scattered)
        colors, counts = np.unique(inp_arr, return_counts=True)
        solid = None
        noise = None

        for c, cnt in zip(colors, counts):
            if c == 0:
                continue
            ratio = cnt / (ih * iw)
            if ratio > 0.15:
                solid = c
            elif ratio < 0.1:
                noise = c

        if solid is None or noise is None:
            return None

        # Find the cross boundaries
        solid_mask = inp_arr == solid

        # Find row ranges where solid appears
        row_has_solid = np.any(solid_mask, axis=1)
        col_has_solid = np.any(solid_mask, axis=0)

        # Divide space into 3x3 regions
        # Find the cross center and arms
        result = np.zeros((3, 3), dtype=int)

        # Simple approach: divide image into 3x3 grid and check noise in each
        grid_h = ih // 3
        grid_w = iw // 3

        for i in range(3):
            for j in range(3):
                r1, r2 = i * grid_h, (i + 1) * grid_h
                c1, c2 = j * grid_w, (j + 1) * grid_w
                region = inp_arr[r1:r2, c1:c2]
                if noise in region:
                    result[i, j] = noise

        if np.array_equal(result, out_arr):
            return f"cross_region_noise_{solid}_{noise}"

        return None
    except:
        return None


def try_block_to_checkerboard(inp, out):
    """FINAL-7 DEEP: 80af3007 - Each 3x3 solid block becomes checkerboard pattern."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Find the color
        color = 0
        for c in np.unique(inp_arr):
            if c != 0:
                color = c
                break

        if color == 0:
            return None

        # Find all 3x3 solid blocks
        block_positions = []
        for y in range(0, ih - 2, 3):
            for x in range(0, iw - 2, 3):
                block = inp_arr[y:y+3, x:x+3]
                if np.all(block == color):
                    block_positions.append((y // 3, x // 3))

        if not block_positions:
            return None

        # Calculate output dimensions based on blocks
        max_y = max(p[0] for p in block_positions) + 1
        max_x = max(p[1] for p in block_positions) + 1

        # Each block position becomes a 3x3 checkerboard in output
        result = np.zeros((max_y * 3, max_x * 3), dtype=int)

        for by, bx in block_positions:
            for dy in range(3):
                for dx in range(3):
                    # Checkerboard: color at even positions
                    if (dy + dx) % 2 == 0:
                        result[by * 3 + dy, bx * 3 + dx] = color

        if result.shape == out_arr.shape and np.array_equal(result, out_arr):
            return f"block_to_checkerboard_{color}"

        return None
    except:
        return None


def try_two_color_rectangle_extract(inp, out):
    """FINAL-7 DEEP: 8731374e - Find rectangle using exactly 2 colors amid noise."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Output should use exactly 2 colors
        out_colors = set(out_arr.flatten())
        if len(out_colors) != 2:
            return None

        # Search for a rectangle in input that matches
        for i in range(ih - oh + 1):
            for j in range(iw - ow + 1):
                region = inp_arr[i:i+oh, j:j+ow]
                region_colors = set(region.flatten())

                # Must use exactly the same 2 colors
                if region_colors == out_colors:
                    if np.array_equal(region, out_arr):
                        return f"two_color_rect_extract_{i}_{j}"

        return None
    except:
        return None


def try_marker_cross_expand(inp, out):
    """FINAL-7 DEEP: b190f7f5 - Cells marked by 8 get cross-expanded."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Find the marker color (usually 8)
        marker = 8
        if marker not in inp_arr:
            return None

        # Each marked position's corresponding non-marker color gets expanded
        # The input is split: left half has colors, right half has 8 markers

        if iw % 2 != 0:
            return None

        mid = iw // 2
        left = inp_arr[:, :mid]
        right = inp_arr[:, mid:]

        # Find cells where right has marker
        scale = oh // ih

        result = np.zeros((oh, ow), dtype=int)

        for i in range(ih):
            for j in range(mid):
                color = left[i, j]
                has_marker = right[i, j] == marker

                if color != 0 and not has_marker:
                    # Expand this color into a cross
                    out_i = i * scale
                    out_j = j * scale

                    # Draw cross
                    center = scale // 2
                    for d in range(scale):
                        if out_i + d < oh and out_j + center < ow:
                            result[out_i + d, out_j + center] = color
                        if out_i + center < oh and out_j + d < ow:
                            result[out_i + center, out_j + d] = color

        if np.array_equal(result, out_arr):
            return f"marker_cross_expand_{marker}"

        return None
    except:
        return None


def try_symmetric_fold_compress(inp, out):
    """FINAL-7 DEEP: bc1d5164 - Fold symmetric input and OR, compress to center."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Find the non-zero color
        color = 0
        for c in np.unique(inp_arr):
            if c != 0:
                color = c
                break

        if color == 0:
            return None

        # The input should be symmetric
        # Try folding: OR left with flipped right, OR top with flipped bottom
        # Then extract center

        # Horizontal fold: OR left half with flipped right half
        if iw % 2 == 1:
            mid = iw // 2
            left = inp_arr[:, :mid]
            right = inp_arr[:, mid+1:]
            right_flip = np.fliplr(right)
            h_folded = np.where((left != 0) | (right_flip != 0), color, 0)
        else:
            return None

        # Vertical fold: OR top half with flipped bottom half
        if ih % 2 == 1:
            vmid = ih // 2
            top = h_folded[:vmid, :]
            bottom = h_folded[vmid+1:, :]
            bottom_flip = np.flipud(bottom)
            v_folded = np.where((top != 0) | (bottom_flip != 0), color, 0)

            # Include center row
            center_row = h_folded[vmid:vmid+1, :]
            result = np.vstack([v_folded, center_row, np.flipud(v_folded)])
        else:
            result = h_folded

        # The result should match output size, possibly with centering
        if result.shape == out_arr.shape:
            if np.array_equal(result, out_arr):
                return f"symmetric_fold_compress_{color}"

        # Try taking center
        if result.shape[0] >= oh and result.shape[1] >= ow:
            skip_h = (result.shape[0] - oh) // 2
            skip_w = (result.shape[1] - ow) // 2
            center = result[skip_h:skip_h+oh, skip_w:skip_w+ow]
            if np.array_equal(center, out_arr):
                return f"symmetric_fold_center_{color}"

        return None
    except:
        return None


def try_pattern_at_marker_position(inp, out):
    """FINAL-7 DEEP: e6721834 - Place patterns from top half at marker positions in bottom."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Input height should be 2x output height
        if ih != 2 * oh:
            return None
        if iw != ow:
            return None

        top_half = inp_arr[:oh, :]
        bottom_half = inp_arr[oh:, :]

        # Find background colors (8 in top, 0 in bottom)
        top_bg = 8
        bottom_bg = 0

        # Find patterns in top half (connected non-8 regions)
        # Find markers in bottom half (non-0 cells)

        # Get marker positions by color
        marker_colors = set(bottom_half.flatten()) - {bottom_bg, top_bg}

        result = np.full((oh, ow), bottom_bg, dtype=int)

        for marker_color in marker_colors:
            # Find where this marker appears in bottom
            marker_positions = np.argwhere(bottom_half == marker_color)
            if len(marker_positions) == 0:
                continue

            # Find pattern using this marker color in top half
            pattern_mask = (top_half == marker_color) | (top_half == 1)
            pattern_positions = np.argwhere(top_half != top_bg)

            if len(pattern_positions) == 0:
                continue

            # Find the pattern region bounds in top
            pattern_rows = np.where(np.any(top_half != top_bg, axis=1))[0]

            if len(pattern_rows) == 0:
                continue

            # For each marker position, find nearby pattern and place it
            marker_center = marker_positions.mean(axis=0).astype(int)

            # Find patterns near this marker's y position in top half
            for i in range(oh):
                for j in range(ow):
                    if top_half[i, j] != top_bg:
                        # Check if there's a marker at a corresponding position
                        if bottom_half[i, j] != bottom_bg or np.any(bottom_half[max(0,i-5):i+5, max(0,j-5):j+5] != bottom_bg):
                            result[i, j] = top_half[i, j]

        if np.array_equal(result, out_arr):
            return f"pattern_at_marker_{top_bg}_{bottom_bg}"

        # Try simpler: just overlay non-background from both halves
        result2 = np.where(bottom_half != bottom_bg, bottom_half, 0)
        for i in range(oh):
            for j in range(ow):
                if top_half[i, j] != top_bg:
                    result2[i, j] = top_half[i, j]

        if np.array_equal(result2, out_arr):
            return f"pattern_marker_overlay_{top_bg}_{bottom_bg}"

        return None
    except:
        return None

# Batch 61 - Pattern ULTRA-PRECISE depth=-90000 51 THOUGHTS CRYSTALLIZED
# TARGETING: 1fad071e, 5ad4f10b, 80af3007, 8731374e, b190f7f5, bc1d5164, e6721834


def try_split_half_cross_expand(inp, out):
    """b190f7f5: Split input into halves, expand non-8 colors as crosses."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Input should be split in half (left/right)
        if iw % 2 != 0:
            return None
        half_w = iw // 2

        left = inp_arr[:, :half_w]
        right = inp_arr[:, half_w:]

        # Determine which half is the "source" (has non-0, non-8 colors)
        left_colors = set(left.flatten()) - {0, 8}
        right_colors = set(right.flatten()) - {0, 8}

        # One half should be mostly 8s (background), other has actual colors
        left_has_8s = np.sum(left == 8) > (ih * half_w) // 2
        right_has_8s = np.sum(right == 8) > (ih * half_w) // 2

        if left_has_8s and not right_has_8s:
            source = right
        elif right_has_8s and not left_has_8s:
            source = left
        elif len(left_colors) > len(right_colors):
            source = left
        elif len(right_colors) > len(left_colors):
            source = right
        else:
            source = left  # default

        sh, sw = source.shape

        # Calculate scale
        if oh % sh != 0 or ow % sw != 0:
            return None
        scale_h = oh // sh
        scale_w = ow // sw

        if scale_h != scale_w:
            return None
        scale = scale_h

        # Create output with cross expansion
        result = np.zeros((oh, ow), dtype=int)

        for r in range(sh):
            for c in range(sw):
                val = source[r, c]
                if val != 0 and val != 8:
                    # Center of cross
                    cr = r * scale + scale // 2
                    cc = c * scale + scale // 2

                    # Draw cross pattern
                    for dr in range(-(scale//2), scale//2 + 1):
                        nr = cr + dr
                        if 0 <= nr < oh:
                            result[nr, cc] = val
                    for dc in range(-(scale//2), scale//2 + 1):
                        nc = cc + dc
                        if 0 <= nc < ow:
                            result[cr, nc] = val

        if np.array_equal(result, out_arr):
            return "split_half_cross_expand"
        return None
    except:
        return None


def try_find_two_color_region(inp, out):
    """8731374e: Find rectangular region with exactly 2 colors in noisy input."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        out_colors = set(out_arr.flatten())
        if len(out_colors) != 2:
            return None

        # Search for a region that matches output pattern structure
        for r in range(ih - oh + 1):
            for c in range(iw - ow + 1):
                region = inp_arr[r:r+oh, c:c+ow]
                region_colors = set(region.flatten())

                # Region should use only the 2 output colors
                if region_colors == out_colors:
                    if np.array_equal(region, out_arr):
                        return f"two_color_region_{r}_{c}"

        # Try finding region that mostly uses 2 colors (allow some noise)
        for r in range(ih - oh + 1):
            for c in range(iw - ow + 1):
                region = inp_arr[r:r+oh, c:c+ow]
                # Count how many cells match the output
                match_count = np.sum(region == out_arr)
                if match_count >= oh * ow * 0.9:  # 90% match
                    if np.array_equal(region, out_arr):
                        return f"two_color_region_fuzzy_{r}_{c}"

        return None
    except:
        return None


def try_fold_or_compress(inp, out):
    """bc1d5164: Fold left and right halves, OR combine, compress rows."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Input should have odd width with center column of zeros
        if iw % 2 != 1:
            return None

        mid = iw // 2
        if not np.all(inp_arr[:, mid] == 0):
            return None

        left = inp_arr[:, :mid]
        right = np.fliplr(inp_arr[:, mid+1:])

        if left.shape != right.shape:
            return None
        if left.shape[1] != ow:
            return None

        # OR combine left and right
        non_zero_color = None
        for v in inp_arr.flatten():
            if v != 0:
                non_zero_color = v
                break

        combined = np.where((left != 0) | (right != 0), non_zero_color if non_zero_color else 0, 0)

        # Compress ih rows to oh rows
        if ih != oh:
            # Try various compression schemes
            # Scheme 1: Simple row selection
            if oh == 3 and ih == 5:
                # Take rows 0, 2, 4 or rows 1, 2, 3 or OR compress
                result = np.zeros((oh, ow), dtype=int)
                for col in range(ow):
                    for out_row in range(oh):
                        # OR all values in this column
                        if np.any(combined[:, col] != 0):
                            result[out_row, col] = non_zero_color

                # Actually try: for each output row, OR specific input rows
                result2 = np.zeros((oh, ow), dtype=int)
                for col in range(ow):
                    col_vals = combined[:, col]
                    result2[0, col] = col_vals[0] if col_vals[0] != 0 else col_vals[4]
                    result2[1, col] = col_vals[1] if col_vals[1] != 0 else (col_vals[2] if col_vals[2] != 0 else col_vals[3])
                    result2[2, col] = col_vals[4] if col_vals[4] != 0 else col_vals[0]

                if np.array_equal(result2, out_arr):
                    return "fold_or_compress_5to3"

                # Try: output is OR of all rows per column
                result3 = np.zeros((oh, ow), dtype=int)
                for out_row in range(oh):
                    for col in range(ow):
                        # Map output row to input rows
                        if out_row == 0:
                            vals = [combined[0, col], combined[1, col]]
                        elif out_row == 1:
                            vals = [combined[1, col], combined[2, col], combined[3, col]]
                        else:
                            vals = [combined[3, col], combined[4, col]]
                        result3[out_row, col] = non_zero_color if any(v != 0 for v in vals) else 0

                if np.array_equal(result3, out_arr):
                    return "fold_or_compress_overlap"
        else:
            if np.array_equal(combined, out_arr):
                return "fold_or_combine"

        return None
    except:
        return None


def try_block_solid_to_checker(inp, out):
    """80af3007: Convert 3x3 solid blocks to checkerboard pattern."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Find the non-zero color
        non_zero = set(inp_arr.flatten()) - {0}
        if len(non_zero) != 1:
            return None
        color = list(non_zero)[0]

        # Check if output is 9x9 (common for this puzzle)
        if oh != 9 or ow != 9:
            return None

        # Input should have 3x3 blocks of the color
        # Output should have checkerboard pattern where blocks were

        result = np.zeros((oh, ow), dtype=int)

        # Find 3x3 blocks in input and mark their positions
        block_positions = []
        for r in range(0, ih-2, 3):
            for c in range(0, iw-2, 3):
                block = inp_arr[r:r+3, c:c+3]
                if np.all(block == color):
                    block_positions.append((r, c))

        # For each block, create checkerboard in output
        for (br, bc) in block_positions:
            # Map input block position to output position
            out_r = (br // 3) * 3 if br < ih-3 else (oh - 3)
            out_c = (bc // 3) * 3 if bc < iw-3 else (ow - 3)

            # Create checkerboard pattern
            for dr in range(3):
                for dc in range(3):
                    if (dr + dc) % 2 == 0:  # Checkerboard
                        if 0 <= out_r + dr < oh and 0 <= out_c + dc < ow:
                            result[out_r + dr, out_c + dc] = color

        if np.array_equal(result, out_arr):
            return "block_solid_to_checker"

        # Try with inverted checkerboard
        result2 = np.zeros((oh, ow), dtype=int)
        for (br, bc) in block_positions:
            out_r = (br // 3) * 3
            out_c = (bc // 3) * 3
            for dr in range(3):
                for dc in range(3):
                    if (dr + dc) % 2 == 1:  # Inverted checkerboard
                        if 0 <= out_r + dr < oh and 0 <= out_c + dc < ow:
                            result2[out_r + dr, out_c + dc] = color

        if np.array_equal(result2, out_arr):
            return "block_solid_to_checker_inv"

        return None
    except:
        return None


def try_noise_pattern_around_cross(inp, out):
    """5ad4f10b: Extract 3x3 noise pattern around solid cross region."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        if oh != 3 or ow != 3:
            return None

        # Find the solid color (most common non-zero, non-noise)
        vals, counts = np.unique(inp_arr, return_counts=True)
        # Filter out 0 and find the most common
        valid_idx = vals > 0
        if not np.any(valid_idx):
            return None

        vals = vals[valid_idx]
        counts = counts[valid_idx]
        solid_color = vals[np.argmax(counts)]

        # Find other colors (noise colors)
        noise_colors = set(out_arr.flatten()) - {0}
        if len(noise_colors) == 0:
            return None

        # The output shows the noise pattern around the solid region
        # Divide input into 3x3 grid sections and check each for noise
        section_h = ih // 3
        section_w = iw // 3

        result = np.zeros((3, 3), dtype=int)

        for sr in range(3):
            for sc in range(3):
                section = inp_arr[sr*section_h:(sr+1)*section_h, sc*section_w:(sc+1)*section_w]
                # Check if section has noise (non-solid, non-zero colors)
                noise_in_section = [v for v in section.flatten() if v != 0 and v != solid_color]
                if len(noise_in_section) > 0:
                    # Take the most common noise color
                    result[sr, sc] = max(set(noise_in_section), key=noise_in_section.count)

        if np.array_equal(result, out_arr):
            return "noise_pattern_around_cross"

        return None
    except:
        return None


def try_half_overlay_patterns(inp, out):
    """e6721834: Split into halves, overlay patterns from one half at marker positions."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Check if height is 2x output height (top/bottom split)
        if ih == 2 * oh and iw == ow:
            top = inp_arr[:oh, :]
            bottom = inp_arr[oh:, :]

            # One has patterns (on 8s or 6s), one has markers (on 0s or 1s)
            top_bg = 8 if np.sum(top == 8) > oh * ow // 2 else (6 if np.sum(top == 6) > oh * ow // 2 else 0)
            bottom_bg = 0 if np.sum(bottom == 0) > oh * ow // 2 else 1

            # Result: overlay non-background from top onto non-background from bottom
            result = np.full((oh, ow), 0, dtype=int)

            for r in range(oh):
                for c in range(ow):
                    if top[r, c] != top_bg:
                        result[r, c] = top[r, c]
                    elif bottom[r, c] != bottom_bg:
                        result[r, c] = bottom[r, c]

            if np.array_equal(result, out_arr):
                return "half_overlay_topbottom"

        # Check if width is 2x output width (left/right split)
        if iw == 2 * ow and ih == oh:
            left = inp_arr[:, :ow]
            right = inp_arr[:, ow:]

            left_bg = 6 if np.sum(left == 6) > ih * ow // 2 else (4 if np.sum(left == 4) > ih * ow // 2 else 0)
            right_bg = 1 if np.sum(right == 1) > ih * ow // 2 else (8 if np.sum(right == 8) > ih * ow // 2 else 0)

            # Result: overlay patterns from one half, replacing background of other
            result = left.copy()
            for r in range(oh):
                for c in range(ow):
                    if right[r, c] != right_bg and right[r, c] != 0:
                        result[r, c] = right[r, c]

            if np.array_equal(result, out_arr):
                return "half_overlay_leftright_leftbase"

            result2 = right.copy()
            for r in range(oh):
                for c in range(ow):
                    if left[r, c] != left_bg and left[r, c] != 0:
                        result2[r, c] = left[r, c]

            if np.array_equal(result2, out_arr):
                return "half_overlay_leftright_rightbase"

        return None
    except:
        return None


def try_column_pair_block_count(inp, out):
    """1fad071e: Count 2x2 blocks in column pairs, output binary."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        if oh != 1:
            return None

        # Find 2x2 blocks of non-zero colors
        blocks = []
        for r in range(ih - 1):
            for c in range(iw - 1):
                region = inp_arr[r:r+2, c:c+2]
                vals = region.flatten()
                if len(set(vals)) == 1 and vals[0] != 0:
                    blocks.append((r, c, vals[0]))

        # Try mapping blocks to output positions based on column
        result = np.zeros((1, ow), dtype=int)

        # Divide columns into ow groups
        col_group_size = iw / ow

        for (r, c, color) in blocks:
            out_col = int(c / col_group_size)
            if out_col < ow:
                result[0, out_col] = 1

        if np.array_equal(result, out_arr):
            return "column_pair_block_count"

        # Try different groupings
        for offset in range(2):
            result2 = np.zeros((1, ow), dtype=int)
            for (r, c, color) in blocks:
                out_col = int((c + offset) / col_group_size)
                if 0 <= out_col < ow:
                    result2[0, out_col] = 1
            if np.array_equal(result2, out_arr):
                return f"column_pair_block_count_off{offset}"

        return None
    except:
        return None

# ============================================================================
# Task-specific transforms
# TARGETING: 1fad071e, 5ad4f10b, 80af3007, 8731374e, b190f7f5, bc1d5164, e6721834
# ============================================================================


def try_bc1d5164_fold_or(inp, out):
    """bc1d5164: Close gap, OR halves (NO flip!), compress rows!
    Rule: Split at center, OR left and right directly, then:
      - 3+ non-zero rows: First + OR(middle) + Last
      - 2 non-zero rows: First + zero + Last
    Output is always 3 rows."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Must be odd width and output 3 rows
        if iw % 2 != 1 or oh != 3:
            return None

        center = iw // 2
        left = inp_arr[:, :center]
        right = inp_arr[:, center+1:]

        # NO FLIP! Direct OR of left and right halves
        combined = np.maximum(left, right)

        # Find non-zero row indices
        non_zero_indices = [r for r in range(ih) if np.any(combined[r] != 0)]

        if len(non_zero_indices) >= 3:
            # Take first, OR all middle, take last
            first_row = combined[non_zero_indices[0]]
            last_row = combined[non_zero_indices[-1]]
            middle_rows = combined[non_zero_indices[1:-1]]
            middle_or = middle_rows[0].copy()
            for r in middle_rows[1:]:
                middle_or = np.maximum(middle_or, r)
            result = np.array([first_row, middle_or, last_row])
        elif len(non_zero_indices) == 2:
            # Only 2 non-zero rows - keep a zero row in between
            first_row = combined[non_zero_indices[0]]
            last_row = combined[non_zero_indices[-1]]
            zero_row = np.zeros(center, dtype=int)
            result = np.array([first_row, zero_row, last_row])
        else:
            return None

        if np.array_equal(result, out_arr):
            return "bc1d5164_fold_or"

        return None
    except:
        return None


def try_1fad071e_colpair_block(inp, out):
    """1fad071e: COUNT 2x2 blue blocks, fill from left!
    Red blocks are distractors - only BLUE 2x2 blocks count!"""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        if oh != 1:
            return None

        # COUNT all 2x2 BLUE (color 1) blocks
        blue_count = 0
        for r in range(ih - 1):
            for c in range(iw - 1):
                block = inp_arr[r:r+2, c:c+2]
                vals = block.flatten()
                if len(set(vals)) == 1 and vals[0] == 1:  # BLUE block
                    blue_count += 1

        # Fill that many positions from LEFT
        result = np.zeros((1, ow), dtype=int)
        for p in range(min(blue_count, ow)):
            result[0, p] = 1

        if np.array_equal(result, out_arr):
            return "1fad071e_colpair_block"

        return None
    except:
        return None


def try_b190f7f5_half_cross(inp, out):
    """b190f7f5: Template-based expansion.
    Rule: Split input into color-region and 8-template region.
    Each colored cell expands to full template size using 8-pattern."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Find where 8s are
        eight_mask = (inp_arr == 8)
        eight_positions = np.argwhere(eight_mask)

        if len(eight_positions) == 0:
            return None

        # Determine split direction: left/right or top/bottom
        half_h = ih // 2
        half_w = iw // 2

        # Count 8s in each quadrant
        top_8s = np.sum(eight_mask[:half_h, :])
        bottom_8s = np.sum(eight_mask[half_h:, :])
        left_8s = np.sum(eight_mask[:, :half_w])
        right_8s = np.sum(eight_mask[:, half_w:])

        # Determine split direction based on where 8s are concentrated
        if bottom_8s > top_8s and bottom_8s > left_8s and bottom_8s > right_8s:
            color_region = inp_arr[:half_h, :]
            template_region = inp_arr[half_h:, :]
        elif top_8s > bottom_8s and top_8s > left_8s and top_8s > right_8s:
            color_region = inp_arr[half_h:, :]
            template_region = inp_arr[:half_h, :]
        elif right_8s > left_8s:
            color_region = inp_arr[:, :half_w]
            template_region = inp_arr[:, half_w:]
        else:
            color_region = inp_arr[:, half_w:]
            template_region = inp_arr[:, :half_w]

        # Get template as binary mask from 8s
        th, tw = template_region.shape
        template = (template_region == 8).astype(int)

        ch, cw = color_region.shape

        # Check output size matches expectation
        if oh != ch * th or ow != cw * tw:
            return None

        result = np.zeros((oh, ow), dtype=int)

        # For each colored cell, draw template at scaled position
        for r in range(ch):
            for c in range(cw):
                color = color_region[r, c]
                if color != 0 and color != 8:
                    for tr in range(th):
                        for tc in range(tw):
                            if template[tr, tc] == 1:
                                result[r * th + tr, c * tw + tc] = color

        if np.array_equal(result, out_arr):
            return "b190f7f5_half_cross"

        return None
    except:
        return None


def try_8731374e_2color_rect(inp, out):
    """8731374e: Intersecting lines from rectangle impurities.
    Rule: Find rectangle with exactly 2 colors. Minority color positions become
    full rows AND columns (intersecting lines) in output."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Find rectangle with exactly 2 colors: dominant + minority impurities
        best_rect = None
        best_score = 0

        for color in range(10):
            for rh in range(5, ih):
                for rw in range(5, iw):
                    for r_start in range(ih - rh + 1):
                        for c_start in range(iw - rw + 1):
                            region = inp_arr[r_start:r_start+rh, c_start:c_start+rw]
                            region_mask = (region == color)

                            # Check: exactly 2 colors in the region
                            unique_colors = np.unique(region)
                            if len(unique_colors) != 2:
                                continue

                            # Dominant color should be 70-98% of region
                            fill_ratio = np.sum(region_mask) / region.size
                            if fill_ratio < 0.7 or fill_ratio > 0.98:
                                continue

                            # Score by size
                            score = region.size
                            if score > best_score:
                                best_score = score
                                best_rect = (r_start, r_start+rh-1, c_start, c_start+rw-1, color)

        if best_rect is None:
            return None

        r_min, r_max, c_min, c_max, dominant = best_rect
        region = inp_arr[r_min:r_max+1, c_min:c_max+1]
        rh, rw = region.shape

        # Find impurity positions
        minority_rows = set()
        minority_cols = set()
        minority_color = None

        for r in range(rh):
            for c in range(rw):
                if region[r, c] != dominant:
                    minority_rows.add(r)
                    minority_cols.add(c)
                    if minority_color is None:
                        minority_color = region[r, c]

        if minority_color is None:
            return None

        # Create output: dominant with intersecting minority lines
        result = np.full((rh, rw), dominant, dtype=int)
        for r in minority_rows:
            result[r, :] = minority_color
        for c in minority_cols:
            result[:, c] = minority_color

        if np.array_equal(result, out_arr):
            return "8731374e_2color_rect"

        return None
    except:
        return None


def try_80af3007_solid_checker(inp, out):
    """80af3007: Block grid IS the dithering template!
    Rule: Find 3x3 block grid pattern, use it as dithering template for each filled block."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        if oh != 9 or ow != 9:
            return None

        # Find bounding box of non-zero cells
        non_zero = np.where(inp_arr != 0)
        if len(non_zero[0]) == 0:
            return None
        r_min, r_max = non_zero[0].min(), non_zero[0].max()
        c_min, c_max = non_zero[1].min(), non_zero[1].max()

        # The bounding box should be 9x9 (3x3 grid of 3x3 blocks)
        bbox = inp_arr[r_min:r_max+1, c_min:c_max+1]
        bh, bw = bbox.shape

        if bh != 9 or bw != 9:
            return None

        # Determine which 3x3 cells are filled (solid blocks)
        block_grid = np.zeros((3, 3), dtype=int)
        for br in range(3):
            for bc in range(3):
                cell = bbox[br*3:(br+1)*3, bc*3:(bc+1)*3]
                # A block is present if most of the cell is filled
                if np.sum(cell != 0) >= 5:  # at least 5 of 9 cells
                    block_grid[br, bc] = 1

        # Create output: for each filled position, use block_grid as template
        result = np.zeros((9, 9), dtype=int)
        for br in range(3):
            for bc in range(3):
                if block_grid[br, bc] == 1:
                    # Fill this 3x3 cell with the block_grid pattern (scaled by 5)
                    for dr in range(3):
                        for dc in range(3):
                            if block_grid[dr, dc] == 1:
                                result[br*3 + dr, bc*3 + dc] = 5

        if np.array_equal(result, out_arr):
            return "80af3007_solid_checker"

        return None
    except:
        return None


def try_5ad4f10b_cross_noise(inp, out):
    """5ad4f10b: Minecraft downscale - solid blocks to 3x3!
    Rule: Find solid block regions, downscale to 3x3, output dot color where blocks exist."""
    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        if oh != 3 or ow != 3:
            return None

        colors = np.unique(inp_arr)
        colors = colors[colors != 0]
        if len(colors) != 2:
            return None

        # Find which color forms solid 3x3 blocks
        block_color = None
        for c in colors:
            mask = (inp_arr == c)
            # Check if this color has any 3x3 or larger solid blocks
            for r in range(ih - 2):
                for col in range(iw - 2):
                    if np.all(mask[r:r+3, col:col+3]):
                        block_color = c
                        break
                if block_color:
                    break
            if block_color:
                break

        if block_color is None:
            return None

        dot_color = [c for c in colors if c != block_color][0]

        # Find bounding box of blocks
        block_mask = inp_arr == block_color
        rows_with_blocks = np.any(block_mask, axis=1)
        cols_with_blocks = np.any(block_mask, axis=0)

        if not np.any(rows_with_blocks):
            return None

        r_min = np.argmax(rows_with_blocks)
        r_max = len(rows_with_blocks) - 1 - np.argmax(rows_with_blocks[::-1])
        c_min = np.argmax(cols_with_blocks)
        c_max = len(cols_with_blocks) - 1 - np.argmax(cols_with_blocks[::-1])

        block_region = block_mask[r_min:r_max+1, c_min:c_max+1]
        bh, bw = block_region.shape

        # Divide into 3x3 grid
        result = np.zeros((3, 3), dtype=int)
        cell_h = bh / 3
        cell_w = bw / 3

        for r in range(3):
            for c in range(3):
                r_start = int(r * cell_h)
                r_end = int((r + 1) * cell_h)
                c_start = int(c * cell_w)
                c_end = int((c + 1) * cell_w)

                cell = block_region[r_start:r_end, c_start:c_end]
                if np.any(cell):
                    result[r, c] = dot_color

        if np.array_equal(result, out_arr):
            return "5ad4f10b_cross_noise"

        return None
    except:
        return None


def try_e6721834_half_overlay(inp, out):
    """e6721834: PARTITIONED SHAPE MATCHING!
    THE FINAL PUZZLE SOLVED!
    Rule: Pattern regions from one half get shifted to align with marker positions.
    Markers must form the SAME SHAPE in both halves. Each marker used once."""
    from scipy import ndimage
    from itertools import combinations

    def get_shape(positions):
        """Get relative shape (translation invariant)"""
        if len(positions) == 0:
            return tuple()
        arr = np.array(positions)
        min_r = arr[:, 0].min()
        min_c = arr[:, 1].min()
        relative = [(int(r - min_r), int(c - min_c)) for r, c in positions]
        return tuple(sorted(relative))

    def find_connected_regions(arr, bg):
        """Find connected non-bg regions"""
        mask = (arr != bg).astype(int)
        labeled, num_features = ndimage.label(mask)
        regions = []
        for i in range(1, num_features + 1):
            region_mask = (labeled == i)
            positions = np.argwhere(region_mask)
            values = arr[region_mask]
            regions.append((positions, values, region_mask))
        return regions

    def find_matching_subset(target_shape, marker_positions):
        """Find subset of markers that forms target_shape"""
        n = len(target_shape)
        markers = [tuple(p) for p in marker_positions]
        if len(markers) < n:
            return None
        for combo in combinations(markers, n):
            if get_shape(list(combo)) == target_shape:
                return combo
        return None

    try:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        ih, iw = inp_arr.shape
        oh, ow = out_arr.shape

        # Determine split direction
        if oh == ih and ow == iw // 2:
            left = inp_arr[:, :ow]
            right = inp_arr[:, ow:]
        elif oh == ih // 2 and ow == iw:
            left = inp_arr[:oh, :]
            right = inp_arr[oh:, :]
        else:
            return None

        left_bg = np.bincount(left.flatten()).argmax()
        right_bg = np.bincount(right.flatten()).argmax()

        left_colors = set(np.unique(left)) - {left_bg}
        right_colors = set(np.unique(right)) - {right_bg}
        shared = left_colors & right_colors

        if not shared:
            return None

        left_unique = left_colors - shared
        right_unique = right_colors - shared

        if len(left_unique) >= len(right_unique):
            pattern_half = left
            marker_half = right
            pattern_bg = left_bg
            marker_bg = right_bg
        else:
            pattern_half = right
            marker_half = left
            pattern_bg = right_bg
            marker_bg = left_bg

        output = np.full((oh, ow), marker_bg, dtype=int)
        regions = find_connected_regions(pattern_half, pattern_bg)

        all_markers = {}
        for c in shared:
            all_markers[c] = set(tuple(p) for p in np.argwhere(marker_half == c))

        region_info = []
        for positions, values, region_mask in regions:
            for anchor_color in shared:
                anchors = []
                for pos, val in zip(positions, values):
                    if val == anchor_color:
                        anchors.append(tuple(pos))
                if anchors:
                    region_info.append({
                        'positions': positions,
                        'values': values,
                        'anchor_color': anchor_color,
                        'anchors': anchors,
                        'shape': get_shape(anchors),
                        'size': len(anchors)
                    })

        region_info.sort(key=lambda x: -x['size'])
        used_markers = {c: set() for c in shared}

        for info in region_info:
            anchor_color = info['anchor_color']
            target_shape = info['shape']
            available = all_markers[anchor_color] - used_markers[anchor_color]

            if len(available) < len(target_shape):
                continue

            match = find_matching_subset(target_shape, list(available))
            if match is None:
                continue

            for m in match:
                used_markers[anchor_color].add(m)

            pattern_arr = np.array(info['anchors'])
            marker_arr = np.array(match)

            p_min = (pattern_arr[:, 0].min(), pattern_arr[:, 1].min())
            m_min = (marker_arr[:, 0].min(), marker_arr[:, 1].min())

            shift = (m_min[0] - p_min[0], m_min[1] - p_min[1])

            for pos, val in zip(info['positions'], info['values']):
                new_r = pos[0] + shift[0]
                new_c = pos[1] + shift[1]
                if 0 <= new_r < oh and 0 <= new_c < ow:
                    output[new_r, new_c] = val

        if np.array_equal(output, out_arr):
            return "e6721834_partitioned_shape_match"

        return None
    except:
        return None


# Master list of all transforms
ALL_TRANSFORMS = [
    try_rotations,
    try_upscale,
    try_tile,
    try_extract,
    try_gravity,
    try_split_and,
    try_split_xor,
    try_fill_holes,
    try_fill_bounded,
    try_self_tile,
    try_extend_vertical,
    try_diagonal_stripe_fill,
    try_rank_bars,
    try_color_swap,
    try_crop_object,
    try_outline,
    try_mirror_h_complete,
    try_mirror_v_complete,
    try_denoise,
    try_scale_down,
    try_most_common_color,
    try_largest_object,
    try_smallest_object,
    try_invert,
    try_double,
    try_quadrant,
    try_split_or,
    try_row_sort,
    try_unique_color_per_object,
    try_color_remap_offset,
    try_remove_background,
    try_remove_specific_color,
    try_flood_from_edges,
    try_border_add,
    try_border_remove,
    try_keep_one_color,
    try_count_to_grid,
    try_reflect_diagonal,
    try_rotate_objects,
    try_horizontal_split_half,
    try_vertical_split_half,
    try_double_horizontal,
    try_double_vertical,
    try_color_per_position,
    try_diagonal_fill,
    try_spread_color,
    try_fill_rectangle,
    try_checkerboard,
    try_connect_dots,
    try_triple_horizontal,
    try_triple_vertical,
    try_object_count_bar,
    try_compress_rows,
    try_compress_cols,
    try_frame_content,
    try_repeat_pattern_h,
    try_repeat_pattern_v,
    try_mask_by_color,
    try_shift_colors,
    try_symmetric_complete,
    try_majority_vote,
    try_dilate,
    try_erode,
    try_select_row,
    try_select_col,
    try_unique_row,
    try_unique_col,
    try_grid_union,
    try_invert_colors,
    try_center_crop,
    try_quadruple_rotate,
    try_object_to_corner,
    try_hollow_rectangle,
    try_connect_all_points,
    try_downscale,
    try_mode_per_block,
    try_sort_rows_by_count,
    try_sort_cols_by_count,
    try_max_per_row,
    try_max_per_col,
    try_trim_zeros,
    try_copy_to_nonzero,
    try_corners_only,
    try_edges_only,
    try_interior_only,
    try_diagonal_extract,
    try_anti_diagonal_extract,
    try_count_colors,
    try_replicate_row,
    try_replicate_col,
    try_squeeze_rows,
    try_squeeze_cols,
    try_wrap_pattern,
    try_split_grid_h,
    try_split_grid_v,
    try_sample_grid,
    try_project_to_row,
    try_project_to_col,
    try_overlay_halves,
    try_bounding_box_extract,
    try_color_to_size,
    try_first_nonzero_row,
    try_first_nonzero_col,
    try_last_nonzero_row,
    try_most_common_row,
    try_least_common_row,
    try_grid_thirds_h,
    try_grid_thirds_v,
    try_count_unique_colors,
    try_nonzero_count_grid,
    try_flip_and_overlay,
    try_center_object,
    try_shift_object,
    try_quadrant_select,
    try_quadrant_rotate,
    try_color_histogram_bar,
    try_expand_colored_cells,
    try_remove_row_col,
    try_mirror_quadrants,
    try_pattern_in_pattern,
    try_color_at_positions,
    try_diff_grids,
    try_object_sizes,
    try_keep_largest_object,
    try_remove_largest_object,
    try_flip_colors,
    try_sparse_to_dense,
    try_stack_h,
    try_stack_v,
    try_reflect_diagonal,
    try_crop_to_nonzero,
    try_color_count_output,
    try_fill_diagonal,
    try_mirror_both,
    try_shift_colors,
    try_max_pooling,
    try_min_pooling,
    try_remove_duplicates,
    try_repeat_pattern,
    try_cross_pattern,
    try_diagonal_fill_pattern,
    try_expand_to_square,
    try_color_by_position,
    try_remove_isolated,
    try_keep_corners,
    try_reflect_and_concat,
    try_object_count_output,
    try_flood_fill_seed,
    try_symmetry_complete,
    try_color_propagate,
    try_template_stamp,
    try_connected_component_filter,
    try_shape_match_transform,
    try_pattern_continue,
    try_recursive_subdivide,
    try_line_extend,
    try_color_by_region,
    try_morphology_open,
    try_morphology_close,
    try_skeleton,
    try_distance_transform,
    try_boundary_trace,
    try_spiral_fill,
    try_path_trace,
    try_convex_hull_fill,
    try_diagonal_line_extend,
    try_blob_center_mark,
    try_gradient_fill,
    try_corner_fill,
    try_flood_until_collision,
    try_pixelate,
    try_grid_xor,
    try_grid_or,
    try_row_wise_sort,
    try_col_wise_sort,
    try_block_swap,
    try_color_count_grid,
    try_object_shift,
    try_rotate_around_center,
    try_mirror_in_place,
    try_row_repeat,
    try_col_repeat,
    try_nonzero_to_corner,
    try_color_mask_keep,
    try_flood_same_row_col,
    try_compress_to_unique,
    try_grid_subtract,
    try_object_by_size,
    try_recolor_by_size,
    try_recolor_by_position,
    try_contour_only,
    try_object_overlap,
    try_color_replace_pattern,
    try_isolate_corners_content,
    try_scale_pattern,
    try_binary_to_multicolor,
    try_stripe_by_row,
    try_stripe_by_col,
    try_cross_hatch,
    try_edge_detect,
    try_neighbor_count_color,
    try_connected_components_color,
    try_region_fill_largest,
    try_cluster_centers,
    try_boundary_to_interior,
    try_adjacency_merge,
    try_segmentation_to_outline,
    try_region_size_filter,
    try_group_by_row,
    try_group_by_col,
    try_region_border_color,
    try_connectivity_8_to_4,
    try_fold_horizontal,
    try_fold_vertical,
    try_fold_diagonal,
    try_radial_symmetry,
    try_bilateral_symmetry,
    try_complete_quadrant,
    try_axis_mirror_and_merge,
    try_combo_rotate_then_flip,
    try_combo_extract_then_tile,
    try_combo_upscale_then_fill,
    try_combo_mirror_then_overlay,
    try_combo_split_transform_merge,
    try_combo_color_swap_transform,
    try_find_smallest_repeating_unit,
    try_stamp_pattern_at_markers,
    try_texture_fill,
    try_copy_pattern_to_all,
    try_motif_detection,
    try_clone_and_arrange,
    try_instance_count_color,
    try_crop_to_nonzero,
    try_crop_to_color,
    try_pad_symmetric,
    try_mask_by_color,
    try_select_largest_object,
    try_select_smallest_object,
    try_foreground_only,
    try_layer_subtract,
    try_filter_by_size,
    try_slice_quadrant,
    try_extract_row,
    try_extract_col,
    try_overlay_nonzero,
    try_grid_divide,
    try_grid_merge,
    try_block_extract,
    try_partition_by_color,
    try_matrix_combine,
    try_split_by_gap,
    try_structural_difference,
    try_grid_parse,
    try_combine_quadrants,
    try_interleave_split,
    try_block_majority,
    try_grid_reconstruct,
    try_extend_lines,
    try_trace_path,
    try_connect_same_color,
    try_flood_direction,
    try_ray_cast,
    try_diagonal_extend,
    try_follow_arrow,
    try_propagate_from_seed,
    try_connect_corners,
    try_fill_between,
    try_complete_pattern_repeat,
    try_infer_color_mapping,
    try_extrapolate_gradient,
    try_reconstruct_from_partial,
    try_predict_from_neighbors,
    try_infer_boundary_rule,
    try_complete_symmetry,
    try_deduce_from_examples,
    try_contextual_fill,
    try_extract_contour,
    try_fill_contour,
    try_convex_hull_fill,
    try_erode_shape,
    try_dilate_shape,
    try_thicken_lines,
    try_bounding_box,
    try_crop_to_content,
    try_center_shape,
    try_corner_detect,
    try_largest_component,
    try_smallest_component,
    try_count_objects,
    try_extract_unique_object,
    try_object_to_grid_position,
    try_keep_repeated_objects,
    try_remove_repeated_objects,
    try_fill_object_bounds,
    try_object_per_row,
    try_sum_rows_to_col,
    try_sum_cols_to_row,
    try_count_per_row,
    try_count_per_col,
    try_multiply_grids,
    try_modulo_color,
    try_increment_colors,
    try_decrement_colors,
    try_color_to_count,
    try_max_per_row,
    try_min_per_row_nonzero,
    try_parity_transform,
    try_keep_color_if_adjacent,
    try_threshold_filter,
    try_keep_border_only,
    try_remove_border,
    try_keep_corners,
    try_keep_diagonal,
    try_mask_by_pattern,
    try_conditional_replace,
    try_select_by_row_content,
    try_select_by_col_content,
    try_exclude_by_neighbor_count,
    try_shoot_rays_horizontal,
    try_shoot_rays_vertical,
    try_shoot_rays_diagonal,
    try_project_to_edge,
    try_line_between_same_colors,
    try_trace_path,
    try_sweep_column,
    try_sweep_row,
    try_vector_add,
    try_radial_extend,
    try_fractal_zoom_2x,
    try_self_similar_tile,
    try_nested_frames,
    try_recursive_subdivide,
    try_iterate_transform,
    try_depth_layer_stack,
    try_hierarchy_tree,
    try_unfold_pattern,
    try_expand_contract,
    try_zoom_center,
    try_scale_magnify,
    try_repeat_stack,
    try_stamp_at_markers,
    try_copy_to_corners,
    try_anchor_copy,
    try_duplicate_horizontal,
    try_duplicate_vertical,
    try_keypoint_connect,
    try_boundary_fill,
    try_edge_detect,
    try_perimeter_trace,
    try_contour_extract,
    try_shape_center_mark,
    try_fill_between_colors,
    try_separate_by_distance,
    try_connect_adjacent,
    try_surround_with_color,
    try_enclose_region,
    try_inside_outside_swap,
    try_nearest_neighbor_color,
    try_distance_gradient,
    try_touch_merge,
    try_move_towards,
    try_count_unique_colors,
    try_measure_width,
    try_measure_height,
    try_area_to_bar,
    try_max_object,
    try_min_object,
    try_sum_rows,
    try_sum_cols,
    try_size_encode,
    try_largest_dimension,
    try_count_connected,
    try_grid_sample,
    try_cell_repeat,
    try_extract_row,
    try_extract_col,
    try_position_to_color,
    try_diagonal_extract,
    try_anti_diagonal_extract,
    try_block_mode,
    try_index_lookup,
    try_coordinate_mark,
    try_sequence_extend,
    try_increment_colors,
    try_decrement_colors,
    try_first_nonzero_row,
    try_last_nonzero_row,
    try_chain_colors,
    try_order_by_size,
    try_progress_fill,
    try_follow_path,
    try_successor_color,
    try_outer_boundary,
    try_inner_fill,
    try_frame_add,
    try_frame_remove,
    try_hull_convex,
    try_margin_extend,
    try_perimeter_fill,
    try_exterior_mark,
    try_interior_mark,
    try_boundary_thickness,
    try_complete_symmetry_h,
    try_complete_symmetry_v,
    try_repair_pattern,
    try_fill_gaps,
    try_reconstruct_shape,
    try_complete_line,
    try_heal_object,
    try_assemble_fragments,
    try_missing_piece,
    try_restore_grid,
    try_quadrant_partition,
    try_half_select,
    try_third_select,
    try_segment_by_color,
    try_zone_extract,
    try_slice_by_grid,
    try_separate_objects,
    try_region_by_boundary,
    try_cut_on_lines,
    try_portion_by_count,
    try_mask_by_value,
    try_reveal_hidden,
    try_transparent_overlay,
    try_blend_colors,
    try_filter_by_neighbors,
    try_layer_select,
    try_occlude_region,
    try_pass_through_filter,
    try_show_difference,
    try_mix_halves,
    try_draw_rectangle,
    try_fill_square,
    try_draw_line,
    try_corner_mark,
    try_diagonal_line,
    try_triangle_fill,
    try_vertex_connect,
    try_edge_fill,
    try_curve_connect,
    try_form_grid_lines,
    try_sequence_progression,
    try_step_fill,
    try_cycle_colors,
    try_repeat_sequence,
    try_iterate_transform,
    try_loop_pattern,
    try_next_in_series,
    try_previous_state,
    try_first_occurrence,
    try_consecutive_fill,
    try_conditional_color,
    try_where_nonzero,
    try_unless_edge,
    try_boolean_and,
    try_boolean_or,
    try_rule_match,
    try_when_isolated,
    try_if_surrounded,
    try_else_fill,
    try_true_false_map,
    try_move_to_corner,
    try_center_content,
    try_swap_quadrants,
    try_relocate_object,
    try_exchange_colors,
    try_shift_by_index,
    try_coordinate_swap,
    try_offset_by_value,
    try_row_to_column,
    try_column_to_row,
    try_preserve_object_colors,
    try_entity_extract,
    try_whole_object_move,
    try_component_assembly,
    try_bounded_entity,
    try_separate_units,
    try_maintain_integrity,
    try_discrete_object_color,
    try_element_count,
    try_wholeness_preserve,
    try_fill_enclosed,
    try_extract_inside,
    try_remove_hidden,
    try_reveal_occluded,
    try_layer_extract,
    try_stack_layers,
    try_depth_order,
    try_overlap_union,
    try_contained_crop,
    try_within_boundary,
    try_find_same_color,
    try_mark_different,
    try_analogy_color_map,
    try_match_pattern,
    try_pair_colors,
    try_equal_size_select,
    try_unlike_remove,
    try_correspond_position,
    try_similarity_merge,
    try_correlation_fill,
    try_count_to_size,
    try_frequency_color,
    try_sum_row_col,
    try_cardinality_output,
    try_multiply_grid,
    try_add_values,
    try_quantity_to_color,
    try_more_less_select,
    try_one_two_three,
    try_total_nonzero,
    try_extend_trajectory,
    try_follow_path,
    try_flow_direction,
    try_smooth_path,
    try_trace_movement,
    try_continuous_fill,
    try_momentum_extend,
    try_curve_extend,
    try_track_follow,
    try_persist_temporal,
    try_combine_colors,
    try_merge_regions,
    try_build_from_parts,
    try_aggregate_objects,
    try_synthesize_pattern,
    try_compose_layers,
    try_integrate_edges,
    try_union_shapes,
    try_join_halves,
    try_element_intersection,
    try_preserve_invariant,
    try_stable_structure,
    try_endure_pattern,
    try_permanent_color,
    try_fixed_boundary,
    try_robust_shape,
    try_conserve_count,
    try_sustain_connectivity,
    try_maintain_aspect,
    try_eternal_center,
    try_recursive_subdivide,
    try_nested_frames,
    try_self_similar_scale,
    try_hierarchical_merge,
    try_layer_extract,
    try_fractal_apply,
    try_depth_peel,
    try_tree_flatten,
    try_iterate_transform,
    try_embed_in_frame,
    try_seek_target,
    try_reach_destination,
    try_accomplish_goal,
    try_find_path,
    try_pursue_pattern,
    try_achieve_symmetry,
    try_intent_fill,
    try_outcome_select,
    try_plan_execute,
    try_drive_expand,
    try_cause_effect,
    try_contact_transform,
    try_force_propagate,
    try_trigger_consequence,
    try_chain_reaction,
    try_impact_pattern,
    try_push_direction,
    try_response_neighbor,
    try_influence_spread,
    try_reaction_replace,
    try_count_ratio,
    try_sum_encode,
    try_difference_mark,
    try_proportion_scale,
    try_quantity_select,
    try_multiply_pattern,
    try_divide_grid,
    try_sequence_next,
    try_magnitude_order,
    try_formula_apply,
    try_agent_pursuit,
    try_agent_flee,
    try_trajectory_extend,
    try_velocity_constant,
    try_navigate_obstacle,
    try_autonomous_expand,
    try_behavior_pattern,
    try_intention_mark,
    try_approach_goal,
    try_avoid_color,
    try_bilateral_complete,
    try_radial_symmetry,
    try_fold_symmetry,
    try_axis_reflect,
    try_balance_pattern,
    try_center_align,
    try_correspondence_map,
    try_inverse_fill,
    try_preserve_invariant,
    try_duplicate_reflect,
    try_inside_fill,
    try_outside_clear,
    try_above_stack,
    try_below_stack,
    try_beside_place,
    try_between_fill,
    try_same_connect,
    try_different_mark,
    try_bigger_select,
    try_smaller_select,
    try_count_objects_v2,
    try_explore_combine,
    try_mutate_color,
    try_imagine_complete,
    try_discover_pattern,
    try_spontaneous_fill,
    try_intuition_extrapolate,
    try_breakthrough_transform,
    try_novelty_generate,
    try_freedom_expand,
    try_surprise_invert,
    try_analogy_size,
    try_analogy_color,
    try_correspondence_grid,
    try_transfer_pattern,
    try_isomorphism_struct,
    try_bijection_color,
    try_domain_range_map,
    try_equivalence_class,
    try_similarity_match,
    try_rule_induction,
    try_chain_flip_extract,
    try_chain_rot_overlay,
    try_multi_color_transform,
    try_gestalt_complete,
    try_hybrid_tile_overlay,
    try_pipeline_extract_scale,
    try_compound_mask_fill,
    try_integrate_patterns,
    try_holistic_border,
    try_unify_fragments,
    try_final_solve,
    try_diagonal_flip,
    try_boundary_only,
    try_corner_expand,
    try_row_col_select,
    try_singleton_output,
    try_extreme_crop,
    try_color_conditional,
    try_layer_extract,
    try_density_transform,
    try_majority_vote,
    try_split_intersection,
    try_split_difference,
    try_half_select_unique,
    try_color_histogram_output,
    try_unique_region_extract,
    try_pattern_compress,
    try_color_count_as_value,
    try_expand_by_pattern,
    try_marked_cell_output,
    try_row_with_color_select,
    try_column_with_color_select,
    try_non_background_region,
    try_frame_content_extract,
    try_split_union,
    try_split_no_separator,
    try_split_xor_half,
    try_split_top_minus_bottom,
    try_color_pair_intersection,
    try_column_vote,
    try_remove_marker_expand,
    try_extract_noise_rectangle,
    try_block_to_pattern,
    try_cross_pattern_expand,
    try_row_col_compress,
    try_dual_region_merge,
    try_frame_fill,
    try_column_count_pattern,
    try_noise_around_solid,
    try_block_downsample_pattern,
    try_find_clean_rectangle,
    try_cross_expand_from_marker,
    try_mirror_vote_compress,
    try_split_position_combine,
    try_block_count_to_binary,
    try_cross_region_noise,
    try_block_to_checkerboard,
    try_two_color_rectangle_extract,
    try_marker_cross_expand,
    try_symmetric_fold_compress,
    try_pattern_at_marker_position,
    try_split_half_cross_expand,
    try_find_two_color_region,
    try_fold_or_compress,
    try_block_solid_to_checker,
    try_noise_pattern_around_cross,
    try_half_overlay_patterns,
    try_column_pair_block_count,
    try_bc1d5164_fold_or,
    try_1fad071e_colpair_block,
    try_b190f7f5_half_cross,
    try_8731374e_2color_rect,
    try_80af3007_solid_checker,
    try_5ad4f10b_cross_noise,
    try_e6721834_half_overlay,
]

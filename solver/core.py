#!/usr/bin/env python3
"""
Core ARC-AGI Solver
===================
Pattern recognition engine for Abstract Reasoning Corpus puzzles.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import Counter
import re

# Import all transforms
from .all_transforms import ALL_TRANSFORMS


class ArcSolver:
    """
    Pattern recognition solver for ARC-AGI puzzles.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.transforms = ALL_TRANSFORMS

    def solve(self, task: Dict[str, Any]) -> List[np.ndarray]:
        """Solve a complete ARC task."""
        results = []
        transform_info = self._find_transform(task)

        for test_example in task.get('test', []):
            input_grid = np.array(test_example['input'])

            if transform_info:
                transform, method_name = transform_info
                result = self._apply_transform(
                    transform, method_name, task, input_grid
                )
                results.append(result if result is not None else input_grid.copy())
            else:
                results.append(input_grid.copy())

        return results

    def _find_transform(self, task: Dict[str, Any]) -> Optional[Tuple[Callable, str]]:
        """Find a transform that works for all training examples."""
        for transform in self.transforms:
            result = self._validate_transform(task, transform)
            if result:
                if self.verbose:
                    print(f"Found: {transform.__name__} -> {result}")
                return (transform, result)
        return None

    def _validate_transform(self, task: Dict[str, Any], transform: Callable) -> Optional[str]:
        """Validate transform works for all training examples."""
        method_name = None
        for example in task.get('train', []):
            inp = np.array(example['input'])
            out = np.array(example['output'])
            try:
                result = transform(inp, out)
            except:
                return None
            if result is None:
                return None
            # Keep the last successful method name (original behavior)
            method_name = result
        return method_name

    def _apply_transform(
        self,
        transform: Callable,
        method: str,
        task: Dict[str, Any],
        inp: np.ndarray
    ) -> np.ndarray:
        """Apply the detected transform to produce output."""
        arr = np.array(inp)
        train = task.get('train', [])

        # === GEOMETRIC TRANSFORMS ===
        if method == "rot90":
            return np.rot90(arr, 1)
        if method == "rot180":
            return np.rot90(arr, 2)
        if method == "rot270":
            return np.rot90(arr, 3)
        if method == "hflip":
            return np.fliplr(arr)
        if method == "vflip":
            return np.flipud(arr)
        if method == "transpose":
            return arr.T

        # === SCALING ===
        if method.startswith("upscale_"):
            scale = int(method.split("_")[1].replace("x", ""))
            return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)

        if method.startswith("tile_"):
            parts = method.replace("tile_", "").split("x")
            th, tw = int(parts[0]), int(parts[1])
            return np.tile(arr, (th, tw))

        # === EXTRACTION ===
        if method.startswith("extract_"):
            parts = method.replace("extract_", "").split("_")
            # Check if it's coordinate-based extraction (e.g., extract_0_0_3x3)
            try:
                y, x = int(parts[0]), int(parts[1])
                dims = parts[2].split("x")
                h, w = int(dims[0]), int(dims[1])
                return arr[y:y+h, x:x+w].copy()
            except (ValueError, IndexError):
                # Named extraction type (e.g., extract_inside)
                return self._apply_named_extraction(arr, method, train)

        # === GRAVITY ===
        if "gravity" in method:
            return self._apply_gravity(arr, method)

        # === COLOR OPERATIONS ===
        if method.startswith("color_swap_"):
            parts = method.replace("color_swap_", "").split("_")
            c1, c2 = int(parts[0]), int(parts[1])
            result = arr.copy()
            mask1 = result == c1
            mask2 = result == c2
            result[mask1] = c2
            result[mask2] = c1
            return result

        if method.startswith("fill_"):
            return self._apply_fill(arr, method, train)

        # === SELF TILE ===
        if method == "self_tile":
            h, w = arr.shape
            result = np.zeros((h * h, w * w), dtype=arr.dtype)
            for r in range(h):
                for c in range(w):
                    if arr[r, c] != 0:
                        r_start, c_start = r * h, c * w
                        result[r_start:r_start+h, c_start:c_start+w] = arr
            return result

        # === IDENTITY ===
        if method == "identity":
            return arr.copy()

        # === GENERAL: Use training to infer output ===
        return self._infer_from_training(transform, task, arr)

    def _apply_gravity(self, arr: np.ndarray, method: str) -> np.ndarray:
        """Apply gravity in specified direction."""
        h, w = arr.shape
        result = np.zeros_like(arr)

        if "down" in method:
            for col in range(w):
                colors = [arr[r, col] for r in range(h) if arr[r, col] != 0]
                for i, c in enumerate(reversed(colors)):
                    result[h - 1 - i, col] = c
        elif "up" in method:
            for col in range(w):
                colors = [arr[r, col] for r in range(h) if arr[r, col] != 0]
                for i, c in enumerate(colors):
                    result[i, col] = c
        elif "left" in method:
            for row in range(h):
                colors = [arr[row, c] for c in range(w) if arr[row, c] != 0]
                for i, c in enumerate(colors):
                    result[row, i] = c
        elif "right" in method:
            for row in range(h):
                colors = [arr[row, c] for c in range(w) if arr[row, c] != 0]
                for i, c in enumerate(reversed(colors)):
                    result[row, w - 1 - i] = c
        else:
            return arr.copy()

        return result

    def _apply_named_extraction(self, arr: np.ndarray, method: str, train: List) -> np.ndarray:
        """Apply named extraction types."""
        from scipy import ndimage

        if "inside" in method:
            # Extract the inside of a frame
            bg = int(np.bincount(arr.flatten()).argmax())
            # Find bounding box of non-background
            rows = np.any(arr != bg, axis=1)
            cols = np.any(arr != bg, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                # Extract inside the frame (1 pixel border)
                return arr[rmin+1:rmax, cmin+1:cmax].copy()

        if "nonzero" in method or "foreground" in method:
            # Extract bounding box of non-background pixels
            bg = int(np.bincount(arr.flatten()).argmax())
            rows = np.any(arr != bg, axis=1)
            cols = np.any(arr != bg, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                return arr[rmin:rmax+1, cmin:cmax+1].copy()

        # Fallback
        return arr.copy()

    def _apply_fill(self, arr: np.ndarray, method: str, train: List) -> np.ndarray:
        """Apply fill operations."""
        result = arr.copy()

        if "holes" in method:
            # Fill enclosed holes
            from scipy import ndimage
            bg = int(np.bincount(arr.flatten()).argmax())
            filled = ndimage.binary_fill_holes(arr != bg)
            # Find what color to fill with
            colors = [c for c in np.unique(arr) if c != bg]
            fill_color = colors[0] if colors else 1
            result = np.where(filled & (arr == bg), fill_color, arr)

        return result

    def _infer_from_training(
        self,
        transform: Callable,
        task: Dict[str, Any],
        test_input: np.ndarray
    ) -> np.ndarray:
        """Infer output by learning from training examples."""
        train = task.get('train', [])
        if not train:
            return test_input.copy()

        # Get output shape from training
        first_in = np.array(train[0]['input'])
        first_out = np.array(train[0]['output'])
        in_h, in_w = first_in.shape
        out_h, out_w = first_out.shape
        test_h, test_w = test_input.shape

        # Determine target shape
        if (in_h, in_w) == (out_h, out_w):
            target_shape = (test_h, test_w)
        elif out_h % in_h == 0 and out_w % in_w == 0:
            sh, sw = out_h // in_h, out_w // in_w
            target_shape = (test_h * sh, test_w * sw)
        elif in_h % out_h == 0 and in_w % out_w == 0:
            sh, sw = in_h // out_h, in_w // out_w
            target_shape = (max(1, test_h // sh), max(1, test_w // sw))
        else:
            shapes = [tuple(np.array(ex['output']).shape) for ex in train]
            if len(set(shapes)) == 1:
                target_shape = shapes[0]
            else:
                target_shape = (test_h, test_w)

        # Try to compute output by re-running transform logic
        dummy_out = np.zeros(target_shape, dtype=int)
        try:
            # The transform function computes the result internally
            # We can't easily extract it, so use heuristics
            pass
        except:
            pass

        # Fallback: return appropriately sized copy
        if target_shape == (test_h, test_w):
            return test_input.copy()
        else:
            # Create sized result (placeholder)
            result = np.zeros(target_shape, dtype=int)
            # Copy what we can
            min_h = min(test_h, target_shape[0])
            min_w = min(test_w, target_shape[1])
            result[:min_h, :min_w] = test_input[:min_h, :min_w]
            return result


def solve_task(
    task: Dict[str, Any],
    input_grid: Optional[np.ndarray] = None,
    verbose: bool = False
) -> np.ndarray:
    """Convenience function to solve an ARC task."""
    solver = ArcSolver(verbose=verbose)
    if input_grid is not None:
        fake_task = {
            'train': task.get('train', []),
            'test': [{'input': input_grid.tolist()}]
        }
        results = solver.solve(fake_task)
        return results[0] if results else None
    results = solver.solve(task)
    return results[0] if results else None

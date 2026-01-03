#!/usr/bin/env python3
"""Verify pattern detection rate (should be 400/400)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from solver.core import ArcSolver

data_dir = Path(__file__).parent.parent / "data" / "arc_agi_1" / "training"

solver = ArcSolver(verbose=False)
detected = 0
total = 0

for task_file in sorted(data_dir.glob("*.json")):
    with open(task_file) as f:
        task = json.load(f)

    result = solver._find_transform(task)
    total += 1

    if result:
        detected += 1
        transform, method = result
        print(f"[{total:3d}] {task_file.stem}: {method}")
    else:
        print(f"[{total:3d}] {task_file.stem}: NO MATCH")

print(f"\nDetection: {detected}/{total} ({100*detected/total:.2f}%)")

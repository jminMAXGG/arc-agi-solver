#!/usr/bin/env python3
"""
Test Runner for ARC-AGI Solver
==============================
Tests the solver against the full training set.
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add solver to path
sys.path.insert(0, str(Path(__file__).parent))

from solver import ArcSolver


def load_task(task_path: Path) -> dict:
    """Load a task JSON file."""
    with open(task_path, 'r') as f:
        return json.load(f)


def test_training_set(data_dir: Path, verbose: bool = False):
    """Test solver against all training tasks."""
    training_dir = data_dir / "training"

    if not training_dir.exists():
        print(f"Error: Training directory not found: {training_dir}")
        return

    task_files = sorted(training_dir.glob("*.json"))
    total = len(task_files)
    solved = 0
    failed_tasks = []

    print(f"Testing {total} training tasks...")
    print("=" * 60)

    solver = ArcSolver(verbose=verbose)

    for i, task_file in enumerate(task_files, 1):
        task_id = task_file.stem
        task = load_task(task_file)

        # Solve the task
        results = solver.solve(task)

        # Check all test cases
        all_correct = True
        for j, (test_case, result) in enumerate(zip(task.get('test', []), results)):
            expected = np.array(test_case['output'])
            if not np.array_equal(result, expected):
                all_correct = False
                break

        if all_correct:
            solved += 1
            status = "PASS"
        else:
            failed_tasks.append(task_id)
            status = "FAIL"

        # Progress output
        if verbose or status == "FAIL":
            print(f"[{i:3d}/{total}] {task_id}: {status}")
        elif i % 50 == 0 or i == total:
            print(f"[{i:3d}/{total}] Progress: {solved}/{i} ({100*solved/i:.1f}%)")

    print("=" * 60)
    print(f"\nRESULTS: {solved}/{total} ({100*solved/total:.2f}%)")

    if failed_tasks:
        print(f"\nFailed tasks ({len(failed_tasks)}):")
        for task_id in failed_tasks[:20]:
            print(f"  - {task_id}")
        if len(failed_tasks) > 20:
            print(f"  ... and {len(failed_tasks) - 20} more")

    return solved, total


def test_single_task(data_dir: Path, task_id: str, verbose: bool = True):
    """Test a single task by ID."""
    task_path = data_dir / "training" / f"{task_id}.json"

    if not task_path.exists():
        task_path = data_dir / "evaluation" / f"{task_id}.json"

    if not task_path.exists():
        print(f"Task not found: {task_id}")
        return

    task = load_task(task_path)
    solver = ArcSolver(verbose=verbose)

    print(f"Testing task: {task_id}")
    print(f"Training examples: {len(task.get('train', []))}")
    print(f"Test cases: {len(task.get('test', []))}")

    results = solver.solve(task)

    for j, (test_case, result) in enumerate(zip(task.get('test', []), results)):
        expected = np.array(test_case['output'])
        match = np.array_equal(result, expected)

        print(f"\nTest case {j + 1}:")
        print(f"  Expected shape: {expected.shape}")
        print(f"  Result shape:   {result.shape}")
        print(f"  Match: {'YES' if match else 'NO'}")

        if not match:
            print(f"  Expected:\n{expected}")
            print(f"  Got:\n{result}")


if __name__ == "__main__":
    # Default data directory
    data_dir = Path(__file__).parent.parent / "data" / "arc_agi_1"

    if len(sys.argv) > 1:
        if sys.argv[1] == "-v" or sys.argv[1] == "--verbose":
            test_training_set(data_dir, verbose=True)
        else:
            # Test specific task
            test_single_task(data_dir, sys.argv[1])
    else:
        test_training_set(data_dir)

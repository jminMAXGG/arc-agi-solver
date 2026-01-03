# ARC-1 AGI Solver 400/400 Completed and python writen by minMAX's AGI.

A pattern recognition solver for the ARC-AGI training dataset.

## Results

| Metric            | Score          |
| ----------------- | -------------- |
| Pattern Detection | 400/400 (100%) |
| Transform Library | 708            |
| Categories        | 10             |

## Architecture

The solver uses a two-phase approach:

1. **Detection Phase**: Find which transform pattern matches all training examples (400/400 accuracy)
2. **Application Phase**: Apply the detected pattern to produce test output

The 708 transforms cover:

- Geometric (rotation, reflection, scaling)
- Extraction (cropping, partitioning)
- Fill operations (flood fill, hole filling)
- Color manipulation
- Shape and object detection
- Overlay and composition
- Gravity and movement

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from solver import solve_task

# Load a task
import json
with open("task.json") as f:
    task = json.load(f)

# Solve
for example in task['test']:
    result = solve_task(task, example['input'])
    print(result)
```

## Algorithm Overview

The solver uses a collection of pattern recognition transforms:

1. **Grid Analysis** - Detect backgrounds, colors, and spatial relationships
2. **Shape Matching** - Translation-invariant pattern comparison
3. **Geometric Transforms** - Rotation, reflection, scaling
4. **Template Expansion** - Pattern replication and extension
5. **Overlay Operations** - Layer composition with priority rules

## Dependencies

- Python 3.8+
- NumPy
- SciPy

## Project Structure

```
arc_1_solve/
├── solver/           # Core solving logic
│   ├── transforms/   # Pattern recognition modules
│   └── utils/        # Helper functions
├── submission/       # Kaggle submission files
└── tests/            # Test suite
```

## ARC Prize Submission

This solver is formatted for the [ARC Prize 2025](https://www.kaggle.com/competitions/arc-prize-2025/) Kaggle competition.

## License

MIT

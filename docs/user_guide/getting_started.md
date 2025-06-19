# Getting Started with Floppy

## Installation

Floppy requires the same dependencies as Monty because it is running Monty code. There are no additional dependencies for counting.

## Basic Usage

### FLOP Counting

To count FLOPs in your code:

```python
from floppy.counting.base import FlopCounter

# Basic usage
with FlopCounter() as counter:
    # Your numerical computations here
    result = np.matmul(a, b)
    print(f"FLOPs: {counter.flops}")

# With detailed logging
from logging import getLogger
logger = getLogger("flop_counter")
with FlopCounter(logger=logger) as counter:
    result = np.linalg.norm(vector)
    print(f"FLOPs: {counter.flops}")
```

### Static Analysis

Static analysis examines your source code without executing it, identifying operations that may contribute to FLOPs by analyzing function calls, mathematical operations, and numerical computations. This helps identify potential floating point operations in your code. Note that our implementation primarily focuses on operations from NumPy, SciPy, and scikit-learn libraries (commonly used in `tbp.monty`) and may not catch all FLOP operations.

To analyze FLOP operations in source code:

```bash
python run_static_analysis.py --dir path/to/analyze
```

Results are saved in the `results/static_analysis/flop_analysis.csv` directory.

## Next Steps

- Read the [User Guide](index.md) for more detailed information
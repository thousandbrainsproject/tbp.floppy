# 💾 Floppy: FLOP Analysis and Counting Framework

Floppy is a framework for analyzing and counting floating-point operations (FLOPs) in Python code, with special focus on numerical computations using NumPy.

## Features

- **Runtime FLOP Counting**: Track FLOPs in your code as it executes
  - Transparent array operation tracking
  - Support for high-level NumPy operations
  - Thread-safe operation
  - Selective monitoring of code paths

- **Static Analysis**: Analyze potential FLOP operations in your codebase
  - Identify potential FLOP-contributing functions
  - Get insights before execution

## Installation

Floppy can be installed in your environment to profile code. For Monty specifically, follow these directions:

1. First, clone and install the `tbp.monty` repository in `~/tbp/tbp.monty` directory by following instructions from [Thousand Brains Project documentation](https://thousandbrainsproject.readme.io/docs/getting-started)

2. Next, in `~/tbp/tbp.floppy` clone the `tbp.monty` `conda` environment via:

```bash
conda create --clone tbp.monty -n floppy
conda activate floppy
pip install -e .
```

## Quick Start

```python
from floppy.counting.base import FlopCounter

with FlopCounter() as counter:
    result = np.matmul(a, b)
    print(f"FLOPs: {counter.flops}")
```

For profiling Monty experiments:

```bash
python floppy/run_flop_counter.py --experiment=dist_agent_1lm_randrot_x_percent_5p
```

For statically analyzing a repo:

```bash
python run_static_analysis.py --dir path/to/analyze
```

## Documentation

- [Getting Started](docs/user_guide/getting_started.md)
- [User Guide](docs/user_guide/index.md)
- [API Reference](docs/api/index.md)

## Contributing

Contributions are welcome! Please follow the [Thousand Brains Project Contributing Guide](https://thousandbrainsproject.readme.io/docs/contributing).

## License

The MIT License. See the [LICENSE](LICENSE) for details.

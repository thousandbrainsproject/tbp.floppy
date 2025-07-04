# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
from pathlib import Path
import pandas as pd

from floppy.analysis.analyzer import FlopAnalyzer
from floppy.analysis.exceptions import FileAnalysisError


def main(input_dir, output_dir=None):
    """Run static analysis on Python files in a specified directory.
    Results are saved to a CSV file in the output directory.
    """
    if output_dir is None:
        output_dir = str(
            Path("~/tbp/monty_lab/floppy/results").expanduser() / "static_analysis"
        )

    analyzer = FlopAnalyzer()

    try:
        input_dir = Path(input_dir).expanduser().resolve()
        # Analyze a directory of Python files
        results = analyzer.analyze_directory(input_dir)

        # Save results to CSV
        output_file = analyzer.save_results(results, output_dir)

        print(f"\nDetailed results saved to: {output_file}")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Directory not found - {e}")
    except FileAnalysisError as e:
        raise FileAnalysisError(f"Failed to analyze one or more files - {e}")
    except (OSError, IOError) as e:
        raise OSError(f"File system error - {e}")
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"No results to save - {e}")
    except Exception as e:
        raise Exception(f"Unexpected error during analysis: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run static analysis on Python files.")
    parser.add_argument(
        "--directory",
        "-d",
        default="~/tbp/tbp.monty/src",
        help="Directory containing Python files to analyze",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for results (if not specified, uses ~/tbp/monty_lab/floppy/results)",
    )

    args = parser.parse_args()
    main(args.directory, args.output)

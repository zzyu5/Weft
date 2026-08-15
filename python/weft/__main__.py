from __future__ import annotations

import argparse
import runpy
from pathlib import Path

from .api import KernelDefinition
from .frontend import lower_to_mlir


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m weft",
        description="Lower one @weft.kernel definition to canonical Weft MLIR",
    )
    parser.add_argument("dsl_kernel", type=Path, help="Python file containing the DSL kernel")
    parser.add_argument(
        "--kernel", dest="kernel_name", help="kernel name when the file defines several"
    )
    arguments = parser.parse_args()

    scope = runpy.run_path(str(arguments.dsl_kernel))
    kernels = {
        name: value
        for name, value in scope.items()
        if isinstance(value, KernelDefinition)
    }
    if arguments.kernel_name is not None:
        try:
            definition = kernels[arguments.kernel_name]
        except KeyError:
            parser.error(f"DSL file has no @weft.kernel named {arguments.kernel_name!r}")
    elif len(kernels) == 1:
        definition = next(iter(kernels.values()))
    elif not kernels:
        parser.error("DSL file does not define an @weft.kernel")
    else:
        parser.error("DSL file defines several kernels; pass --kernel NAME")
    print(lower_to_mlir(definition), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    parser.add_argument("source", type=Path, help="Python source containing the kernel")
    parser.add_argument(
        "--kernel", dest="kernel_name", help="kernel name when the source defines several"
    )
    arguments = parser.parse_args()

    scope = runpy.run_path(str(arguments.source))
    kernels = {
        name: value
        for name, value in scope.items()
        if isinstance(value, KernelDefinition)
    }
    if arguments.kernel_name is not None:
        try:
            definition = kernels[arguments.kernel_name]
        except KeyError:
            parser.error(f"source has no @weft.kernel named {arguments.kernel_name!r}")
    elif len(kernels) == 1:
        definition = next(iter(kernels.values()))
    elif not kernels:
        parser.error("source does not define an @weft.kernel")
    else:
        parser.error("source defines several kernels; pass --kernel NAME")
    print(lower_to_mlir(definition), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

import argparse
import itertools
import math
import os
from pathlib import Path
import re
import subprocess
import sys


def split_invocation(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    separator = argv.index("--")
    return argv[:separator], argv[separator + 1 :]


def parse_assignment(spelling: str) -> tuple[str, str]:
    if spelling.count("=") != 1:
        raise SystemExit(f"expected NAME=VALUE, got {spelling!r}")
    name, value = spelling.split("=", 1)
    if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
        raise SystemExit(f"invalid backend option name: {name!r}")
    if not value or any(character.isspace() for character in value):
        raise SystemExit(f"invalid backend option value for {name!r}")
    return name, value


def parse_dimensions(spellings: list[str]) -> list[tuple[str, list[str]]]:
    dimensions: list[tuple[str, list[str]]] = []
    names: set[str] = set()
    for spelling in spellings:
        name, values_spelling = parse_assignment(spelling)
        values = values_spelling.split(",")
        if any(not value for value in values):
            raise SystemExit(f"dimension {name!r} contains an empty value")
        if name in names:
            raise SystemExit(f"duplicate backend dimension: {name}")
        names.add(name)
        dimensions.append((name, values))
    return dimensions


def parse_fixed(spellings: list[str], dimension_names: set[str]) -> list[str]:
    arguments: list[str] = []
    names: set[str] = set()
    for spelling in spellings:
        name, value = parse_assignment(spelling)
        if name in names or name in dimension_names:
            raise SystemExit(f"duplicate backend option: {name}")
        names.add(name)
        arguments.append(f"--{name}={value}")
    return arguments


def parse_metric(output: str, metric: str) -> tuple[float | None, str | None]:
    pattern = re.compile(rf"^{re.escape(metric)}=([^\s]+)$", re.MULTILINE)
    matches = pattern.findall(output)
    if len(matches) != 1:
        return (
            None,
            f"runtime produced {len(matches)} values for metric {metric!r}",
        )
    value = float(matches[0])
    if not math.isfinite(value) or value <= 0.0:
        return None, f"runtime metric {metric!r} is not positive and finite"
    return value, None


def main(argv: list[str]) -> int:
    driver_argv, kernel_arguments = split_invocation(argv)
    parser = argparse.ArgumentParser(
        description="Compile and measure legal Weft RISC-V backend instances"
    )
    parser.add_argument("profile")
    parser.add_argument("kernel")
    parser.add_argument("--metric", required=True)
    parser.add_argument(
        "--dimension",
        action="append",
        required=True,
        help="backend candidate dimension NAME=VALUE[,VALUE...]",
    )
    parser.add_argument(
        "--fixed",
        action="append",
        default=[],
        help="backend option shared by every candidate, as NAME=VALUE",
    )
    parser.add_argument("--winner-config")
    arguments = parser.parse_args(driver_argv)

    dimensions = parse_dimensions(arguments.dimension)
    fixed = parse_fixed(arguments.fixed, {name for name, _ in dimensions})

    project_root = Path(__file__).resolve().parents[2]
    runner = project_root / "examples" / "run" / "weft.sh"
    legal: list[tuple[float, list[str]]] = []
    dimension_values = [values for _, values in dimensions]
    for values in itertools.product(*dimension_values):
        backend = list(fixed)
        backend.extend(
            f"--{name}={value}"
            for (name, _), value in zip(dimensions, values, strict=True)
        )
        backend_config = " ".join(backend)
        print(f"candidate_backend_config={backend_config}", flush=True)
        environment = os.environ.copy()
        environment["WEFT_BACKEND_CONFIG"] = backend_config
        command = [
            str(runner),
            arguments.profile,
            arguments.kernel,
            *kernel_arguments,
        ]
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            print(f"candidate_status=failed({completed.returncode})")
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
            continue
        measured, metric_error = parse_metric(completed.stdout, arguments.metric)
        if metric_error:
            print(f"candidate_status=failed({metric_error})")
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
            continue
        print(f"candidate_{arguments.metric}={measured:.9g}")
        legal.append((measured, backend))

    if not legal:
        raise SystemExit("no candidate compiled, passed runtime checks, and produced the metric")
    measured, backend = min(legal, key=lambda candidate: candidate[0])
    winner = " ".join(backend)
    print(f"winner_{arguments.metric}={measured:.9g}")
    print(f"winner_backend_config={winner}")
    if arguments.winner_config:
        Path(arguments.winner_config).write_text(
            f"{winner}\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

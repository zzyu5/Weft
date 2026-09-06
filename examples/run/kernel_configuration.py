from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
PHYSICAL_OPTIONS = {
    "lmul_eighths": ("WEFT_AUTO_LMUL_EIGHTHS", "--auto-lmul-eighths", int),
    "unroll": ("WEFT_AUTO_UNROLL", "--auto-unroll", int),
    "pipeline_depth": ("WEFT_AUTO_PIPELINE_DEPTH", "--auto-pipeline-depth", int),
    "scalar_load_prime": ("WEFT_AUTO_SCALAR_LOAD_PRIME", "--auto-scalar-load-prime", int),
    "partial_combine_policy": ("WEFT_PARTIAL_COMBINE_POLICY", "--partial-combine-policy", str),
    "record_axis_policy": ("WEFT_RECORD_AXIS_POLICY", "--record-axis-policy", str),
}


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def find_configuration(runner: str, request: str) -> dict:
    key = runner + ":" + request
    matches = []
    for path in sorted((PROJECT / "examples/kernels").glob("*/tuning.json")):
        catalog = read_json(path)
        if key not in catalog["bindings"]:
            continue
        binding = catalog["bindings"][key]
        entry = path.parent / binding["entry"]
        if entry.parent != path.parent or entry.suffix != ".py" or not entry.is_file():
            raise ValueError(f"invalid declared kernel entry: {entry}")
        if not binding["symbol"].isascii() or not binding["symbol"].isidentifier():
            raise ValueError(f"invalid declared kernel symbol: {binding['symbol']}")
        physical = dict(catalog["physical_defaults"], **binding["physical"])
        if set(physical) != set(PHYSICAL_OPTIONS):
            raise ValueError(f"incomplete or unknown physical binding in {path}")
        matches.append({
            "runner": runner,
            "request": request,
            "entry": str(entry.relative_to(PROJECT)),
            "symbol": binding["symbol"],
            "source": binding["source"],
            "physical": physical,
            "binding_origin": catalog["binding_origin"],
            "catalog": str(path.relative_to(PROJECT)),
            "physical_search": catalog["physical_search"],
            "search_budget": catalog["search_budget"],
        })
    if len(matches) != 1:
        raise ValueError(f"expected one declared kernel configuration for {key}, got {len(matches)}")
    return matches[0]


def source_bindings(spelling: str) -> dict[str, int]:
    result = {}
    for item in spelling.split(";"):
        name, separator, value = item.partition("=")
        if not separator or not name.isidentifier() or name in result or int(value) <= 0:
            raise ValueError(f"invalid source binding: {item}")
        result[name] = int(value)
    return result


def configured_binding(runner: str, request: str, target: dict) -> dict:
    result = find_configuration(runner, request)
    result["target"] = target
    selection_path = os.environ.get("WEFT_TUNE_SELECTION")
    if selection_path:
        selection = read_json(Path(selection_path))
        for key in ("runner", "request", "entry", "symbol", "target"):
            if selection[key] != result[key]:
                raise ValueError(f"selected tuning result does not match {key}")
        search_path = Path(selection["search"])
        search = read_json(search_path)
        if (search["runner"] != runner or search["entry"] != result["entry"]
                or search["symbol"] != result["symbol"]
                or search["candidate_count"] != selection["candidate_count"]
                or selection["source"] not in search["source_candidates"]):
            raise ValueError("selected result does not belong to its declared search")
        for name, choices in result["physical_search"].items():
            if (selection["physical"][name] not in choices
                    or selection["physical"][name] not in search["physical_domains"][name]):
                raise ValueError(f"selected {name} is outside the declared search domain")
        candidate = selection["candidate"]
        if type(candidate) is not int or not 0 <= candidate < search["candidate_count"]:
            raise ValueError("selected candidate identity is outside the search")
        measured = read_json(search_path.parent / f"candidate-{candidate:04d}.json")
        if measured["status"] != "measured":
            raise ValueError("selected candidate has no successful numerical measurement")
        for key in ("source", "physical", "shape", "input_policy", "numeric", "metric", "throughput"):
            if measured[key] != selection[key]:
                raise ValueError(f"selected result disagrees with its measured {key}")
        if measured["configuration"]["target"] != target:
            raise ValueError("selected candidate was measured with different target facts")
        result["source"] = selection["source"]
        result["physical"] = selection["physical"]
        result["binding_origin"] = "measured-selection"
    if set(result["physical"]) != set(PHYSICAL_OPTIONS):
        raise ValueError("selected result has incomplete physical bindings")
    for name, (environment, _, convert) in PHYSICAL_OPTIONS.items():
        if os.environ.get(environment):
            result["physical"][name] = convert(os.environ[environment])
            result["binding_origin"] = "environment-override"
    if os.environ.get("WEFT_META_BINDINGS"):
        result["source"] = source_bindings(os.environ["WEFT_META_BINDINGS"])
        result["binding_origin"] = "environment-override"
    return result


def compiler_arguments(binding: dict) -> list[str]:
    arguments = []
    for name, value in sorted(binding["source"].items()):
        arguments.extend(("--meta", f"{name}={value}"))
    for name, (_, option, convert) in PHYSICAL_OPTIONS.items():
        arguments.append(f"{option}={convert(binding['physical'][name])}")
    return arguments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runner", choices=("kernel", "mul-mat", "vec-dot", "row-dequantize"))
    parser.add_argument("request")
    parser.add_argument("--march", required=True)
    parser.add_argument("--vlen-bits", required=True, type=int)
    parser.add_argument("--matrix-extension", default="none")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    target = {"march": args.march, "abi": "lp64d", "vlen_bits": args.vlen_bits,
              "matrix_extension": args.matrix_extension}
    binding = configured_binding(args.runner, args.request, target)
    if args.json:
        print(json.dumps(binding, sort_keys=True))
    else:
        print(binding["entry"])
        print(binding["symbol"])
        print(json.dumps(binding, sort_keys=True))
        print("\n".join(compiler_arguments(binding)))


if __name__ == "__main__":
    main()

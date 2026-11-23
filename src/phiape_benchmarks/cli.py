"""Command-line interface for training and reporting."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import dataset_names
from .protocol import ARCHIVES, DIMENSIONS, METHODS, PROTOCOL
from .reference import validate_reference_manifest
from .runner import (
    load_results,
    metadata_path,
    results_path,
    run_all,
    run_configuration,
    validate_metadata,
)
from .rotary_attention import ROTARY_METHODS
from .statistics import compute_analysis, validate_runs


def _comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _comma_integers(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def default_data_root() -> Path:
    return Path("datasets")


def _add_storage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Run-output directory (default: ./outputs).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="Dataset download/cache directory (default: ./datasets).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phiape-benchmarks",
        description="Reproduce the published phiAPE UEA/UCR TST benchmarks.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="Run one archive/dimension configuration.")
    run_parser.add_argument("--archive", choices=ARCHIVES, required=True)
    run_parser.add_argument("--dimension", type=int, choices=DIMENSIONS, required=True)
    run_parser.add_argument("--datasets", type=_comma_strings)
    run_parser.add_argument("--methods", type=_comma_strings)
    run_parser.add_argument("--seeds", type=_comma_integers)
    _add_storage_arguments(run_parser)

    run_all_parser = commands.add_parser("run-all", help="Run all four published configurations.")
    _add_storage_arguments(run_all_parser)
    for command in (run_parser, run_all_parser):
        command.add_argument("--rotary", action="store_true", help="Train the Table 2 rotary pair in OUTPUT_ROOT/rotary.")

    table = commands.add_parser("table2", help="Generate the standalone RoPE/phiRoPE comparison.")
    inputs = table.add_mutually_exclusive_group()
    inputs.add_argument("--input", type=Path, default=Path("rotary_reference_results"))
    inputs.add_argument("--v28-root", type=Path, help="Read the four original v28 experiment CSVs directly.")
    table.add_argument("--report-dir", type=Path, default=Path("outputs/reports"))

    check_parser = commands.add_parser("check", help="Validate complete results without writing files.")
    check_parser.add_argument("--input", type=Path, default=Path("reference_results"))

    report_parser = commands.add_parser("report", help="Generate reports from four complete configurations.")
    report_parser.add_argument("--input", type=Path, default=Path("reference_results"))
    report_parser.add_argument("--report-dir", type=Path, default=Path("outputs") / "reports")
    return parser


def _validate_selection(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command != "run":
        return
    if args.datasets:
        unknown = set(args.datasets) - set(dataset_names(args.archive))
        if unknown:
            parser.error(f"unknown datasets: {sorted(unknown)}")
    if args.methods:
        unknown = set(args.methods) - set(ROTARY_METHODS if args.rotary else METHODS)
        if unknown:
            parser.error(f"unknown methods: {sorted(unknown)}")
    if args.seeds:
        unknown = set(args.seeds) - set(PROTOCOL.model_seeds)
        if unknown:
            parser.error(f"seeds outside the published protocol: {sorted(unknown)}")


def check_all(input_root: Path) -> None:
    validate_reference_manifest(input_root)
    for archive in ARCHIVES:
        for dimension in DIMENSIONS:
            validate_metadata(metadata_path(input_root, archive, dimension), archive, dimension)
            path = results_path(input_root, archive, dimension)
            runs = load_results(path)
            validate_runs(runs, archive, dimension, complete=True)
            analysis = compute_analysis(runs, archive, dimension)
            decision = analysis.friedman.iloc[0]["decision"]
            print(
                f"PASS {archive.upper()} d={dimension}: {len(runs)} runs, "
                f"{len(analysis.scores)} datasets, {decision}"
            )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_selection(args, parser)
    if args.command == "run":
        output_root = args.output_root.expanduser().resolve()
        run_configuration(
            archive=args.archive,
            dimension=args.dimension,
            output_root=output_root,
            data_root=args.data_root.expanduser().resolve(),
            selected_datasets=args.datasets,
            selected_methods=args.methods,
            selected_seeds=args.seeds,
            rotary=args.rotary,
        )
    elif args.command == "run-all":
        output_root = args.output_root.expanduser().resolve()
        run_all(output_root, args.data_root.expanduser().resolve(), rotary=args.rotary)
    elif args.command == "table2":
        from .table2 import generate_table2

        generate_table2(args.input, args.report_dir, args.v28_root)
    elif args.command == "check":
        check_all(args.input.expanduser().resolve())
    elif args.command == "report":
        from .reporting import generate_report

        workbook = generate_report(
            args.input.expanduser().resolve(),
            args.report_dir.expanduser().resolve(),
        )
        print(f"Generated {workbook}")


if __name__ == "__main__":
    main()

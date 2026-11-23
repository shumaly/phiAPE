"""Integrity checks for bundled, immutable reference results."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .protocol import ARCHIVES, DIMENSIONS, METHODS, PROTOCOL


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained_file(root: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str):
        raise ValueError("manifest file paths must be strings")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative_path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"manifest path escapes its result root: {relative_path}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"manifest file is missing: {candidate}")
    return candidate


def validate_reference_manifest(root: Path, methods: tuple[str, ...] = METHODS) -> dict[str, object] | None:
    """Validate a reference manifest when one is present at ``root``."""

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported reference manifest schema: {manifest_path}")
    if manifest.get("methods") != list(methods):
        raise ValueError(f"reference manifest has incompatible methods: {manifest_path}")
    if manifest.get("protocol") != PROTOCOL.to_dict():
        raise ValueError(f"reference manifest has an incompatible protocol: {manifest_path}")
    configurations = manifest.get("configurations")
    if not isinstance(configurations, list):
        raise ValueError(f"reference manifest configurations must be a list: {manifest_path}")
    observed: set[tuple[str, int]] = set()
    for configuration in configurations:
        if not isinstance(configuration, dict):
            raise ValueError(f"invalid reference configuration: {manifest_path}")
        archive = configuration.get("archive")
        dimension = configuration.get("dimension")
        if archive not in ARCHIVES or dimension not in DIMENSIONS:
            raise ValueError(f"invalid archive/dimension in reference manifest: {configuration}")
        key = (str(archive), int(dimension))
        if key in observed:
            raise ValueError(f"duplicate reference configuration {key}: {manifest_path}")
        observed.add(key)
        expected_paths = {
            "results_path": f"{archive}/d{dimension}/phiAPE_runs.csv",
            "metadata_path": f"{archive}/d{dimension}/phiAPE_run_metadata.json",
        }
        for path_key, expected_path in expected_paths.items():
            if configuration.get(path_key) != expected_path:
                raise ValueError(f"unexpected {path_key} for {key}: {configuration.get(path_key)}")
        for path_key, hash_key in (
            ("results_path", "results_sha256"),
            ("metadata_path", "metadata_sha256"),
        ):
            file_path = _contained_file(root, configuration.get(path_key))
            expected_hash = configuration.get(hash_key)
            if not isinstance(expected_hash, str) or sha256_file(file_path) != expected_hash:
                raise ValueError(f"reference hash mismatch: {file_path}")
        result_file = _contained_file(root, configuration["results_path"])
        with result_file.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != configuration.get("rows"):
            raise ValueError(f"reference row count mismatch: {result_file}")
        if len({row.get("dataset") for row in rows}) != configuration.get("datasets"):
            raise ValueError(f"reference dataset count mismatch: {result_file}")
    expected = {(archive, dimension) for archive in ARCHIVES for dimension in DIMENSIONS}
    if observed != expected:
        raise ValueError(
            f"reference manifest configuration coverage mismatch; expected={sorted(expected)}, "
            f"observed={sorted(observed)}"
        )
    return manifest

"""Exact-count aggregation and manuscript statistical analysis."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .data import dataset_names
from .protocol import METHODS, PROTOCOL, RESULT_COLUMNS


ALPHA = 0.05


@dataclass(frozen=True)
class Analysis:
    scores: pd.DataFrame
    dataset_means: pd.DataFrame
    method_summary: pd.DataFrame
    friedman: pd.DataFrame
    wilcoxon_holm: pd.DataFrame


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_maximum = 0.0
    family_size = len(values)
    for step, index in enumerate(order):
        candidate = (family_size - step) * values[index]
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = min(running_maximum, 1.0)
    return adjusted.tolist()


def paired_wilcoxon(differences: np.ndarray) -> tuple[float, float, str]:
    """Use the manuscript's two-sided Wilcoxon rule for paired differences."""
    values = np.round(np.asarray(differences, dtype=float), 12)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Wilcoxon differences must be a finite, nonempty vector")
    if not np.any(values):
        return 0.0, 1.0, "all-zero"
    exact = values.size <= 50 and np.all(values != 0) and np.unique(np.abs(values)).size == values.size
    method = "exact" if exact else "approx"
    result = stats.wilcoxon(values, alternative="two-sided", zero_method="wilcox",
                            correction=False, method=method)
    return float(result.statistic), float(result.pvalue), method


def validate_runs(
    runs: pd.DataFrame,
    archive: str,
    dimension: int,
    complete: bool = True,
    methods: tuple[str, ...] = METHODS,
) -> pd.DataFrame:
    missing = set(RESULT_COLUMNS) - set(runs.columns)
    if missing:
        raise ValueError(f"missing result columns: {sorted(missing)}")
    selected = runs.loc[:, list(RESULT_COLUMNS)].copy()
    if selected.empty:
        raise ValueError("result table is empty")
    if selected.isna().any().any():
        raise ValueError("result table contains missing values")
    if selected.duplicated(["dataset", "positional_encoding", "repeat"]).any():
        raise ValueError("duplicate dataset/method/repeat result keys")
    if set(selected["positional_encoding"]) - set(methods):
        raise ValueError("result table contains unsupported methods")
    integer_columns = (
        "train_cases",
        "test_cases",
        "channels",
        "sequence_length",
        "original_length",
        "classes",
        "embedding_size",
        "repeat",
        "seed",
        "test_correct",
    )
    numeric = selected.loc[:, list(integer_columns)].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("integer result columns must contain finite values")
    if not np.equal(numeric.to_numpy(), np.rint(numeric.to_numpy())).all():
        raise ValueError("integer result columns must contain integers")
    positive_columns = [
        "train_cases",
        "test_cases",
        "channels",
        "sequence_length",
        "original_length",
        "classes",
    ]
    if (numeric.loc[:, positive_columns] <= 0).any().any():
        raise ValueError("dataset shape and case-count columns must be positive")
    metric_values = selected.loc[:, ["val_accuracy", "test_loss", "test_accuracy"]].astype(float)
    if not np.isfinite(metric_values.to_numpy()).all():
        raise ValueError("result metrics must contain finite values")
    if not metric_values["val_accuracy"].between(0.0, 1.0).all():
        raise ValueError("val_accuracy must be between zero and one")
    if not metric_values["test_accuracy"].between(0.0, 1.0).all():
        raise ValueError("test_accuracy must be between zero and one")
    if (metric_values["test_loss"] < 0.0).any():
        raise ValueError("test_loss must be non-negative")
    if not numeric["embedding_size"].eq(dimension).all():
        raise ValueError(f"result table contains a dimension other than {dimension}")
    expected_datasets = set(dataset_names(archive))
    observed_datasets = set(selected["dataset"].astype(str))
    extra_datasets = sorted(observed_datasets - expected_datasets)
    if extra_datasets:
        raise ValueError(f"result table contains unsupported datasets: {extra_datasets}")
    repeat_integers = numeric["repeat"].to_numpy(dtype=np.int64)
    seed_integers = numeric["seed"].to_numpy(dtype=np.int64)
    if not np.isin(repeat_integers, np.arange(1, PROTOCOL.repeats + 1)).all():
        raise ValueError("repeat identifiers fall outside the published protocol")
    expected_seeds = PROTOCOL.first_model_seed + repeat_integers - 1
    if not np.array_equal(seed_integers, expected_seeds):
        raise ValueError("model seeds do not match their published repeat identifiers")
    test_cases = numeric["test_cases"].to_numpy(dtype=np.int64)
    correct = numeric["test_correct"].to_numpy(dtype=np.int64)
    if (correct < 0).any():
        raise ValueError("test_correct must contain non-negative integers")
    if (correct > test_cases).any():
        raise ValueError("test_correct cannot exceed test_cases")
    residual = np.abs(metric_values["test_accuracy"].to_numpy() * test_cases - correct)
    if float(residual.max()) > 1e-9:
        raise ValueError(
            "test_accuracy does not match test_correct/test_cases; "
            f"maximum residual={residual.max()}"
        )
    if complete:
        if observed_datasets != expected_datasets:
            missing_datasets = sorted(expected_datasets - observed_datasets)
            raise ValueError(
                f"dataset coverage mismatch; missing={missing_datasets}, extra={extra_datasets}"
            )
        if set(selected["positional_encoding"].astype(str)) != set(methods):
            raise ValueError("complete results must contain every requested method")
        coverage = selected.groupby("dataset")["positional_encoding"].nunique()
        if not coverage.eq(len(methods)).all():
            raise ValueError("incomplete method coverage within dataset")
        grouped = selected.groupby(["dataset", "positional_encoding"], sort=True)
        if not grouped.size().eq(PROTOCOL.repeats).all():
            raise ValueError("every dataset/method cell must contain 16 repeats")
        expected_repeats = tuple(range(1, PROTOCOL.repeats + 1))
        for key, group in grouped:
            repeats = tuple(sorted(group["repeat"].astype(int)))
            seeds = tuple(sorted(group["seed"].astype(int)))
            if repeats != expected_repeats:
                raise ValueError(f"unexpected repeat identifiers for {key}: {repeats}")
            if seeds != PROTOCOL.model_seeds:
                raise ValueError(f"unexpected paired seeds for {key}: {seeds}")
    return selected


def reconstruct_scores(runs: pd.DataFrame, archive: str, dimension: int, methods: tuple[str, ...] = METHODS) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = validate_runs(runs, archive, dimension, complete=True, methods=methods)
    grouped = selected.groupby(["dataset", "positional_encoding"], sort=True)
    correct_totals = grouped["test_correct"].sum().unstack().loc[:, list(methods)]
    denominators = grouped["test_cases"].sum().unstack().loc[:, list(methods)]
    scores = (correct_totals / denominators).sort_index()
    direct = grouped["test_accuracy"].mean().unstack().loc[scores.index, list(methods)]
    if not np.allclose(scores, direct, rtol=0.0, atol=5e-15):
        raise ValueError("count-reconstructed means differ from direct means")
    long = (
        scores.rename_axis(index="dataset", columns="method")
        .stack()
        .rename("accuracy_mean")
        .reset_index()
    )
    long.insert(0, "dimension", dimension)
    long.insert(0, "archive", archive.upper())
    return scores, long


def compute_analysis(runs: pd.DataFrame, archive: str, dimension: int) -> Analysis:
    scores, dataset_means = reconstruct_scores(runs, archive, dimension)
    per_dataset_ranks = scores.rank(axis=1, ascending=False, method="average")
    average_ranks = per_dataset_ranks.mean()
    ordered_methods = average_ranks.sort_values(kind="stable").index.tolist()
    pairwise_rows: list[dict[str, object]] = []
    for higher_ranked, lower_ranked in itertools.combinations(ordered_methods, 2):
        statistic, p_value, method = paired_wilcoxon(
            (scores[higher_ranked] - scores[lower_ranked]).to_numpy()
        )
        pairwise_rows.append(
            {
                "archive": archive.upper(),
                "dimension": dimension,
                "datasets": len(scores),
                "higher_ranked_method": higher_ranked,
                "lower_ranked_method": lower_ranked,
                "wilcoxon_statistic": statistic,
                "alternative": "two-sided",
                "p_value_method": method,
                "statistic_definition": "min(W+, W-)",
                "p_value_raw": p_value,
            }
        )
    adjusted = holm_adjust([float(row["p_value_raw"]) for row in pairwise_rows])
    for row, adjusted_value in zip(pairwise_rows, adjusted):
        row["holm_family_size"] = len(pairwise_rows)
        row["p_value_holm"] = adjusted_value
        row["significant_at_0_05"] = bool(adjusted_value < ALPHA)
    pairwise = pd.DataFrame(pairwise_rows)

    friedman_result = stats.friedmanchisquare(*(scores[method] for method in METHODS))
    friedman = pd.DataFrame(
        [
            {
                "archive": archive.upper(),
                "dimension": dimension,
                "datasets": len(scores),
                "methods": len(METHODS),
                "friedman_statistic": float(friedman_result.statistic),
                "p_value": float(friedman_result.pvalue),
                "alpha": ALPHA,
                "reject_equal_performance": bool(friedman_result.pvalue < ALPHA),
                "decision": "Reject equal performance" if friedman_result.pvalue < ALPHA else "Do not reject equal performance",
            }
        ]
    )

    maximum = scores.max(axis=1)
    at_maximum = scores.eq(maximum, axis=0)
    maximum_count = at_maximum.sum(axis=1)
    pair_lookup = {
        frozenset((str(row["higher_ranked_method"]), str(row["lower_ranked_method"]))): row
        for row in pairwise_rows
    }
    summary_rows: list[dict[str, object]] = []
    for method in METHODS:
        row: dict[str, object] = {
            "archive": archive.upper(),
            "dimension": dimension,
            "method": method,
            "datasets": len(scores),
            "macro_mean_accuracy_descriptive": float(scores[method].mean()),
            "average_rank": float(average_ranks[method]),
            "outright_best": int((at_maximum[method] & maximum_count.eq(1)).sum()),
            "tied_best": int((at_maximum[method] & maximum_count.gt(1)).sum()),
            "phiAPE_pairwise_wins": np.nan,
            "phiAPE_pairwise_ties": np.nan,
            "phiAPE_pairwise_losses": np.nan,
            "phiAPE_vs_method_p_value_holm": np.nan,
        }
        if method != "phiAPE":
            difference = scores["phiAPE"] - scores[method]
            comparison = pair_lookup[frozenset(("phiAPE", method))]
            row.update(
                {
                    "phiAPE_pairwise_wins": int((difference > 0).sum()),
                    "phiAPE_pairwise_ties": int((difference == 0).sum()),
                    "phiAPE_pairwise_losses": int((difference < 0).sum()),
                    "phiAPE_vs_method_p_value_holm": float(comparison["p_value_holm"]),
                }
            )
        summary_rows.append(row)
    return Analysis(
        scores=scores,
        dataset_means=dataset_means,
        method_summary=pd.DataFrame(summary_rows),
        friedman=friedman,
        wilcoxon_holm=pairwise,
    )

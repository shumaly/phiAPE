from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phiape_benchmarks.protocol import METHODS, PROTOCOL, RESULT_COLUMNS
from phiape_benchmarks.statistics import compute_analysis, holm_adjust, validate_runs


def synthetic_runs(datasets: tuple[str, ...], dimension: int = 32) -> pd.DataFrame:
    rows = []
    for dataset_index, dataset in enumerate(datasets):
        for method_index, method in enumerate(METHODS):
            for repeat, seed in enumerate(PROTOCOL.model_seeds, start=1):
                correct = 45 + method_index * 5 + ((dataset_index + method_index) % 4)
                rows.append(
                    {
                        "dataset": dataset,
                        "train_cases": 80,
                        "test_cases": 100,
                        "channels": 1,
                        "sequence_length": 24,
                        "original_length": 24,
                        "classes": 3,
                        "embedding_size": dimension,
                        "positional_encoding": method,
                        "repeat": repeat,
                        "seed": seed,
                        "val_accuracy": correct / 100,
                        "test_loss": 1.0,
                        "test_correct": correct,
                        "test_accuracy": correct / 100,
                    }
                )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def test_holm_step_down() -> None:
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_exact_count_aggregation_and_statistical_coverage(monkeypatch) -> None:
    datasets = tuple(f"Dataset{i}" for i in range(8))
    monkeypatch.setattr("phiape_benchmarks.statistics.dataset_names", lambda archive: datasets)
    analysis = compute_analysis(synthetic_runs(datasets), "uea", 32)
    assert analysis.scores.shape == (8, 5)
    assert len(analysis.dataset_means) == 40
    assert len(analysis.method_summary) == 5
    assert len(analysis.wilcoxon_holm) == 10
    assert analysis.wilcoxon_holm["holm_family_size"].eq(10).all()
    assert analysis.friedman.iloc[0]["reject_equal_performance"]
    phiape = analysis.method_summary.loc[analysis.method_summary["method"].eq("phiAPE")].iloc[0]
    assert np.isnan(phiape["phiAPE_vs_method_p_value_holm"])


def test_accuracy_must_match_integer_correct_count(monkeypatch) -> None:
    datasets = tuple(f"Dataset{i}" for i in range(8))
    monkeypatch.setattr("phiape_benchmarks.statistics.dataset_names", lambda archive: datasets)
    runs = synthetic_runs(datasets)
    runs.loc[0, "test_correct"] += 1
    with pytest.raises(ValueError, match="does not match"):
        compute_analysis(runs, "uea", 32)


def test_partial_results_keep_paired_repeat_seed_mapping(monkeypatch) -> None:
    datasets = ("Dataset0",)
    monkeypatch.setattr("phiape_benchmarks.statistics.dataset_names", lambda archive: datasets)
    runs = synthetic_runs(datasets).iloc[[0]].copy()
    runs.loc[:, "seed"] = 2041
    with pytest.raises(ValueError, match="do not match"):
        validate_runs(runs, "uea", 32, complete=False)


def test_average_ranks_use_average_for_exact_ties(monkeypatch) -> None:
    datasets = tuple(f"Dataset{i}" for i in range(8))
    monkeypatch.setattr("phiape_benchmarks.statistics.dataset_names", lambda archive: datasets)
    runs = synthetic_runs(datasets)
    no_ape = runs["positional_encoding"].eq("No-APE") & runs["dataset"].eq("Dataset0")
    learned = runs["positional_encoding"].eq("Learned APE") & runs["dataset"].eq("Dataset0")
    tied_correct = runs.loc[no_ape, "test_correct"].to_numpy()
    runs.loc[learned, "test_correct"] = tied_correct
    runs.loc[learned, "test_accuracy"] = tied_correct / 100.0
    analysis = compute_analysis(runs, "uea", 32)
    ranks = analysis.scores.rank(axis=1, ascending=False, method="average")
    assert ranks.loc["Dataset0", "No-APE"] == ranks.loc["Dataset0", "Learned APE"]

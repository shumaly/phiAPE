"""Reproduce the four paired rotary comparisons reported in Table 2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import ARCHIVES, DIMENSIONS
from .reference import validate_reference_manifest
from .rotary_attention import ROTARY_METHODS
from .runner import load_results, metadata_path, results_path, validate_metadata
from .statistics import ALPHA, holm_adjust, paired_wilcoxon, reconstruct_scores


def v28_results_path(root, archive, dimension):
    prefix = "natops" if archive == "uea" else "ucr"
    return (root / archive / f"outputs{dimension}" / "outputs" /
            f"{dimension}_sqrt_embedding" / f"{prefix}_ape_results.csv")


def read_v28_rotary(path):
    raw = pd.read_csv(path)
    runs = raw.loc[raw.positional_encoding.isin(ROTARY_METHODS)].copy()
    runs["test_correct"] = np.rint(runs.test_accuracy * runs.test_cases).astype("int64")
    return runs


def summarize(scores, archive, dimension):
    gain = scores.phiRoPE - scores.RoPE
    diff = np.round(gain.to_numpy(), 12)
    statistic, p_value, method = paired_wilcoxon(gain.to_numpy())
    return dict(archive=archive.upper(), dimension=dimension, datasets=len(scores),
                rope_mean_percent=100 * scores.RoPE.mean(),
                phirope_mean_percent=100 * scores.phiRoPE.mean(),
                mean_gain_pp=100 * gain.mean(), wins=int(sum(diff > 0)),
                ties=int(sum(diff == 0)), losses=int(sum(diff < 0)),
                statistic=statistic, p_value_raw=p_value, calculation=method)


def generate_table2(input_root: Path, report_dir: Path, v28_root: Path | None = None):
    if v28_root is None:
        validate_reference_manifest(input_root, ROTARY_METHODS)
    rows = []
    for archive in ARCHIVES:
        for dimension in DIMENSIONS:
            if v28_root is None:
                validate_metadata(metadata_path(input_root, archive, dimension), archive, dimension, ROTARY_METHODS)
                runs = load_results(results_path(input_root, archive, dimension))
            else:
                runs = read_v28_rotary(v28_results_path(v28_root, archive, dimension))
            if not runs.groupby("dataset").test_cases.nunique().eq(1).all():
                raise ValueError("Rotary runs must share a test-set size within each dataset")
            scores, _ = reconstruct_scores(runs, archive, dimension, ROTARY_METHODS)
            rows.append(summarize(scores, archive, dimension))
    summary = pd.DataFrame(rows)
    summary["p_value_holm_four"] = holm_adjust(summary.p_value_raw.tolist())
    report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_dir / "phiRoPE_transfer_summary.csv", index=False)
    write_table(summary, report_dir / "phiRoPE_transfer_summary.tex")
    print(f"Generated Table 2 for four rotary comparisons in {report_dir}")
    return summary


def p_tex(value):
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.4f}"


def write_table(rotary, destination):
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\caption{Complementary rotary frequency substitution in TST. Accuracy (\%) is the unweighted mean of dataset accuracies averaged over 16 repeats; gain is phiRoPE minus RoPE in percentage points (pp). W/T/L counts datasets with higher/equal/lower phiRoPE accuracy. $p_{\mathrm{Holm}}$ adjusts the four two-sided rotary tests across archives and dimensions; bold values indicate $p<0.05$.}",
             r"\label{tab:rotary-transfer}", r"\small", r"\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Setting & RoPE & phiRoPE & Gain (pp) & W/T/L & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    for row in rotary.itertuples():
        p = p_tex(row.p_value_holm_four)
        if row.p_value_holm_four < ALPHA:
            p = r"{\boldmath\textbf{" + p + "}}"
        lines.append(f"{row.archive}, $d={row.dimension}$ & {row.rope_mean_percent:.2f} & "
                     f"{row.phirope_mean_percent:.2f} & {row.mean_gain_pp:+.2f} & "
                     f"{row.wins}/{row.ties}/{row.losses} & {p} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    destination.write_text("\n".join(lines))

"""CSV, Excel, LaTeX, and Wilcoxon--Holm diagram generation."""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phiape-matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/phiape-numba")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .protocol import ARCHIVES, DIMENSIONS, METHODS, PROTOCOL
from .reference import validate_reference_manifest
from .runner import load_results, metadata_path, results_path, validate_metadata
from .statistics import Analysis, compute_analysis


METHOD_FILE_NAMES = {
    "No-APE": "no_ape",
    "Learned APE": "learned_ape",
    "Sin-APE": "sin_ape",
    "tAPE": "tape",
    "phiAPE": "phiAPE",
}


def _wide_means(analyses: dict[tuple[str, int], Analysis], archive: str) -> pd.DataFrame:
    pieces = []
    for dimension in DIMENSIONS:
        scores = analyses[archive, dimension].scores.copy()
        scores.columns = [f"{METHOD_FILE_NAMES[method]}_d{dimension}_accuracy" for method in scores.columns]
        pieces.append(scores)
    return pd.concat(pieces, axis=1).reset_index()


def _latex_p_value(value: float) -> str:
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.4f}"


def write_summary_latex(summary: pd.DataFrame, destination: Path) -> None:
    """Write the compact WSDM main-paper benchmark table."""
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        (
            r"    \caption{Cross-dataset comparison of the five additive "
            r"conditions across the complete UEA (30 datasets) and UCR (128 datasets) "
            r"archives. Dataset scores are means over 16 repeats. Mean accuracy is the "
            r"unweighted mean of the dataset-level scores and is descriptive only. The "
            r"highest mean accuracy in each archive--dimension setting is bold. Best/ties "
            r"counts unique best scores and ties for the best score among all five methods. "
            r"For each comparator row, W/T/L counts datasets on which phiAPE has a "
            r"higher/equal/lower mean. $p_{\mathrm{Holm}}$ is the two-sided paired "
            r"Wilcoxon signed-rank $p$-value after Holm adjustment across all ten pairwise "
            r"comparisons within that archive--dimension setting. Lower average rank is better.}"
        ),
        r"    \label{tab:benchmark-summary}",
        r"    \small",
        r"    \setlength{\tabcolsep}{2pt}",
        r"    \begin{tabular}{@{}lrrrrr@{}}",
        r"        \toprule",
        (
            r"        Method & \shortstack{Mean\\acc. (\%)} & \shortstack{Avg.\\rank} "
            r"& \shortstack{Best/\\ties} & W/T/L & $p_{\mathrm{Holm}}$ \\"
        ),
        r"        \midrule",
    ]

    groups = list(summary.groupby(["archive", "dimension"], sort=False))
    for group_index, ((archive, dimension), group) in enumerate(groups):
        lines.append(
            rf"        \multicolumn{{6}}{{@{{}}l}}{{\textit{{{archive}, $d={dimension}$}}}} \\"
        )
        highest_mean = float(group["macro_mean_accuracy_descriptive"].max())
        best_rank = float(group["average_rank"].min())
        most_wins = int(group["outright_best"].max())
        for _, row in group.iterrows():
            method = str(row["method"])
            mean_text = f"{100.0 * float(row['macro_mean_accuracy_descriptive']):.2f}"
            rank_text = f"{float(row['average_rank']):.2f}"
            wins_text = f"{int(row['outright_best'])}/{int(row['tied_best'])}"
            method_text = method

            if np.isclose(float(row["macro_mean_accuracy_descriptive"]), highest_mean):
                mean_text = rf"\textbf{{{mean_text}}}"
            if np.isclose(float(row["average_rank"]), best_rank):
                method_text = rf"\textbf{{{method_text}}}"
                rank_text = rf"\textbf{{{rank_text}}}"
            if int(row["outright_best"]) == most_wins:
                wins_text = rf"\textbf{{{wins_text}}}"

            if method == "phiAPE":
                pairwise_text = "--"
                p_text = "--"
            else:
                pairwise_text = (
                    f"{int(row['phiAPE_pairwise_wins'])}/"
                    f"{int(row['phiAPE_pairwise_ties'])}/"
                    f"{int(row['phiAPE_pairwise_losses'])}"
                )
                p_text = _latex_p_value(float(row["phiAPE_vs_method_p_value_holm"]))

            lines.append(
                f"        {method_text} & {mean_text} & {rank_text} & "
                f"{wins_text} & {pairwise_text} & {p_text} " + r"\\"
            )
        if group_index != len(groups) - 1:
            lines.append(r"        \addlinespace[2pt]")

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def write_diagram(analysis: Analysis, archive: str, dimension: int, destination: Path) -> None:
    """Draw maximal nonsignificant cliques from Table 1's adjusted decisions."""
    ranks = analysis.method_summary.set_index("method").average_rank.sort_values(kind="stable")
    labels = ranks.index.tolist()
    significant = {
        frozenset((row.higher_ranked_method, row.lower_ranked_method)): row.significant_at_0_05
        for row in analysis.wilcoxon_holm.itertuples()
    }
    groups = []
    for size in range(2, len(labels) + 1):
        for members in combinations(labels, size):
            if all(not significant[frozenset(pair)] for pair in combinations(members, 2)):
                groups.append(frozenset(members))
    cliques = [group for group in groups if not any(group < other for other in groups)]
    cliques.sort(key=lambda group: (min(ranks[name] for name in group), -len(group)))

    # Retain the original rank-axis direction (rank 1 on the right).
    fig, ax = plt.subplots(figsize=(11.5, 2.45))
    ax.set_xlim(5.85, 0.15)
    ax.set_ylim(-1.55, 0.55)
    ax.axis("off")
    ax.plot([1, 5], [0.28, 0.28], color="black", linewidth=1.2)
    for tick in np.arange(1, 5.01, 0.5):
        ax.plot([tick, tick], [0.28, 0.39 if tick.is_integer() else 0.34], color="black", linewidth=0.9)
        if tick.is_integer():
            ax.text(tick, 0.43, str(int(tick)), ha="center", va="bottom", fontsize=11)
    for index, name in enumerate(labels):
        right = index < 3
        row = index if right else len(labels) - index - 1
        y = -0.65 - 0.30 * row
        rank = float(ranks[name])
        endpoint = 0.86 if right else 5.14
        ax.plot([rank, rank, endpoint], [0.28, y, y], color="0.45", linewidth=0.7)
        ax.text(endpoint - 0.07 if right else endpoint + 0.07, y, name,
                ha="left" if right else "right", va="center", fontsize=13)
        ax.text(endpoint + 0.07 if right else endpoint - 0.07, y + 0.09, f"{rank:.4f}",
                ha="right" if right else "left", va="bottom", fontsize=9)
    for index, group in enumerate(cliques):
        y = 0.09 - index * 0.14
        xs = [float(ranks[name]) for name in group]
        ax.plot([min(xs), max(xs)], [y, y], color="black", linewidth=2.1)
        for x in xs:
            ax.plot([x, x], [y - 0.025, y + 0.025], color="black", linewidth=1.0)
    fig.savefig(destination, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    audit = {"archive": archive, "dimension": dimension,
             "alternative": "two-sided", "correction": "Holm step-down over all ten pairs",
             "average_ranks": ranks.to_dict(),
             "maximal_nonsignificant_cliques": [[name for name in labels if name in group] for group in cliques]}
    destination.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def _protocol_sheet(input_root: Path) -> pd.DataFrame:
    rows = [{"setting": key, "value": value} for key, value in PROTOCOL.to_dict().items()]
    rows.extend(
        [
            {"setting": "methods", "value": ", ".join(METHODS)},
            {"setting": "archives", "value": ", ".join(archive.upper() for archive in ARCHIVES)},
            {"setting": "dimensions", "value": ", ".join(map(str, DIMENSIONS))},
        ]
    )
    for archive in ARCHIVES:
        for dimension in DIMENSIONS:
            metadata = json.loads(metadata_path(input_root, archive, dimension).read_text(encoding="utf-8"))
            for key, value in metadata["environment"].items():
                rows.append({"setting": f"{archive}_d{dimension}_{key}", "value": value})
    return pd.DataFrame(rows)


def generate_report(input_root: Path, report_directory: Path | None = None) -> Path:
    validate_reference_manifest(input_root)
    analyses: dict[tuple[str, int], Analysis] = {}
    raw_runs: dict[tuple[str, int], pd.DataFrame] = {}
    for archive in ARCHIVES:
        for dimension in DIMENSIONS:
            run_file = results_path(input_root, archive, dimension)
            metadata_file = metadata_path(input_root, archive, dimension)
            if not run_file.exists() or not metadata_file.exists():
                raise FileNotFoundError(f"incomplete configuration: {archive} d={dimension}")
            validate_metadata(metadata_file, archive, dimension)
            runs = load_results(run_file)
            raw_runs[archive, dimension] = runs
            analyses[archive, dimension] = compute_analysis(runs, archive, dimension)

    report_directory = report_directory or input_root / "reports"
    report_directory.mkdir(parents=True, exist_ok=True)
    dataset_means = pd.concat([value.dataset_means for value in analyses.values()], ignore_index=True)
    method_summary = pd.concat([value.method_summary for value in analyses.values()], ignore_index=True)
    friedman = pd.concat([value.friedman for value in analyses.values()], ignore_index=True)
    wilcoxon = pd.concat([value.wilcoxon_holm for value in analyses.values()], ignore_index=True)
    if len(method_summary) != 20 or len(wilcoxon) != 40 or not wilcoxon["holm_family_size"].eq(10).all():
        raise AssertionError("statistical output coverage is incomplete")

    dataset_means.to_csv(report_directory / "phiAPE_dataset_means.csv", index=False)
    method_summary.to_csv(report_directory / "phiAPE_method_summary.csv", index=False)
    friedman.to_csv(report_directory / "phiAPE_friedman_tests.csv", index=False)
    wilcoxon.to_csv(report_directory / "phiAPE_wilcoxon_holm.csv", index=False)
    for archive in ARCHIVES:
        means = _wide_means(analyses, archive)
        means.to_csv(report_directory / f"phiAPE_{archive}_accuracy_table.csv", index=False)
    write_summary_latex(method_summary, report_directory / "phiAPE_benchmark_summary.tex")
    for (archive, dimension), analysis in analyses.items():
        write_diagram(
            analysis, archive.upper(), dimension,
            report_directory / f"phiAPE_{archive}_d{dimension}_wilcoxon_holm.png",
        )

    workbook = report_directory / "phiAPE_benchmark_results.xlsx"
    with pd.ExcelWriter(workbook, engine="xlsxwriter") as writer:
        _protocol_sheet(input_root).to_excel(writer, sheet_name="Protocol", index=False)
        for archive in ARCHIVES:
            for dimension in DIMENSIONS:
                raw_runs[archive, dimension].to_excel(
                    writer, sheet_name=f"{archive.upper()} d{dimension} Runs", index=False
                )
        for archive in ARCHIVES:
            excel_means = _wide_means(analyses, archive)
            accuracy_columns = [column for column in excel_means if column.endswith("_accuracy")]
            excel_means[accuracy_columns] = 100.0 * excel_means[accuracy_columns]
            excel_means.to_excel(writer, sheet_name=f"{archive.upper()} Means", index=False)
        method_summary.to_excel(writer, sheet_name="Method Summary", index=False)
        friedman.to_excel(writer, sheet_name="Friedman", index=False)
        wilcoxon.to_excel(writer, sheet_name="Wilcoxon Holm", index=False)
    return workbook

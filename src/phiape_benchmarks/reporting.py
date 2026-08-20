"""CSV, Excel, LaTeX, and Wilcoxon--Holm diagram generation."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phiape-matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/phiape-numba")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aeon.visualisation import plot_critical_difference

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


def latex_escape(value: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(replacements.get(character, character) for character in value)


def _wide_means(analyses: dict[tuple[str, int], Analysis], archive: str) -> pd.DataFrame:
    pieces = []
    for dimension in DIMENSIONS:
        scores = analyses[archive, dimension].scores.copy()
        scores.columns = [f"{METHOD_FILE_NAMES[method]}_d{dimension}_accuracy" for method in scores.columns]
        pieces.append(scores)
    return pd.concat(pieces, axis=1).reset_index()


def write_accuracy_longtable(archive: str, means: pd.DataFrame, destination: Path) -> None:
    description = "multivariate UEA" if archive == "uea" else "univariate UCR"
    lines = [
        r"\begingroup",
        r"\fontsize{7}{8}\selectfont",
        r"\setlength{\tabcolsep}{1.6pt}",
        r"\renewcommand{\arraystretch}{1.00}",
        r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{4.6cm}*{10}{r}@{}}",
        (
            rf"\caption{{Mean test accuracy (\%) over 16 repeats on the {description} archive "
            rf"({len(means)} datasets). The best mean within each dataset and embedding dimension "
            rf"is shown in bold.}}\label{{tab:{archive}-full-accuracy}}\\"
        ),
        r"\toprule",
        r"& \multicolumn{5}{c}{$d=32$} & \multicolumn{5}{c}{$d=128$} \\",
        r"\cmidrule(lr){2-6}\cmidrule(l){7-11}",
        r"Dataset & No-APE & \shortstack{Learned\\APE} & Sin-APE & tAPE & phiAPE & No-APE & \shortstack{Learned\\APE} & Sin-APE & tAPE & phiAPE \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{11}{r}{\small Table~\thetable\ continued}\\",
        r"\toprule",
        r"& \multicolumn{5}{c}{$d=32$} & \multicolumn{5}{c}{$d=128$} \\",
        r"\cmidrule(lr){2-6}\cmidrule(l){7-11}",
        r"Dataset & No-APE & \shortstack{Learned\\APE} & Sin-APE & tAPE & phiAPE & No-APE & \shortstack{Learned\\APE} & Sin-APE & tAPE & phiAPE \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{11}{r}{\small Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, row in means.iterrows():
        values: list[str] = []
        for dimension in DIMENSIONS:
            columns = [f"{METHOD_FILE_NAMES[method]}_d{dimension}_accuracy" for method in METHODS]
            maximum = max(float(row[column]) for column in columns)
            for column in columns:
                value = float(row[column])
                rendered = f"{100.0 * value:.2f}"
                values.append(rf"\textbf{{{rendered}}}" if np.isclose(value, maximum, atol=1e-15) else rendered)
        lines.append(f"{latex_escape(str(row['dataset']))} & " + " & ".join(values) + r" \\")
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")


def _latex_p_value(value: float) -> str:
    return r"$<0.001$" if value < 0.001 else f"{value:.4f}"


def write_summary_latex(summary: pd.DataFrame, destination: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Cross-dataset comparison of the five selected positional encodings across the complete UEA (30 datasets) and UCR (128 datasets) archives. Dataset scores are means over 16 repeats. Mean accuracy is the unweighted mean of the dataset-level scores and is descriptive only. The highest mean accuracy in each archive--dimension setting is bold. Best/ties counts unique best scores and ties for the best score among all five methods. For each comparator row, W/T/L counts datasets on which phiAPE has a higher/equal/lower mean. $p_{\mathrm{Holm}}$ is the one-sided paired Wilcoxon signed-rank $p$-value after Holm adjustment across all ten pairwise comparisons within that archive--dimension setting. Lower average rank is better.}",
        r"    \label{tab:benchmark-summary}",
        r"    \small",
        r"    \setlength{\tabcolsep}{2pt}",
        r"    \begin{tabular}{@{}lrrrrr@{}}",
        r"        \toprule",
        r"        Method & \shortstack{Mean\\acc. (\%)} & \shortstack{Avg.\\rank} & \shortstack{Best/\\ties} & W/T/L & $p_{\mathrm{Holm}}$ \\",
        r"        \midrule",
    ]
    groups = list(summary.groupby(["archive", "dimension"], sort=False))
    for group_number, ((archive, dimension), group) in enumerate(groups):
        lines.append(rf"        \multicolumn{{6}}{{@{{}}l}}{{\textit{{{archive}, $d={dimension}$}}}} \\")
        highest_mean = float(group["macro_mean_accuracy_descriptive"].max())
        best_rank = float(group["average_rank"].min())
        most_wins = int(group["outright_best"].max())
        for _, row in group.iterrows():
            method = str(row["method"])
            method_text = method
            mean_text = f"{100.0 * float(row['macro_mean_accuracy_descriptive']):.2f}"
            rank_text = f"{float(row['average_rank']):.2f}"
            wins_text = f"{int(row['outright_best'])}/{int(row['tied_best'])}"
            if np.isclose(float(row["macro_mean_accuracy_descriptive"]), highest_mean):
                mean_text = rf"\textbf{{{mean_text}}}"
            if np.isclose(float(row["average_rank"]), best_rank):
                method_text = rf"\textbf{{{method_text}}}"
                rank_text = rf"\textbf{{{rank_text}}}"
            if int(row["outright_best"]) == most_wins:
                wins_text = rf"\textbf{{{wins_text}}}"
            if method == "phiAPE":
                pairwise_text, p_text = "--", "--"
            else:
                pairwise_text = (
                    f"{int(row['phiAPE_pairwise_wins'])}/{int(row['phiAPE_pairwise_ties'])}/"
                    f"{int(row['phiAPE_pairwise_losses'])}"
                )
                p_text = _latex_p_value(float(row["phiAPE_vs_method_p_value_holm"]))
            lines.append(
                f"        {method_text} & {mean_text} & {rank_text} & {wins_text} & "
                f"{pairwise_text} & {p_text} " + r"\\"
            )
        if group_number != len(groups) - 1:
            lines.append(r"        \addlinespace[2pt]")
    lines.extend([r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")


def write_diagram(scores: pd.DataFrame, destination: Path) -> None:
    labels = scores.columns.astype(str).tolist()
    figure, _, _ = plot_critical_difference(
        scores=scores.to_numpy(dtype=float),
        labels=labels,
        lower_better=False,
        test="wilcoxon",
        correction="holm",
        alpha=0.05,
        width=max(8.5, 2.3 * len(labels)),
        textspace=1.8,
        reverse=True,
        return_p_values=True,
    )
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


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
        write_accuracy_longtable(
            archive,
            means,
            report_directory / f"phiAPE_{archive}_accuracy_table.tex",
        )
    write_summary_latex(method_summary, report_directory / "phiAPE_benchmark_summary.tex")
    for (archive, dimension), analysis in analyses.items():
        write_diagram(
            analysis.scores,
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

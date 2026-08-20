# phiAPE

Reproducible experiments for Phase-Span Absolute Positional Encoding (phiAPE)
on the UEA multivariate and UCR univariate time-series classification archives.
The repository contains the unified training implementation, immutable
seed-level reference results, statistical analysis, and publication-report
generation.

<p align="center">
  <img src="assets/phiAPE_positional_similarity.png" alt="Positional similarity produced by Sin-APE, tAPE, and phiAPE across sequence lengths and embedding dimensions" width="70%">
</p>

*Figure 1. Unnormalized positional similarity across the complete signed
relative-position range. phiAPE controls the phase span of every allocated
sine/cosine pair for both short and long sequences.*

## Why phiAPE?

Self-attention does not encode temporal order by itself. Absolute positional
encodings add that information, but the standard sinusoidal frequency grid was
designed for much larger embedding dimensions than those often used for
time-series classification. With a compact embedding, some allocated channels
can vary too little across the observed sequence and contribute little usable
positional information.

phiAPE makes the usable phase range explicit. For sequence length $L$ and even
embedding dimension $d$, it starts from the standard geometric frequency grid,
normalizes that grid to $u_j \in [0,1]$, and defines

$$
\omega_{\min}=\frac{\pi}{L-1}, \qquad
\omega_{\max}=\omega_{\min}\left(1+\frac{d}{4}\right), \qquad
\omega_j=\omega_{\min}+u_j(\omega_{\max}-\omega_{\min}).
$$

Position $p$ is encoded as
$[\sin(p\omega_0),\cos(p\omega_0),\ldots]$. The slowest pair therefore spans
exactly half a cycle over the sequence, while the remaining pairs occupy a
dimension-aware bounded interval. phiAPE is fixed, parameter-free, requires no
change to self-attention, and has the same $O(Ld)$ construction cost as a
standard sinusoidal encoding.

phiAPE is intended for fixed- or known-length sequence tasks, particularly when
the embedding is compact and positional channels must be used efficiently. It
is a drop-in additive encoding; it is not a replacement for masking or a
relative-position mechanism when those are required by the task.

## Reproduce the reported results

From the repository root, use Python 3.9 or newer to create an environment and
install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

PyTorch packages are device-specific. Install the appropriate PyTorch build
before the editable install when the default package-index build is unsuitable
for the machine.

Validate the bundled reference results:

```bash
phiape-benchmarks check
```

Generate the Excel workbook, full-precision CSV summaries, LaTeX tables, and
four Wilcoxon--Holm diagrams:

```bash
phiape-benchmarks report
```

The default input is `reference_results/`, and generated reports are written to
`outputs/reports/`. The manifest is checked before analysis, including file
hashes, archive/dimension coverage, row counts, dataset counts, methods, and the
locked protocol. The statistical validator additionally checks schemas,
duplicate keys, all paired seeds, all 16 repeats, integer correct-prediction
counts, and exact archive coverage.

The reference data contain 25,280 seed-level rows:

| Archive | Dimension | Datasets | Methods | Repeats | Rows |
|---|---:|---:|---:|---:|---:|
| UEA | 32 | 30 | 5 | 16 | 2,400 |
| UEA | 128 | 30 | 5 | 16 | 2,400 |
| UCR | 32 | 128 | 5 | 16 | 10,240 |
| UCR | 128 | 128 | 5 | 16 | 10,240 |

## Run the experiments

Run one archive and embedding dimension:

```bash
phiape-benchmarks run --archive uea --dimension 32
```

Run all four configurations:

```bash
phiape-benchmarks run-all
```

Restrict a run to selected datasets, methods, or model seeds:

```bash
phiape-benchmarks run \
  --archive ucr \
  --dimension 128 \
  --datasets ECG200,Coffee \
  --methods No-APE,phiAPE \
  --seeds 2026,2027
```

Runs use CUDA when available and otherwise use CPU. Downloaded datasets default
to `datasets/` in the repository root, separated into `datasets/uea/` and
`datasets/ucr/`. New results default to `outputs/`. Both generated directories
are excluded from version control. Use `--data-root` and `--output-root` to
choose different locations. Execution is resume-safe at the dataset, method,
and repeat level; an existing configuration is rejected if its protocol or
recorded software environment is incompatible.

The loaders download a dataset only when it is absent from the selected data
root. A complete `run-all` performs 25,280 neural-network training runs and is
therefore intended for substantial compute resources. Use `--datasets`,
`--methods`, and `--seeds` for smaller verification runs.

Validate and report newly trained results with:

```bash
phiape-benchmarks check --input outputs
phiape-benchmarks report --input outputs --report-dir outputs/reports
```

## Experimental protocol

The comparison includes No-APE, Learned APE, Sin-APE, tAPE, and phiAPE at
embedding dimensions 32 and 128. Every dataset--method configuration has 16
paired repeats with model seeds 2026--2041. The train/validation split uses a
separate fixed seed of 2026. The exact published archive membership is locked
in source: 30 UEA datasets and 128 UCR datasets.

| Setting | Value |
|---|---:|
| Maximum epochs | 200 |
| Batch size | 64 |
| Early-stopping patience | 32 |
| Validation fraction | 0.20 |
| Maximum sequence length | 512 |
| Encoder layers | 3 |
| Attention heads | 8 |
| Feed-forward dimension | 256 |
| Dropout | 0.20 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Gradient clipping norm | 4.0 |

The input projection is scaled by $\sqrt{d}$, the best checkpoint is selected
by validation loss, and validation/test cross-entropy uses label smoothing
0.05. Dataset means are reconstructed from integer correct-prediction counts.
Each archive/dimension setting uses a five-method Friedman test followed by all
ten directional one-sided paired Wilcoxon signed-rank tests. Differences are
rounded to 12 decimals, `zero_method="wilcox"` is used, and Holm adjustment is
applied within each ten-test family.

## Repository layout

```text
assets/                 Figure used in this README
datasets/               Downloaded UEA/UCR datasets (generated and ignored)
outputs/                New runs and generated reports (generated and ignored)
reference_results/      Immutable seed-level publication results and manifest
src/phiape_benchmarks/  Model, data adapters, runner, statistics, and reporting
tests/                  Unit and integration tests
```

Run the test suite with:

```bash
pytest
```

## Reproducibility scope

The bundled reference results reproduce the reported aggregate values and
statistical analyses exactly. Fresh neural-network training is deterministic
within a compatible numerical stack, but PyTorch, CUDA, cuDNN, and GPU changes
can affect optimization and checkpoint selection despite identical seeds. Each
new run records its software and device environment so incompatible results
cannot be combined accidentally.

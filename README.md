# phiAPE

**Positional encoding for time-series Transformers**

Many real-world time series record only a few values at each time step.
A three-axis accelerometer measures three values, for example, and an ECG
from a single lead records one. A Transformer projects these inputs to an
embedding of a chosen width. With a 32-dimensional embedding, only 16
sine/cosine pairs encode position. If some barely change across the sequence,
those pairs contribute little information about time order.

**phiAPE** uses the known sequence length to allocate those frequencies.
The slowest pair spans half a cycle from the first sample to the last, while
faster pairs capture finer positional scales. It adds no trainable parameters.

- **Time-series researchers** can reproduce the WSDM 2027 results from
  35,392 archived UEA and UCR runs or retrain the benchmarks.
- **Transformer developers** can reuse the [additive encoding](src/phiape_benchmarks/core.py)
  or try [phiRoPE](src/phiape_benchmarks/rotary_attention.py) in rotary attention.
  phiRoPE applies the same rule at the per-head width and rotates projected
  queries and keys while leaving values unrotated.

<p align="center">
  <img src="assets/phiAPE_positional_similarity.png" width="70%" alt="Line plots and heatmaps comparing Sin-APE, tAPE, and phiAPE for sequence lengths 25 and 400 and embedding widths 32 and 128.">
</p>

*Figure 1 from the paper.* Positional similarity for Sin-APE, tAPE, and
phiAPE across the full signed range of position differences. Each heatmap
shows the same values as its line plot, using a common scale across methods
within that panel.

## What the paper found

In the five-method additive comparison, phiAPE achieved the best average rank
on both archives at widths 32 and 128. In the separate rotary comparison,
phiRoPE improved mean accuracy over RoPE in all four settings, with significant
gains at width 32 on both archives. Both comparisons use paired dataset means
over 16 seeds and two-sided Wilcoxon tests with Holm adjustment.

## Install and reproduce

From the repository root, use Python 3.12 and the tested
[requirements](requirements.txt):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .

phiape-benchmarks check
phiape-benchmarks report   # Table 1 and Figure 4
phiape-benchmarks table2   # RoPE versus phiRoPE
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate
it with `.venv\Scripts\Activate.ps1` in PowerShell. The reporting commands run
on CPU and write their outputs to `outputs/reports/`. The
[additive](reference_results/README.md) and [rotary](rotary_reference_results/README.md)
reference results include provenance and run configurations.

The original Raven training jobs used Python 3.9, PyTorch 1.13.0,
CUDA 11.4, and cuDNN 8.2.4. [requirements-hpc.txt](requirements-hpc.txt)
records the dependencies declared in the original job environments.
Fresh training scores can vary across hardware and software versions.

## Retrain

Start with one dataset and one seed to check your setup.

```bash
phiape-benchmarks run --rotary --archive ucr --dimension 32 --datasets ECG200 --seeds 2026
```

The full comparisons use 30 UEA and 128 UCR datasets at widths 32 and 128.
The [run protocol](src/phiape_benchmarks/protocol.py) records the shared
training settings. Runs use CUDA when available, save after each repeat,
and resume when you rerun the same command. Install `gdown` if the UEA
loader needs to download a dataset from Google Drive.

```bash
phiape-benchmarks run-all
phiape-benchmarks report --input outputs --report-dir outputs/retrained-additive

phiape-benchmarks run-all --rotary
phiape-benchmarks table2 --input outputs/rotary --report-dir outputs/retrained-rotary
```

## License

The repository software is licensed under the [MIT License](LICENSE).
The ACM paper is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

Sajjad Shumaly and Mahsa Salehi. 2027. *Rethinking Absolute Positional
Encoding in Transformers for Time-Series Modeling*. In Proceedings of the
20th ACM International Conference on Web Search and Data Mining (WSDM ’27),
Hong Kong, February 15–19, 2027.

```bibtex
@inproceedings{shumaly2027phiape,
  author    = {Shumaly, Sajjad and Salehi, Mahsa},
  title     = {Rethinking Absolute Positional Encoding in Transformers for Time-Series Modeling},
  booktitle = {Proceedings of the 20th ACM International Conference on Web Search and Data Mining},
  series    = {WSDM '27},
  year      = {2027},
  month     = feb,
  address   = {Hong Kong}
}
```

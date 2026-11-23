# Additive reference results

These files contain the 25,280 seed-level TST results for No-APE, Learned
APE, Sin-APE, tAPE, and phiAPE. They cover the 30 UEA and 128 UCR datasets
at embedding widths 32 and 128, with 16 paired repeats per setting.

[manifest.json](manifest.json) records the protocol, original software
environment, expected dataset and row counts, and SHA-256 hashes of both
the source result tables and the bundled files. The bundled CSVs preserve
the archived metric values, use the public method names, and include the
reconstructed integer `test_correct` count for every run.

From the repository root, run

```bash
phiape-benchmarks check
phiape-benchmarks report
```

The report reproduces Table 1 and Figure 4 using the manuscript's two-sided
paired Wilcoxon tests with Holm adjustment over ten pairs per archive and
embedding width. Table 2 is generated separately from the
[rotary reference results](../rotary_reference_results/README.md).

# Reference results

This directory contains the immutable seed-level results used for the
publication tables and statistical analyses. It covers UEA and UCR at embedding
dimensions 32 and 128, with five methods and 16 paired repeats per dataset.

`manifest.json` records the protocol, source environment, row and dataset
counts, hashes of the normalized files, and hashes of the source result tables.
The normalized CSVs preserve the archived metric strings, use the public method
names, and add the exact integer `test_correct` count for every run.

Validate the files and reproduce the publication reports from the repository
root:

```bash
phiape-benchmarks check --input reference_results
phiape-benchmarks report --input reference_results --report-dir outputs/reports
```

The reporting command never modifies these reference files. Generated reports
are excluded from version control.

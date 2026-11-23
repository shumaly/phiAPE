# RoPE and phiRoPE reference results

These 10,112 seed-level rows are the inputs to the standalone rotary
comparison in Table 2 of the WSDM paper. They were extracted from the four
original v28 TST experiment CSVs and retain both rotary methods for all
datasets at embedding widths 32 and 128.

| Archive | Datasets | Model widths | Paired repeats per width | Run rows |
|---|---:|---|---:|---:|
| UEA | 30 | 32 and 128 | 16 | 1,920 |
| UCR | 128 | 32 and 128 | 16 | 8,192 |

The files preserve the archived outcome values. The integer `test_correct`
column reconstructs the number of correct predictions from the reported
accuracy and test-set size. [manifest.json](manifest.json) records source
paths and SHA-256 hashes, expected coverage, and the rotary implementation
and experiment configuration hashes.

From the repository root, run `phiape-benchmarks table2` to validate the
inputs and generate the LaTeX and CSV table. The statistical analysis pairs
the methods by dataset after averaging repeats. The four hypotheses use one
Holm correction across both archives and widths.

# ConvTran downstream results

Saved outcomes for the controlled comparison of ConvTran with tAPE+eRPE
and phiAPE+eRPE on the 30 UEA datasets at embedding width 32. Each
condition has ten paired repeats, giving 600 run rows and 300 paired
repeat rows. The absolute positional frequency schedule is the changed
model component.

The archived method identifier RN-APE_c4 denotes phiAPE with c=4.
Column names beginning with rn_ape and rn_minus_tape refer to phiAPE
and phiAPE minus tAPE, respectively. Accuracies are stored as fractions.

| File | Contents |
|---|---|
| phiAPE_convtran_results.csv | All 600 individual training outcomes, including seeds, checkpoint epochs, accuracy, loss, and recorded runtime |
| phiAPE_convtran_repeat_pairs.csv | The 300 matched method pairs |
| phiAPE_convtran_dataset_effects.csv | Mean accuracies and paired effects for all 30 datasets |
| phiAPE_convtran_experiment_summary.csv | Per-dataset, per-method means and standard deviations |
| phiAPE_convtran_configuration.json | Recorded architecture and training settings |
| phiAPE_convtran_two_sided_statistics.json | Archive summary and statistical test |
| manifest.json | Coverage, method mapping, export notes, and SHA-256 file hashes |

These saved results support checking Figure 5 and the reported downstream
comparison. Average the ten repeats within each dataset before applying
the paired Wilcoxon signed-rank test to the 30 dataset means. The reported
test is two-sided, discards zero differences, and uses the tie-adjusted
normal approximation without continuity correction.

The unweighted mean gain is 0.422246 percentage points, with 20 wins,
one tie, and nine losses. The Wilcoxon statistic is 111 and the p-value
is 0.0212861. With a single method-pair comparison, Holm adjustment
leaves this p-value unchanged.

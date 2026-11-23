# README figures

`phiAPE_positional_similarity.png` is the numerical positional-similarity
figure shown in the project README. It compares Sin-APE, tAPE, and phiAPE
for two sequence lengths and two embedding widths.

## Planned overview illustration

**Target file:** `phiAPE_method_overview.png` (1600 × 900 pixels, 16:9).
Place it below “One frequency rule, two ways to encode position.”

### Illustration brief

Create a clean, wide scientific diagram for a research repository. Use a
white background, dark text, and two accessible accent colors. Use simple
geometric forms and generous spacing so labels remain legible when the
figure is displayed at 800 pixels wide.

Show one left-to-right flow with a fork at the end:

1. **Standard frequency grid.** Draw a horizontal frequency ruler with a
   handful of unevenly spaced markers, including several close to zero.
   Label the markers collectively “Sine/cosine pairs.”
2. **Phase-span remapping.** Show an arrow from the original ruler to a
   bounded frequency ruler. Keep the marker order and show a positive lower
   bound. Label the endpoints “π/(L−1)” and “π/(L−1) × (1 + width/4).”
   Beside the lower bound, show a small half-circle from start to end with
   the label “Half-cycle across the sequence.”
3. **Two branches.** The upper branch is labeled “phiAPE · additive” and
   “width = embedding dimension.” Show a position vector entering a plus
   node with the projected input. The lower branch is labeled
   “phiRoPE · rotary” and “width = per-head dimension.” Show separate
   circular arrows on Q and K, with V as a straight passthrough. Give the
   branches equal visual weight.

Keep the illustration schematic. Use only these technical labels and avoid
performance values or invented experimental plots. The existing numerical
figure shows the computed geometry of the encodings.

**Alt text:** A geometric frequency grid is remapped to a bounded interval.
The slowest pair covers half a cycle across the sequence. The result feeds
additive input encoding in phiAPE and per-head query/key rotation in phiRoPE.

**Caption:** Phase-span remapping sets the slowest pair to half a cycle over
the sequence. phiAPE uses the model width to encode inputs additively.
phiRoPE uses the head width to rotate queries and keys.

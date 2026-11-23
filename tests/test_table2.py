from pathlib import Path

import numpy as np
import pytest
from phiape_benchmarks.cli import main
from phiape_benchmarks.table2 import generate_table2


def test_reproduce_rotary_table(tmp_path):
    reference = Path(__file__).resolve().parents[1] / "rotary_reference_results"
    table = generate_table2(reference, tmp_path)
    assert list(zip(table.archive, table.dimension)) == [
        ("UEA", 32), ("UEA", 128), ("UCR", 32), ("UCR", 128)]
    np.testing.assert_array_equal(table[["wins", "ties", "losses"]], [
        [24, 1, 5], [19, 0, 11], [94, 3, 31], [71, 4, 53]])
    np.testing.assert_allclose(table.p_value_holm_four, [
        0.0014369143264602053, 0.0654417984187603,
        2.409210650673044e-08, 0.0859978929956618], rtol=1e-12, atol=0)
    assert list(table.calculation) == ["approx", "exact", "approx", "approx"]
    assert (tmp_path / "phiRoPE_transfer_summary.tex").is_file()


def test_mixed_method_selection_rejected_before_training():
    with pytest.raises(SystemExit):
        main(["run", "--rotary", "--archive", "uea", "--dimension", "32",
              "--methods", "phiAPE,phiRoPE"])

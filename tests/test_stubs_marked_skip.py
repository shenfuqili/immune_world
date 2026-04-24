"""Placeholder tests for pending implementations; all marked `skip("stub")`.

These exist so that the first unskip-and-implement cycle has a ready-made assertion site.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="stub — body pending")


def test_gene_embedding_shape() -> None:
    raise AssertionError("pending implementation (Sec. 4.2, Eq. 2)")


def test_trajectory_attention_causal_mask() -> None:
    raise AssertionError("pending implementation (Sec. 4.3, Eq. 4)")


def test_composite_loss_weighting() -> None:
    raise AssertionError("pending implementation (Sec. 4.6, Eq. 9)")


def test_pearson_r_on_identical_vectors_equals_one() -> None:
    raise AssertionError("pending implementation (Table 1)")


def test_cbdir_on_reversed_ordering_is_negative() -> None:
    raise AssertionError("pending implementation (Table 1)")


def test_training_smoke_loss_decreases() -> None:
    raise AssertionError("pending implementation (_smoke.yaml)")

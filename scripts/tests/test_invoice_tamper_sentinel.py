"""Tests for invoice_tamper_sentinel."""
import pytest
from scripts.invoice_tamper_sentinel import assert_invoice_total_matches


def test_invoice_tamper_sentinel():
    assert_invoice_total_matches([100, 250], 350)
    with pytest.raises(ValueError):
        assert_invoice_total_matches([100, 250], 100)

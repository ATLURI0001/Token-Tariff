"""Tests for negative_credit_sentinel."""
import pytest
from scripts.negative_credit_sentinel import assert_non_negative_credit


def test_negative_credit_sentinel():
    assert_non_negative_credit(0)
    assert_non_negative_credit(500)
    with pytest.raises(ValueError):
        assert_non_negative_credit(-1)

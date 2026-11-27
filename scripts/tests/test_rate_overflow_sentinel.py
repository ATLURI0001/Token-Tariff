"""Tests for rate_overflow_sentinel."""
import pytest
from scripts.rate_overflow_sentinel import assert_rate_within_bounds


def test_rate_overflow_sentinel():
    assert_rate_within_bounds(2.5)
    with pytest.raises(ValueError):
        assert_rate_within_bounds(5000.0)

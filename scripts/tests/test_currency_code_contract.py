"""Tests for currency_code_contract."""
import pytest
from scripts.currency_code_contract import validate_currency_code


def test_currency_code_contract():
    validate_currency_code("USD")
    with pytest.raises(ValueError):
        validate_currency_code("us")

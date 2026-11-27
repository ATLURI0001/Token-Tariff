"""Tests for tariff_tier_contract."""
import sys
sys.path.insert(0, "scripts")
from tariff_tier_contract import TariffTierContract


def test_tariff_tier_contract():
    c = TariffTierContract(["free", "pro"])
    assert c.validate("pro") is True
    assert c.validate("unknown") is False

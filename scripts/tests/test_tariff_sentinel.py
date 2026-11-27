import pytest
from scripts.tariff_sentinel import validate_tariff


def test_tariff_sentinel():
    validate_tariff({"tok_a": 0.01, "tok_b": 0.005})
    with pytest.raises(ValueError, match="negative rate"):
        validate_tariff({"tok_x": -1.0})

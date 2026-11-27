import pytest
from scripts.billing_event_sentinel import validate_billing_event


def test_billing_event_sentinel():
    validate_billing_event({"event_id": "ev-1", "token_type": "gpt4", "amount": 0.05})
    with pytest.raises(ValueError, match="missing fields"):
        validate_billing_event({"event_id": "ev-2"})

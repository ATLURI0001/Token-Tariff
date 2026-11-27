"""Token-Tariff: billing event sentinel – rejects malformed events."""


def validate_billing_event(event: dict) -> None:
    required = {"event_id", "token_type", "amount"}
    missing = required - event.keys()
    if missing:
        raise ValueError(f"billing event missing fields: {missing}")
    if event.get("amount", 0) < 0:
        raise ValueError("billing event amount cannot be negative")

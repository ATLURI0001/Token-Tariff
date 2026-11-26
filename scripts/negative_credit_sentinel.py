"""Negative credit balance sentinel."""


def assert_non_negative_credit(balance_cents: int) -> None:
    if balance_cents < 0:
        raise ValueError(f"sentinel violation: negative credit balance {balance_cents}")

"""Token tariff sentinel – rejects tariff configs with negative rates."""


def validate_tariff(config: dict) -> None:
    for token, rate in config.items():
        if rate < 0:
            raise ValueError(f"negative rate for {token}: {rate}")

"""Tariff rate overflow sentinel."""


def assert_rate_within_bounds(rate_per_1k: float, max_rate: float = 1000.0) -> None:
    if rate_per_1k < 0 or rate_per_1k > max_rate:
        raise ValueError(f"sentinel violation: rate {rate_per_1k} out of bounds [0, {max_rate}]")

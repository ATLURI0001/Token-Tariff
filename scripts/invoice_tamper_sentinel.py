"""Invoice total tampering sentinel."""


def assert_invoice_total_matches(line_items: list[int], total_cents: int) -> None:
    if sum(line_items) != total_cents:
        raise ValueError(f"sentinel violation: line items sum {sum(line_items)} != total {total_cents}")

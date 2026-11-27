"""ISO-4217 currency code contract validator."""
import re

_CURR_RE = re.compile(r"^[A-Z]{3}$")


def validate_currency_code(code: str) -> None:
    if not isinstance(code, str) or not _CURR_RE.match(code):
        raise ValueError(f"contract violation: invalid ISO-4217 currency code '{code}'")

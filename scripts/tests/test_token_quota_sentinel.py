"""Tests for token_quota_sentinel."""
import sys
sys.path.insert(0, "scripts")
from token_quota_sentinel import TokenQuotaSentinel


def test_token_quota_sentinel():
    s = TokenQuotaSentinel(limit=100)
    assert s.check(50) is True
    assert s.check(150) is False
    assert s.overruns == 1

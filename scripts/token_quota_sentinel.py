"""token_quota_sentinel.py – sentinel for token quota overruns."""


class TokenQuotaSentinel:
    def __init__(self, limit: int):
        self.limit = limit
        self.overruns = 0

    def check(self, tokens: int) -> bool:
        if tokens > self.limit:
            self.overruns += 1
            return False
        return True

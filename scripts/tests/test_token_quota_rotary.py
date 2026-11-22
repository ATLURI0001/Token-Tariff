import time
import unittest
import threading
from scripts.token_quota_rotary import (
    TokenQuotaRotary,
    ProviderQuotaConfig,
    QuotaExceededError,
    InvalidQuotaConfigError,
)


class TestTokenQuotaRotary(unittest.TestCase):

    def setUp(self):
        self.configs = [
            ProviderQuotaConfig(provider="openai", tpm_limit=50000, rpm_limit=100, tier="flagship", priority=1),
            ProviderQuotaConfig(provider="anthropic", tpm_limit=40000, rpm_limit=80, tier="flagship", priority=2),
            ProviderQuotaConfig(provider="google", tpm_limit=60000, rpm_limit=120, tier="economy", priority=1),
        ]
        self.rotary = TokenQuotaRotary(self.configs)

    def test_config_validation(self):
        with self.assertRaises(InvalidQuotaConfigError):
            TokenQuotaRotary([])

        with self.assertRaises(InvalidQuotaConfigError):
            ProviderQuotaConfig(provider="", tpm_limit=1000, rpm_limit=10)

        with self.assertRaises(InvalidQuotaConfigError):
            ProviderQuotaConfig(provider="openai", tpm_limit=0, rpm_limit=10)

        with self.assertRaises(InvalidQuotaConfigError):
            ProviderQuotaConfig(provider="openai", tpm_limit=1000, rpm_limit=-5)

    def test_successful_quota_reservation(self):
        prov, info = self.rotary.reserve_quota(requested_tokens=5000, preferred_provider="openai")
        self.assertEqual(prov, "openai")
        self.assertEqual(info["allocated_tokens"], 5000)
        self.assertFalse(info["is_failover"])
        self.assertEqual(info["remaining_tokens"], 45000.0)

    def test_automatic_failover_when_preferred_exhausted(self):
        # Exhaust openai tokens (50,000 available)
        self.rotary.reserve_quota(requested_tokens=48000, preferred_provider="openai")

        # Next request for 5000 tokens cannot fit in openai (only 2000 remaining)
        # Should automatically failover to Anthropic (priority 2 in flagship tier)
        prov, info = self.rotary.reserve_quota(
            requested_tokens=5000,
            preferred_provider="openai",
            tier="flagship"
        )
        self.assertEqual(prov, "anthropic")
        self.assertTrue(info["is_failover"])

        telemetry = self.rotary.get_telemetry()
        self.assertEqual(telemetry["failover_routes"], 1)

    def test_quota_exceeded_error_when_all_exhausted(self):
        # Attempt to reserve 200,000 tokens (greater than any single provider's capacity)
        with self.assertRaises(QuotaExceededError):
            self.rotary.reserve_quota(requested_tokens=200000)

        telemetry = self.rotary.get_telemetry()
        self.assertEqual(telemetry["throttled_requests"], 1)

    def test_continuous_replenishment(self):
        rotary = TokenQuotaRotary([
            ProviderQuotaConfig(provider="fast", tpm_limit=6000, rpm_limit=60)
        ])
        # 6000 TPM = 100 tokens / sec
        rotary.reserve_quota(requested_tokens=6000, preferred_provider="fast")
        self.assertFalse(rotary.check_capacity("fast", 100))

        # Sleep briefly (e.g. 50ms) and manually fast-forward bucket time
        bucket = rotary._buckets["fast"]
        bucket.last_updated -= 1.0  # simulate 1 second elapsed

        # After 1 simulated second, 100 tokens should be replenished
        self.assertTrue(rotary.check_capacity("fast", 50))

    def test_release_unspent_tokens(self):
        rotary = TokenQuotaRotary([
            ProviderQuotaConfig(provider="test", tpm_limit=10000, rpm_limit=100)
        ])
        rotary.reserve_quota(requested_tokens=5000, preferred_provider="test")
        self.assertEqual(rotary._buckets["test"].available_tokens, 5000.0)

        # Release 2000 unused tokens
        rotary.release_unspent_tokens("test", 2000)
        self.assertEqual(rotary._buckets["test"].available_tokens, 7000.0)

    def test_concurrent_reservations(self):
        rotary = TokenQuotaRotary([
            ProviderQuotaConfig(provider="pool", tpm_limit=100000, rpm_limit=500)
        ])
        threads = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    rotary.reserve_quota(requested_tokens=100, preferred_provider="pool")
            except Exception as e:
                errors.append(e)

        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        telemetry = rotary.get_telemetry()
        self.assertEqual(telemetry["successful_reservations"], 100)
        self.assertEqual(telemetry["total_tokens_allocated"], 10000)


if __name__ == "__main__":
    unittest.main()

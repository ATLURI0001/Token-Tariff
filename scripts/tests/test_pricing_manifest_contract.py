"""Unit tests for Model Pricing Manifest Contract and Billing Unit Sanitizer."""

import unittest
from scripts.pricing_manifest_contract import (
    validate_and_sanitize_pricing_manifest,
)


class TestPricingManifestContract(unittest.TestCase):
    def test_valid_manifest_list_structure(self) -> None:
        raw_manifest = [
            {
                "model_id": "openai/gpt-4o",
                "input_cost_per_m": 5.0,
                "output_cost_per_m": 15.0,
                "cached_input_cost_per_m": 2.5,
                "currency": "USD",
                "effective_date": "2026-01-01",
            },
            {
                "model_id": "anthropic/claude-3-5-sonnet",
                "input_cost_per_m": 3.0,
                "output_cost_per_m": 15.0,
                "currency": "usd",  # should be normalized to uppercase
            },
        ]

        result = validate_and_sanitize_pricing_manifest(raw_manifest)
        self.assertTrue(result.valid, f"Expected valid, got violations: {result.violations}")
        self.assertEqual(len(result.sanitized_entries), 2)

        entry1 = result.sanitized_entries[0]
        self.assertEqual(entry1.provider, "openai")
        self.assertEqual(entry1.model_name, "gpt-4o")
        self.assertEqual(entry1.input_cost_per_m, 5.0)
        self.assertEqual(entry1.output_cost_per_m, 15.0)
        self.assertEqual(entry1.cached_input_cost_per_m, 2.5)
        self.assertEqual(entry1.currency, "USD")
        self.assertEqual(entry1.effective_date, "2026-01-01")

        entry2 = result.sanitized_entries[1]
        self.assertEqual(entry2.currency, "USD")
        self.assertIsNone(entry2.cached_input_cost_per_m)

    def test_valid_manifest_dict_structure(self) -> None:
        raw_dict = {
            "google/gemini-1.5-pro": {
                "input_cost_per_m": 3.5,
                "output_cost_per_m": 10.5,
                "currency": "EUR",
            }
        }
        result = validate_and_sanitize_pricing_manifest(raw_dict)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.sanitized_entries), 1)
        self.assertEqual(result.sanitized_entries[0].model_id, "google/gemini-1.5-pro")

    def test_invalid_model_id_rejections(self) -> None:
        # 1. Missing slash provider/model separator
        raw = [{"model_id": "gpt-4o", "input_cost_per_m": 1.0, "output_cost_per_m": 2.0}]
        res = validate_and_sanitize_pricing_manifest(raw)
        self.assertFalse(res.valid)
        self.assertTrue(any(v.rule == "invalid_format" for v in res.violations))

        # 2. Empty model_id
        raw_empty = [{"model_id": "  ", "input_cost_per_m": 1.0, "output_cost_per_m": 2.0}]
        res_empty = validate_and_sanitize_pricing_manifest(raw_empty)
        self.assertFalse(res_empty.valid)

    def test_currency_and_date_validation(self) -> None:
        # Invalid currency
        raw_curr = [{"model_id": "meta/llama-3", "input_cost_per_m": 0.5, "output_cost_per_m": 0.5, "currency": "XYZ"}]
        res_curr = validate_and_sanitize_pricing_manifest(raw_curr)
        self.assertFalse(res_curr.valid)
        self.assertTrue(any(v.rule == "invalid_currency" for v in res_curr.violations))

        # Invalid date format
        raw_date = [{"model_id": "meta/llama-3", "input_cost_per_m": 0.5, "output_cost_per_m": 0.5, "effective_date": "11/22/2026"}]
        res_date = validate_and_sanitize_pricing_manifest(raw_date)
        self.assertFalse(res_date.valid)
        self.assertTrue(any(v.rule == "invalid_date_format" for v in res_date.violations))

    def test_pricing_bounds_and_numeric_checks(self) -> None:
        # Negative cost
        raw_neg = [{"model_id": "meta/llama-3", "input_cost_per_m": -1.0, "output_cost_per_m": 2.0}]
        self.assertFalse(validate_and_sanitize_pricing_manifest(raw_neg).valid)

        # Exceeds max boundary (10,000)
        raw_huge = [{"model_id": "meta/llama-3", "input_cost_per_m": 15000.0, "output_cost_per_m": 2.0}]
        self.assertFalse(validate_and_sanitize_pricing_manifest(raw_huge).valid)

        # NaN / Inf cost
        raw_nan = [{"model_id": "meta/llama-3", "input_cost_per_m": float("nan"), "output_cost_per_m": 2.0}]
        self.assertFalse(validate_and_sanitize_pricing_manifest(raw_nan).valid)

    def test_prototype_pollution_detection(self) -> None:
        raw_pollution = [{"model_id": "meta/llama-3", "input_cost_per_m": 1.0, "output_cost_per_m": 2.0, "__proto__": {"injected": True}}]
        res = validate_and_sanitize_pricing_manifest(raw_pollution)
        self.assertFalse(res.valid)
        self.assertTrue(any(v.rule == "prototype_pollution" for v in res.violations))


if __name__ == "__main__":
    unittest.main()

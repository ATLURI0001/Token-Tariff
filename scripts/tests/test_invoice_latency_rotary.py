"""Tests for InvoiceLatencyRotary."""
from scripts.invoice_latency_rotary import InvoiceLatencyRotary


def test_invoice_latency_rotary():
    r = InvoiceLatencyRotary(capacity=2)
    r.record(10.0)
    r.record(30.0)
    r.record(50.0)
    assert r.avg_ms() == 40.0

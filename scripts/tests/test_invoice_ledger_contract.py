from scripts.invoice_ledger_contract import (
    InvoiceLedgerContract,
    InvoiceLineItem,
    InvoiceRecord,
)


def test_valid_invoice_ledger():
    inv = InvoiceRecord(
        invoice_id="INV-2026-001",
        currency="USD",
        line_items=[
            InvoiceLineItem("GPT-4o Input Tokens", 10, 500),
            InvoiceLineItem("GPT-4o Output Tokens", 5, 1500),
        ],
        declared_total_cents=12500,
    )
    res = InvoiceLedgerContract.validate(inv)
    assert res.valid is True
    assert res.computed_total_cents == 12500


def test_ledger_mismatch_and_invalid_currency():
    inv = InvoiceRecord(
        invoice_id="INV-2026-002",
        currency="XYZ",
        line_items=[InvoiceLineItem("Tokens", -1, 100)],
        declared_total_cents=999,
    )
    res = InvoiceLedgerContract.validate(inv)
    assert res.valid is False
    assert any("ERR_UNSUPPORTED_CURRENCY" in e for e in res.errors)
    assert any("ERR_INVALID_QUANTITY" in e for e in res.errors)
    assert any("ERR_LEDGER_TOTAL_MISMATCH" in e for e in res.errors)

"""Billing Invoice Double-Entry Ledger & Rounding Invariant Contract.

Enforces strict financial invariants: non-negative line items, exact subtotal
summation matching invoice totals within floating-point epsilon, and ISO-4217 currency bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class InvoiceLineItem:
    description: str
    quantity: int
    unit_price_cents: int

    @property
    def total_cents(self) -> int:
        return self.quantity * self.unit_price_cents


@dataclass(frozen=True)
class InvoiceRecord:
    invoice_id: str
    currency: str
    line_items: List[InvoiceLineItem]
    declared_total_cents: int


@dataclass
class LedgerContractResult:
    valid: bool
    computed_total_cents: int
    errors: List[str] = field(default_factory=list)


class InvoiceLedgerContract:
    ALLOWED_CURRENCIES = {"USD", "EUR", "GBP", "JPY"}

    @classmethod
    def validate(cls, invoice: InvoiceRecord) -> LedgerContractResult:
        errors: List[str] = []

        if not invoice.invoice_id:
            errors.append("ERR_MISSING_INVOICE_ID")

        if invoice.currency not in cls.ALLOWED_CURRENCIES:
            errors.append(f"ERR_UNSUPPORTED_CURRENCY: {invoice.currency}")

        if not invoice.line_items:
            errors.append("ERR_EMPTY_LINE_ITEMS")

        computed = 0
        for idx, item in enumerate(invoice.line_items):
            if item.quantity <= 0:
                errors.append(f"ERR_INVALID_QUANTITY: line {idx} quantity={item.quantity}")
            if item.unit_price_cents < 0:
                errors.append(f"ERR_NEGATIVE_UNIT_PRICE: line {idx} price={item.unit_price_cents}")
            computed += item.total_cents

        if computed != invoice.declared_total_cents:
            errors.append(
                f"ERR_LEDGER_TOTAL_MISMATCH: computed={computed} != declared={invoice.declared_total_cents}"
            )

        return LedgerContractResult(
            valid=len(errors) == 0,
            computed_total_cents=computed,
            errors=errors,
        )

"""Conservative local topic screening; this is a heuristic, not a compliance verdict."""
import re


FINANCE_TERMS = (
    "finance", "financial", "accounting", "accountant", "audit", "auditing",
    "invoice", "invoices", "revenue", "expense", "expenses", "payroll",
    "budget", "ledger", "tax", "taxation", "treasury", "liquidity",
    "receivable", "receivables", "payable", "payables", "reconciliation",
    "balance sheet", "cash flow", "credit risk", "market risk", "bank statement",
    "banking", "payment", "payments", "transaction", "transactions",
    "procurement", "asset", "assets", "liabilities", "equity", "debit", "credit",
)
FRAMEWORK_TERMS = ("frmc", "sox", "coso", "icfr", "sarbanes oxley",
                   "financial risk management", "internal control over financial reporting")
CONTROL_TERMS = ("control", "controls", "risk", "compliance", "approval",
                 "approved", "evidence", "policy", "policies", "testing",
                 "monitoring", "governance", "segregation of duties",
                 "material weakness", "deficiency", "assessment")


def check_finance_content(text):
    normalized = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    def matches(terms):
        return [term for term in terms if re.search(r"\b" + re.escape(term) + r"\b", normalized)]
    finance = matches(FINANCE_TERMS)
    framework = matches(FRAMEWORK_TERMS)
    controls = matches(CONTROL_TERMS)
    signals = finance + framework + controls
    # Distinct corroborating terms prevent a generic 'risk' or 'approval' from passing.
    relevant = (bool(framework) and len(signals) >= 2) or len(finance) >= 2 or (bool(finance) and bool(controls))
    meaningful_words = re.findall(r"\b[a-z]{3,}\b", normalized)
    signal_words = sum(len(re.findall(r"\b" + re.escape(term) + r"\b", normalized))
                       * len(term.split()) for term in signals)
    relevant = relevant and signal_words / max(len(meaningful_words), 1) >= 0.02
    return {
        "accepted": relevant,
        "category": "FRMC / financial controls" if framework or (finance and controls) else "Finance",
        "matched_terms": signals,
        "reason": ("Finance/FRMC content detected." if relevant else
                   "Only finance/FRMC-related documents are accepted. The extracted text does not contain enough financial or financial-control context."),
    }


def require_finance_content(text, filename):
    if not text.strip():
        raise ValueError(f"{filename}: no readable text. Scanned documents require OCR before upload.")
    result = check_finance_content(text)
    if not result["accepted"]:
        raise ValueError(f"{filename}: {result['reason']}")
    return result

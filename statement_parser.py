"""Deterministic bank-statement parser.

Reads the text layer of a statement PDF with pdfplumber, finds the transaction
table by its header, turns every row into a transaction and proves the result
with the running balance. No model involved; returns a coverage figure so the
caller can decide whether an LLM fallback is worth it.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Iterable

import pdfplumber
from dateutil import parser as dateparser

# --- header vocabulary --------------------------------------------------------

ROLE_KEYWORDS = {
    "date": ["tran date", "txn date", "transaction date", "value date", "posting date", "trans date", "date"],
    "debit": ["withdrawal", "withdrawals", "debit", "debits", "paid out", "money out", "dr", "payments"],
    "credit": ["deposit", "deposits", "credit", "credits", "paid in", "money in", "cr", "receipts"],
    "balance": ["closing balance", "running balance", "available balance", "balance", "bal"],
    "amount": ["transaction amount", "amount", "amt"],
    "desc": ["particulars", "description", "narration", "narrative", "details", "remarks", "memo", "payee", "transaction"],
    "ref": ["chq", "cheque", "ref", "reference", "utr", "instrument"],
    "valuedate": ["value date", "value dt", "val date"],
    "ignore": ["init br", "branch", "s no", "sl no", "sr no", "serial"],
}
NUMERIC_ROLES = {"debit", "credit", "amount", "balance"}
DATE_ROLES = {"date", "valuedate"}
SKIP_ROW = re.compile(r"^(opening|closing)\s+balance|^balance\s+(b/?f|c/?f|brought|carried)|^total\b|^grand total|^statement summary|^page \d+", re.I)
DATE_RE = re.compile(r"\b(\d{1,2}[-/. ]\d{1,2}[-/. ]\d{2,4}|\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}[-/ ]?[A-Za-z]{3,9}[-/ ]?,?\s?\d{2,4})\b")
AMOUNT_RE = re.compile(r"^\(?-?(?:[₹$£€]|rs\.?|inr)?\s*\d[\d,.]*(?:[.,]\d{1,2})?\)?\s*(?:dr|cr)?\.?$", re.I)
SUFFIX_WORD = re.compile(r"^(dr|cr)\.?$", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9/ ]+", " ", (s or "").lower()).strip()


def header_role(cell: str) -> str | None:
    n = _norm(cell)
    if not n:
        return None
    # exact matches first (avoid "cr" inside "description")
    for role, words in ROLE_KEYWORDS.items():
        if n in words:
            return role
    for role, words in ROLE_KEYWORDS.items():
        for w in words:
            if len(w) > 2 and re.search(rf"(^|\s){re.escape(w)}(\s|$)", n):
                return role
    return None


def parse_amount(text: str, signed: bool = False) -> tuple[Decimal | None, str | None]:
    """Returns (amount, 'dr'|'cr'|None from a suffix/sign). Absolute unless
    `signed` (balances keep their sign; a Dr suffix means negative)."""
    if text is None:
        return None, None
    t = text.strip().replace(" ", " ")
    if not t or t in {"-", "--", "—"}:
        return None, None
    if not AMOUNT_RE.match(t):
        return None, None
    suffix = None
    m = re.search(r"(dr|cr)\.?$", t, re.I)
    if m:
        suffix = m.group(1).lower()
        t = t[: m.start()].strip()
    negative = t.startswith("(") and t.endswith(")") or t.startswith("-")
    t = re.sub(r"[()₹$£€\s-]|rs\.?|inr", "", t, flags=re.I)
    if re.search(r",\d{2}$", t) and (t.count(".") >= 1 or t.count(",") == 1 and len(t.split(",")[0].replace(".", "")) <= 3 or "." in t):
        t = t.replace(".", "").replace(",", ".")  # 1.234,56
    elif re.search(r",\d{2}$", t) and "." not in t:
        t = t.replace(",", ".")  # 1234,56
    else:
        t = t.replace(",", "")
    try:
        value = Decimal(t)
    except InvalidOperation:
        return None, None
    if negative and not suffix:
        suffix = "dr"
    if signed:
        return (-abs(value) if (negative or suffix == "dr") else abs(value)), suffix
    return abs(value), suffix


def parse_date(text: str) -> str | None:
    if not text:
        return None
    m = DATE_RE.search(text)
    if not m:
        return None
    raw = m.group(1)
    try:
        dayfirst = not re.match(r"^\d{4}[-/]", raw)
        d = dateparser.parse(raw, dayfirst=dayfirst, fuzzy=False)
        return d.date().isoformat()
    except (ValueError, OverflowError):
        return None


# --- rails, merchants, categories ---------------------------------------------

RAIL_TOKENS = {"upi", "p2a", "p2m", "p2p", "imps", "neft", "rtgs", "inb", "ift", "ib", "mb", "cr", "dr", "tparty", "trans", "tparty trans", "pos", "ecom", "vps", "atm", "cash", "nfs", "tfr", "trf", "to", "from", "by", "payment", "pmt", "int", "pd"}
BANK_WORDS = re.compile(r"\b(bank|hdfc|icici|sbi|axis|kotak|pnb|punjabnationalbank|dbs|yes|idfc|paytm|canara|boi|bob|indusind|federal|ltd|limited|india)\b", re.I)


def _looks_like_code(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return True
    if t.lower() in RAIL_TOKENS:
        return True
    if "@" in t:
        return True
    if re.fullmatch(r"[Xx*]{2,}\d*|\d[\d*Xx]{4,}", t):
        return True
    if re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", t):  # IFSC
        return True
    if re.fullmatch(r"[A-Za-z]{2,8}\d{3,}[A-Za-z0-9]*", t):  # bank reference codes: AXISN123456, HDFCN…, SBIN…
        return True
    if sum(ch.isdigit() for ch in t) >= max(4, len(t) // 2):
        return True
    if re.fullmatch(r"[\d\-/:. ]+", t):
        return True
    if BANK_WORDS.fullmatch(t):
        return True
    return False


def describe(description: str, txn_type: str) -> tuple[str, str, str]:
    """→ (merchant, paymentMethod, category) from the raw narration."""
    d = " ".join((description or "").split())
    low = d.lower()
    method = "bank_transfer"
    merchant = ""
    if low.startswith("atm") or "atm-cash" in low or "atm cash" in low or "atmwdl" in low or "cash wdl" in low:
        return "ATM cash withdrawal", "cash", "other"
    if re.search(r"\bint\.?\s*pd\b|interest", low):
        return "Bank interest", "bank_transfer", "other"
    if low.startswith("upi") or "/upi/" in low or "upi-" in low:
        method = "bank_transfer"  # the app's payment-method vocabulary has no UPI id
    elif low.startswith(("imps", "neft", "rtgs", "inb", "ib ", "mb ", "ift")):
        method = "bank_transfer"
    elif low.startswith(("pos", "ecom", "vps", "pcd", "pur/")) or re.search(r"\b(pos|ecom)\b", low):
        method = "debit_card"
    tokens = [t for t in re.split(r"[/\-|]+", d) if t.strip()]
    if len(tokens) <= 1:  # "POS 512967XXXXXX1234 AMAZON PAY INDIA" — space separated
        words = [w for w in d.split() if not _looks_like_code(w)]
        tokens = [" ".join(words)] if words else tokens
    cands = [t.strip() for t in tokens if not _looks_like_code(t.strip())]
    if cands:
        merchant = cands[0]
    else:
        merchant = re.sub(r"[\d/\-:]+", " ", d).strip()[:40] or "Unknown"
    merchant = BANK_WORDS.sub("", merchant).strip(" -/.") or merchant
    merchant = merchant[:48]
    return merchant, method, categorise(merchant + " " + d, txn_type)


CATEGORY_RULES = [
    ("food_dining", r"swiggy|zomato|restaurant|cafe|coffee|domino|pizza|kfc|mcdonald|burger|dhaba|eat|food|bakery|starbucks|subway"),
    ("shopping", r"amazon|flipkart|myntra|ajio|meesho|nykaa|dmart|bigbasket|blinkit|zepto|instamart|grocer|supermarket|reliance|croma|decathlon|ikea|mall|store|mart"),
    ("transportation", r"uber|ola|rapido|irctc|metro|petrol|fuel|hpcl|bpcl|iocl|indian oil|parking|toll|fastag|redbus|bus|cab|taxi"),
    ("bills_utilities", r"electric|bses|tata power|adani|airtel|jio|vodafone|\bvi\b|broadband|act fibernet|bill ?desk|billdesk|water|gas|lpg|indane|mobile recharge|recharge|dth|tata sky|postpaid|insurance|lic|premium|rent|society|maintenance|emi|loan"),
    ("entertainment", r"netflix|spotify|prime video|hotstar|disney|youtube|bookmyshow|pvr|inox|cinema|game|steam|playstation|xbox|apple\.com|itunes"),
    ("healthcare", r"pharm|apollo|medplus|hospital|clinic|doctor|lab|diagnostic|1mg|pharmeasy|netmeds|dental|health"),
    ("travel", r"makemytrip|goibibo|cleartrip|airline|indigo|air india|vistara|spicejet|hotel|oyo|airbnb|booking\.com|agoda|yatra|ixigo|flight|railway"),
]


def categorise(text: str, txn_type: str) -> str:
    low = text.lower()
    for cat, pattern in CATEGORY_RULES:
        if re.search(pattern, low):
            return cat
    return "other"


# --- table extraction ---------------------------------------------------------

@dataclass
class Row:
    date: str | None
    desc: str
    debit: Decimal | None
    credit: Decimal | None
    amount: Decimal | None
    amount_suffix: str | None
    balance: Decimal | None
    page: int


@dataclass
class ParseResult:
    transactions: list[dict] = field(default_factory=list)
    pages: int = 0
    header_found: bool = False
    rows_seen: int = 0
    rows_parsed: int = 0
    balance_checked: int = 0
    balance_ok: int = 0
    warnings: list[str] = field(default_factory=list)
    text: str = ""

    @property
    def coverage(self) -> float:
        if not self.rows_seen:
            return 0.0
        return self.rows_parsed / self.rows_seen


def _roles_from_cells(cells: list[str]) -> dict[int, str] | None:
    roles: dict[int, str] = {}
    for i, c in enumerate(cells):
        r = header_role(c or "")
        if r in ("ignore", "valuedate"):
            continue
        if r and r not in roles.values():
            roles[i] = r
    have = set(roles.values())
    if "date" in have and ({"debit", "credit", "amount"} & have):
        return roles
    return None


def _rows_from_ruled_tables(page, page_no: int, roles_prev: dict[int, str] | None) -> tuple[list[Row], dict[int, str] | None, int]:
    rows: list[Row] = []
    roles = None
    seen = 0
    try:
        tables = page.extract_tables({"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 4, "intersection_tolerance": 6})
    except Exception:
        tables = []
    for table in tables:
        local_roles = None
        for r in table:
            cells = [(c or "").replace("\n", " ").strip() for c in r]
            if local_roles is None:
                local_roles = _roles_from_cells(cells)
                if local_roles:
                    roles = local_roles
                    continue
                if roles_prev and len(cells) >= len(roles_prev) and any(cells):
                    local_roles = roles_prev  # header only on the first page
                    roles = local_roles
                else:
                    continue
            if not any(cells):
                continue
            row = _row_from_cells(cells, local_roles, page_no)
            if row:
                if row.date is not None:
                    seen += 1
                rows.append(row)
    return rows, roles, seen


def _row_from_cells(cells: list[str], roles: dict[int, str], page_no: int) -> Row | None:
    get = lambda role: next((cells[i] for i, r in roles.items() if r == role and i < len(cells)), "")
    desc = get("desc")
    date = parse_date(get("date"))
    debit, _ = parse_amount(get("debit"))
    credit, _ = parse_amount(get("credit"))
    amount, suffix = parse_amount(get("amount"))
    balance, _ = parse_amount(get("balance"), signed=True)
    if not desc:
        # description may live in an unlabelled column
        desc = " ".join(c for i, c in enumerate(cells) if i not in roles and c)
    if SKIP_ROW.search(desc.strip()) or SKIP_ROW.search(get("date").strip()):
        return Row(None, desc, None, None, None, None, balance, page_no)
    if date is None and debit is None and credit is None and amount is None:
        return None
    return Row(date, desc, debit, credit, amount, suffix, balance, page_no)


def _lines(words, tol=3.0):
    lines = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1]["top"] - w["top"]) <= tol:
            lines[-1]["words"].append(w)
        else:
            lines.append({"top": w["top"], "bottom": w["bottom"], "words": [w]})
    for ln in lines:
        ln["words"].sort(key=lambda w: w["x0"])
    return lines


def _merge_cells(words, gap=7.0):
    cells = []
    for w in words:
        if cells and w["x0"] - cells[-1]["x1"] <= gap:
            cells[-1]["text"] += " " + w["text"]
            cells[-1]["x1"] = w["x1"]
        else:
            cells.append({"text": w["text"], "x0": w["x0"], "x1": w["x1"]})
    return cells


def _header_bands(line_words):
    """Group a header line's words into labelled cells, matching multi-word
    phrases from the vocabulary first ("Withdrawal Amt.", "Closing Balance")."""
    words = sorted(line_words, key=lambda w: w["x0"])
    cells = []
    i = 0
    while i < len(words):
        placed = False
        for k in (3, 2, 1):
            if i + k > len(words):
                continue
            chunk = words[i:i + k]
            text = " ".join(w["text"] for w in chunk)
            role = header_role(text)
            if role and (k == 1 or _norm(text) in {p for ws in ROLE_KEYWORDS.values() for p in ws}):
                cells.append({"role": role, "text": text, "x0": chunk[0]["x0"], "x1": chunk[-1]["x1"]})
                i += k
                placed = True
                break
        if not placed:
            w = words[i]
            if cells and w["x0"] - cells[-1]["x1"] <= 7 and cells[-1]["role"] is None:
                cells[-1]["text"] += " " + w["text"]
                cells[-1]["x1"] = w["x1"]
            else:
                cells.append({"role": None, "text": w["text"], "x0": w["x0"], "x1": w["x1"]})
            i += 1
    roles = [c["role"] for c in cells if c["role"] and c["role"] not in ("ignore",)]
    if "date" in roles and ({"debit", "credit", "amount"} & set(roles)):
        seen = set()
        for c in cells:
            if c["role"] in seen and c["role"] not in ("ignore", None):
                c["role"] = None  # keep the first column of a repeated role
            seen.add(c["role"])
            c["cx"] = (c["x0"] + c["x1"]) / 2
        return cells
    return None


def _assign(bands, line_words):
    """Map a data line's words onto header bands."""
    numeric = [i for i, b in enumerate(bands) if b["role"] in NUMERIC_ROLES]
    dates = [i for i, b in enumerate(bands) if b["role"] in DATE_ROLES]
    textual = [i for i, b in enumerate(bands) if b["role"] not in NUMERIC_ROLES and b["role"] not in DATE_ROLES and b["role"] != "ignore"]
    first_numeric_x0 = min((bands[i]["x0"] for i in numeric), default=10**6)

    def edge_distance(i, w):
        b = bands[i]
        x0, x1 = b["x0"], b["x1"]
        if b["role"] == "desc":
            x1 = max(x1, first_numeric_x0 - 4)
        cx = (w["x0"] + w["x1"]) / 2
        if x0 <= cx <= x1:
            return 0
        return min(abs(cx - x0), abs(cx - x1))

    def right_distance(i, w):
        b = bands[i]
        return min(abs(w["x1"] - b["x1"]), abs((w["x0"] + w["x1"]) / 2 - b["cx"]))

    out = {}
    prev_band = None
    prev_amount = False
    for w in sorted(line_words, key=lambda w: w["x0"]):
        t = w["text"]
        is_date = bool(DATE_RE.fullmatch(t))
        is_amount = bool(AMOUNT_RE.match(t)) and not is_date
        if is_date and dates:
            band = min(dates, key=lambda i: edge_distance(i, w))
        elif is_amount and numeric and (w["x1"] >= first_numeric_x0 - 30):
            band = min(numeric, key=lambda i: right_distance(i, w))
        elif SUFFIX_WORD.match(t) and prev_amount and prev_band is not None:
            band = prev_band
        elif textual:
            inside = [i for i in textual if bands[i]["role"] != "desc" and bands[i]["x0"] - 2 <= (w["x0"] + w["x1"]) / 2 <= bands[i]["x1"] + 2]
            band = inside[0] if inside else min(textual, key=lambda i: edge_distance(i, w))
        else:
            band = min(range(len(bands)), key=lambda i: edge_distance(i, w))
        out.setdefault(band, []).append(t)
        prev_band, prev_amount = band, is_amount
    return out


def _rows_from_words(page, page_no: int, bands_prev) -> tuple[list[Row], object, int]:
    words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False)
    lines = _lines(words)
    bands = bands_prev
    rows: list[Row] = []
    seen = 0
    pending_text: list[str] = []
    last_row: Row | None = None
    last_bottom = None
    start = 0
    found_here = None
    for i, ln in enumerate(lines[: (len(lines) if bands is None else 8)]):
        cells = _header_bands(ln["words"])
        if cells:
            found_here = cells
            start = i + 1
            break
    if found_here:
        bands = found_here
    if bands is None:
        return rows, None, 0
    roles = {i: b["role"] for i, b in enumerate(bands) if b["role"] and b["role"] not in ("ignore", "valuedate")}

    for ln in lines[start:]:
        by_band = _assign(bands, ln["words"])
        cells = [" ".join(by_band.get(i, [])) for i in range(len(bands))]
        text_all = " ".join(c for c in cells if c).strip()
        if not text_all:
            continue
        if _header_bands(ln["words"]):
            continue  # a repeated header further down the page
        if SKIP_ROW.search(text_all):
            bal_idx = next((i for i, r in roles.items() if r == "balance"), None)
            bal, _ = parse_amount(cells[bal_idx], signed=True) if bal_idx is not None else (None, None)
            if bal is not None:
                rows.append(Row(None, text_all, None, None, None, None, bal, page_no))
            continue
        row = _row_from_cells(cells, roles, page_no)
        has_amount = row is not None and (row.debit is not None or row.credit is not None or row.amount is not None or row.balance is not None)
        if row is not None and has_amount:
            seen += 1
            if pending_text:
                row.desc = (" ".join(pending_text) + " " + row.desc).strip()
                pending_text = []
            rows.append(row)
            last_row = row
            last_bottom = ln["bottom"]
        else:
            gap = (ln["top"] - last_bottom) if last_bottom is not None else 99
            if last_row is not None and gap < 6:
                last_row.desc = (last_row.desc + " " + text_all).strip()
                last_bottom = ln["bottom"]
            else:
                pending_text.append(text_all)
                if len(pending_text) > 3:
                    pending_text = pending_text[-3:]
    return rows, bands, seen


def parse_statement(pdf_bytes: bytes, password: str | None = None) -> ParseResult:
    res = ParseResult()
    with pdfplumber.open(io.BytesIO(pdf_bytes), password=password or "") as pdf:
        res.pages = len(pdf.pages)
        all_rows: list[Row] = []
        roles_prev = None
        bands_prev = None
        texts = []
        for pno, page in enumerate(pdf.pages, start=1):
            texts.append(page.extract_text() or "")
            rows, roles, seen = _rows_from_ruled_tables(page, pno, roles_prev)
            if roles:
                roles_prev = roles
            if not rows:
                rows, bands, seen = _rows_from_words(page, pno, bands_prev)
                if bands:
                    bands_prev = bands
            res.rows_seen += seen
            all_rows.extend(rows)
        res.text = "\n\n".join(texts)
    res.header_found = roles_prev is not None or bands_prev is not None
    res.transactions = _finalise(all_rows, res)
    return res


def _finalise(rows: list[Row], res: ParseResult) -> list[dict]:
    txns: list[dict] = []
    prev_balance: Decimal | None = None
    swap_votes = 0
    keep_votes = 0
    # first pass: decide debit/credit orientation from the balance
    for r in rows:
        if r.date is None and r.balance is not None and r.debit is None and r.credit is None and r.amount is None:
            prev_balance = r.balance
            continue
        if r.date is None:
            continue
        if r.balance is not None and prev_balance is not None:
            debit = r.debit or Decimal(0)
            credit = r.credit or Decimal(0)
            if r.amount is not None and r.debit is None and r.credit is None:
                pass
            elif abs((prev_balance - debit + credit) - r.balance) <= Decimal("0.011"):
                keep_votes += 1
            elif abs((prev_balance + debit - credit) - r.balance) <= Decimal("0.011"):
                swap_votes += 1
        if r.balance is not None:
            prev_balance = r.balance
    swap = swap_votes > keep_votes
    if swap:
        res.warnings.append("debit/credit columns were swapped by the balance check")
    prev_balance = None
    for r in rows:
        if r.date is None and r.balance is not None and r.debit is None and r.credit is None and r.amount is None:
            prev_balance = r.balance
            continue
        if r.date is None:
            continue
        debit, credit = (r.credit, r.debit) if swap else (r.debit, r.credit)
        txn_type = None
        amount = None
        if debit is not None and debit > 0:
            txn_type, amount = "debit", debit
        elif credit is not None and credit > 0:
            txn_type, amount = "credit", credit
        elif r.amount is not None:
            amount = r.amount
            if r.balance is not None and prev_balance is not None:
                delta = r.balance - prev_balance
                txn_type = "credit" if delta > 0 else "debit"
            elif r.amount_suffix:
                txn_type = "credit" if r.amount_suffix == "cr" else "debit"
            else:
                txn_type = "debit"
        if amount is None or amount == 0:
            if r.balance is not None:
                prev_balance = r.balance
            continue
        verified = None
        if r.balance is not None and prev_balance is not None:
            expected = prev_balance + (amount if txn_type == "credit" else -amount)
            verified = abs(expected - r.balance) <= Decimal("0.011")
            res.balance_checked += 1
            if verified:
                res.balance_ok += 1
        if r.balance is not None:
            prev_balance = r.balance
        merchant, method, category = describe(r.desc, txn_type)
        confidence = 0.97 if verified else (0.5 if verified is False else 0.8)
        txns.append({
            "amount": float(amount),
            "date": r.date,
            "merchant": merchant or "Unknown",
            "category": category,
            "paymentMethod": method,
            "description": " ".join(r.desc.split())[:200],
            "transactionType": txn_type,
            "confidence": confidence,
            "balanceVerified": verified,
            "page": r.page,
        })
        res.rows_parsed += 1
    return txns


# --- CSV ----------------------------------------------------------------------

def parse_csv(text: str) -> ParseResult:
    import csv

    res = ParseResult()
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text.lstrip("﻿")), dialect)
    rows_raw = [r for r in reader if any(c.strip() for c in r)]
    if not rows_raw:
        return res
    header_idx = None
    roles = None
    for i, r in enumerate(rows_raw[:10]):
        roles = _roles_from_cells(r)
        if roles:
            header_idx = i
            break
    if roles is None:
        res.warnings.append("no header row with date + amount columns")
        return res
    res.header_found = True
    rows: list[Row] = []
    for r in rows_raw[header_idx + 1:]:
        res.rows_seen += 1
        row = _row_from_cells([c.strip() for c in r], roles, 1)
        if row:
            rows.append(row)
    res.transactions = _finalise(rows, res)
    return res


# --- LLM fallback (extracted text → expense-ai-proxy) -------------------------

STATEMENT_PROMPT = """Extract every transaction from this bank statement text.
Return ONLY a JSON object {"transactions": [...]} where each item has:
amount (positive number), date (YYYY-MM-DD), merchant, description,
transactionType ("debit" or "credit" — use the Debit/Withdrawal vs Credit/Deposit
column, or the running balance direction), paymentMethod ("bank_transfer",
"debit_card", "credit_card", "cash" or "other"), category (one of food_dining,
transportation, shopping, entertainment, bills_utilities, healthcare, travel,
other) and confidence (0-1). Skip opening/closing balance and total rows."""


def llm_fallback(text: str, proxy_url: str, proxy_key: str, timeout: float = 120.0) -> list[dict]:
    import json

    import httpx

    body = {
        "model": "gpt-5-nano",
        "messages": [
            {"role": "system", "content": STATEMENT_PROMPT},
            {"role": "user", "content": text[:120_000]},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 30_000,
    }
    r = httpx.post(
        proxy_url.rstrip("/") + "/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {proxy_key}", "apikey": proxy_key},
        timeout=timeout,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    data = json.loads(content)
    items = data.get("transactions") if isinstance(data, dict) else data
    out = []
    for it in items or []:
        try:
            amount = float(it.get("amount") or 0)
        except (TypeError, ValueError):
            continue
        if amount <= 0:
            continue
        out.append({
            "amount": amount,
            "date": parse_date(str(it.get("date") or "")) or it.get("date"),
            "merchant": (it.get("merchant") or "Unknown")[:48],
            "category": it.get("category") if it.get("category") in {c for c, _ in CATEGORY_RULES} | {"other"} else "other",
            "paymentMethod": it.get("paymentMethod") if it.get("paymentMethod") in {"bank_transfer", "debit_card", "credit_card", "cash", "other"} else "other",
            "description": (it.get("description") or "")[:200],
            "transactionType": "credit" if it.get("transactionType") == "credit" else "debit",
            "confidence": min(max(float(it.get("confidence") or 0.6), 0), 1),
            "balanceVerified": None,
            "page": None,
        })
    return out

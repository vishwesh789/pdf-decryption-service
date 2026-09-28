"""Deterministic bank-statement parser for worldwide layouts.

pdfplumber reads the text layer; a multilingual header vocabulary names the
columns; numbers and dates are parsed with the locale detected from the
document itself; every row becomes a transaction and the running balance
proves amounts and debit/credit wherever a balance column exists. Layouts
without one (US sectioned statements, credit cards, signed single-amount
columns) get their direction from the sign, a type column, the section
heading or the statement kind. No model involved; `coverage` tells the
caller whether an LLM fallback is worth it.
"""
from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

import pdfplumber
from dateutil import parser as dateparser

# --- header vocabulary (english variants + de/fr/es/pt/it/nl) -------------------

ROLE_KEYWORDS = {
    "date": ["tran date", "txn date", "transaction date", "trans date", "posting date", "post date", "posted date",
             "date posted", "booking date", "date", "buchungstag", "buchungsdatum", "buchung", "datum", "fecha",
             "fecha operacion", "data", "data operazione", "data movimento", "date operation", "date d operation",
             "boekdatum", "transactiedatum", "rentedatum", "dato"],
    "debit": ["withdrawal", "withdrawals", "withdrawal amt", "withdrawal amount", "debit", "debits", "debit amount",
              "paid out", "money out", "payments", "payments and withdrawals", "withdrawals and debits", "charges",
              "purchases", "dr", "out", "outgoing", "soll", "belastung", "lastschrift", "ausgang", "ausgaben",
              "debit eur", "cargo", "cargos", "retiro", "retiros", "debito", "debitos", "saida", "saidas",
              "uscite", "addebiti", "af", "afschrijving", "afschrijvingen", "udbetaling", "uttag"],
    "credit": ["deposit", "deposits", "deposit amt", "deposit amount", "credit", "credits", "credit amount",
               "paid in", "money in", "receipts", "deposits and credits", "deposits and additions", "in", "incoming",
               "haben", "gutschrift", "eingang", "einnahmen", "credit eur", "abono", "abonos", "ingreso", "ingresos",
               "deposito", "credito", "creditos", "entrada", "entradas", "entrate", "accrediti", "bij",
               "bijschrijving", "bijschrijvingen", "indbetaling", "insattning"],
    "balance": ["closing balance", "running balance", "available balance", "balance", "bal", "new balance",
                "saldo", "solde", "kontostand", "ledger balance", "balance amount"],
    "amount": ["transaction amount", "amount", "amt", "betrag", "umsatz", "montant", "importe", "valor", "importo",
               "bedrag", "belob", "belopp", "amount eur", "amount usd", "amount gbp"],
    "desc": ["particulars", "description", "narration", "narrative", "details", "transaction details",
             "details of transaction", "transaction description", "remarks", "memo", "payee", "transaction",
             "activity", "verwendungszweck", "buchungstext", "umsatzdetails", "beschreibung", "libelle",
             "libelle operation", "designation", "concepto", "descripcion", "descricao", "historico", "descrizione",
             "causale", "omschrijving", "tekst", "text"],
    "type": ["type", "dr/cr", "cr/dr", "d/c", "debit/credit", "transaction type", "art"],
    "ref": ["chq", "cheque", "cheque no", "ref", "ref no", "reference", "utr", "instrument", "check", "check no",
            "cheque/reference"],
    "valuedate": ["value date", "value dt", "val date", "valuta", "wertstellung", "date de valeur", "fecha valor",
                  "data valuta", "valutadatum"],
    "ignore": ["init br", "branch", "s no", "sl no", "sr no", "serial", "page", "no"],
}
NUMERIC_ROLES = {"debit", "credit", "amount", "balance"}
DATE_ROLES = {"date", "valuedate"}
KEYWORD_PHRASES = {p for ws in ROLE_KEYWORDS.values() for p in ws}

SKIP_ROW = re.compile(
    r"^(opening|closing|beginning|ending|previous|new)\s+balance|^balance\s+(b/?f|c/?f|brought|carried|forward)|"
    r"^(sub)?total\b|^grand total|^statement summary|^page \d+|^anfangssaldo|^endsaldo|^saldo (anterior|final|inicial)|"
    r"^solde (initial|final|precedent)|^continued|^carried forward", re.I)
DATE_TOKEN = re.compile(
    r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/ ]?[A-Za-zÀ-ÿ]{3,10}\.?[-/ ]?,?\s?\d{2,4}|"
    r"[A-Za-z]{3,9}\.? \d{1,2},? \d{4}|\d{1,2}/\d{1,2}(?![/.\d]))")
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    "mär": 3, "maerz": 3, "marz": 3, "mai": 5, "okt": 10, "dez": 12,  # de
    "janv": 1, "fevr": 2, "févr": 2, "mars": 3, "avr": 4, "juin": 6, "juil": 7, "août": 8, "aout": 8, "déc": 12,  # fr
    "ene": 1, "abr": 4, "ago": 8, "dic": 12,  # es
    "fev": 2, "set": 9, "out": 10,  # pt
    "gen": 1, "mag": 5, "giu": 6, "lug": 7, "ott": 10,  # it
    "mrt": 3, "mei": 5,  # nl
}
CURRENCY_RE = re.compile(r"(?:[₹$£€¥]|rs\.?|inr|usd|eur|gbp|chf|aud|cad|sgd|aed|r\$|a\$|c\$|s\$|kr\.?|zł|zar)", re.I)
AMOUNT_CORE = re.compile(r"^[-−+]?\(?\s*\d[\d.,'   ]*\)?[-−]?$")
SUFFIX_WORD = re.compile(r"^(dr|cr|d|c|db|debit|credit)\.?$", re.I)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _norm(s: str) -> str:
    s = _strip_accents((s or "").lower())
    s = CURRENCY_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9/ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def header_role(cell: str) -> str | None:
    n = _norm(cell)
    if not n:
        return None
    for role, words in ROLE_KEYWORDS.items():
        if n in words:
            return role
    for role, words in ROLE_KEYWORDS.items():
        for w in words:
            if len(w) > 2 and re.search(rf"(^|\s){re.escape(w)}(\s|$)", n):
                return role
    return None


# --- locale detection ---------------------------------------------------------

@dataclass
class Locale:
    decimal: str = "."      # "." or ","
    dayfirst: bool = True
    credit_card: bool = False
    year: int | None = None  # the statement's year, for dates printed without one


def detect_locale(text: str) -> Locale:
    loc = Locale()
    numeric_text = DATE_TOKEN.sub(" ", text)
    comma_dec = len(re.findall(r"(?<![\d.])\d{1,3}(?:[. '\u00a0\u202f]\d{3})*,\d{2}(?![\d,])", numeric_text))
    dot_dec = len(re.findall(r"(?<![\d,])\d{1,3}(?:[, '\u00a0\u202f]\d{2,3})*\.\d{2}(?![\d.])", numeric_text))
    loc.decimal = "," if comma_dec > dot_dec else "."
    first_gt12 = second_gt12 = 0
    for m in re.finditer(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b", text):
        a, b = int(m.group(1)), int(m.group(2))
        if a > 12:
            first_gt12 += 1
        if b > 12:
            second_gt12 += 1
    if first_gt12 and not second_gt12:
        loc.dayfirst = True
    elif second_gt12 and not first_gt12:
        loc.dayfirst = False
    else:
        low = text.lower()
        us_hint = ("$" in text and not re.search(r"\b(a\$|c\$|s\$|aud|cad|sgd|nzd)\b", low)) or " usd" in low
        loc.dayfirst = not us_hint
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
    if years:
        loc.year = max(set(years), key=years.count)
    low = text.lower()
    loc.credit_card = bool(re.search(r"credit card statement|minimum payment|statement balance|credit limit|"
                                     r"payment due date|kreditkarte|carte de credit|tarjeta de credito", _strip_accents(low)))
    return loc


# --- amounts and dates ----------------------------------------------------------

def is_amount_like(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    t = re.sub(r"\s*(dr|cr|db|d|c)\.?$", "", t, flags=re.I)
    t = CURRENCY_RE.sub("", t).strip()
    return bool(AMOUNT_CORE.match(t)) and any(ch.isdigit() for ch in t) and not DATE_TOKEN.fullmatch(text.strip() or "x")


def parse_amount(text: str, locale: Locale | None = None, signed: bool = False) -> tuple[Decimal | None, str | None]:
    """→ (amount, 'dr'|'cr'|None). Absolute unless `signed` (balances keep
    their sign; a trailing/leading minus, parentheses or a Dr suffix mean
    negative)."""
    if text is None:
        return None, None
    t = text.strip().replace(" ", " ").replace(" ", " ")
    if not t or t in {"-", "--", "—", "–"}:
        return None, None
    suffix = None
    m = re.search(r"\s*(dr|cr|db|d|c|debit|credit)\.?$", t, re.I)
    if m and len(t) > len(m.group(0)):
        w = m.group(1).lower()
        suffix = "cr" if w in ("cr", "c", "credit") else "dr"
        t = t[: m.start()].strip()
    t = CURRENCY_RE.sub("", t).strip()
    if not AMOUNT_CORE.match(t):
        return None, None
    negative = t.startswith(("-", "−")) or t.endswith(("-", "−")) or (t.startswith("(") and t.endswith(")"))
    t = re.sub(r"[()−\-+\s']", "", t)
    if not t or not any(ch.isdigit() for ch in t):
        return None, None
    dec = (locale.decimal if locale else None)
    if dec is None:
        # decide from the token itself
        if re.search(r",\d{1,2}$", t) and not re.search(r"\.\d{1,2}$", t):
            dec = ","
        else:
            dec = "."
    if dec == ",":
        if re.search(r",\d{1,2}$", t):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(".", "").replace(",", "")  # "1.234" thousands only
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


def _translate_month(token: str) -> str:
    def repl(m):
        w = m.group(0)
        key = _strip_accents(w.lower()).rstrip(".")
        for k in (key, key[:4], key[:3]):
            if k in MONTHS:
                return ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"][MONTHS[k] - 1]
        return w
    return re.sub(r"[A-Za-zÀ-ÿ]{3,10}", repl, token)


def parse_date(text: str, locale: Locale | None = None, year_hint: int | None = None) -> str | None:
    if not text:
        return None
    m = DATE_TOKEN.search(text)
    if not m:
        return None
    raw = m.group(1).strip()
    dayfirst = locale.dayfirst if locale else True
    if re.match(r"^\d{4}[-/.]", raw):
        dayfirst = False
    raw = _translate_month(raw)
    try:
        if re.fullmatch(r"\d{1,2}/\d{1,2}", raw):  # "01/05" (US statements omit the year)
            a, b = (int(x) for x in raw.split("/"))
            d, mo = (a, b) if dayfirst else (b, a)
            y = year_hint or (locale.year if locale else None) or __import__("datetime").date.today().year
            return __import__("datetime").date(y, mo, d).isoformat()
        d = dateparser.parse(raw, dayfirst=dayfirst, fuzzy=False)
        return d.date().isoformat()
    except (ValueError, OverflowError, TypeError):
        return None


# --- merchants and categories ---------------------------------------------------

NOISE = re.compile(
    r"\b(upi|p2a|p2m|p2p|imps|neft|rtgs|inb|ift|ib|mb|tparty|trans|pos|ecom|vps|atm|cash|nfs|tfr|trf|to|from|by|"
    r"payment|pmt|purchase|card|debit|credit|visa|mastercard|maestro|amex|contactless|chip|pin|online|mobile|"
    r"direct debit|dd|standing order|so|faster payments?|fp|fpi|fpo|bacs|chaps|sepa|lastschrift|ueberweisung|"
    r"uberweisung|kartenzahlung|girocard|dauerauftrag|gutschrift|belastung|prelevement|virement|cb|carte|"
    r"paiement|transferencia|pago|cargo|abono|pagamento|bonifico|addebito|ach|wire|check|cheque|chq|ref|"
    r"reference|txn|transaction|id|no|nr|paypal \*|pp\*|sq \*|tst\*|apple pay|google pay|gpay|phonepe|"
    r"received|sent|thank you|autopay|recurring|international|domestic|intl|fee|charges?|us|uk|in|de|fr|es|it|nl|"
    r"a favore di|in favore di|favore di|beneficiario|zugunsten|an|fur|fuer|au profit de|a favor de|mandat|mandate|"
    r"glaubiger|glaeubiger|end to end|e2e|iban|bic|ordre|order|on|facture|du|recu|prlv|vir|zelle|girocard|receipt|"
    r"receipts|paid|deposit|withdrawal|transfer|trsf|xfer|bill payment|pmnt|pymt|purch|pur)\b\.?",
    re.I)
DATE_FRAGMENT = re.compile(r"\b(on\s+)?\d{1,2}[ /.-](?:\d{1,2}|[A-Za-z]{3,9})(?:[ /.-]\d{2,4})?\b", re.I)
BANK_WORDS = re.compile(r"\b(bank|hdfc|icici|sbi|axis|kotak|pnb|punjabnationalbank|dbs|yes|idfc|paytm|canara|boi|bob|"
                        r"indusind|federal|ltd|limited|india|plc|llc|inc|gmbh|ag|sa|sarl|bv|nv|co)\b\.?", re.I)


def _looks_like_code(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return True
    if "@" in t:
        return True
    if re.fullmatch(r"[Xx*#]{2,}\d*|\d[\d*Xx#]{4,}|#\d+", t):
        return True
    if re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", t):  # IFSC
        return True
    if re.fullmatch(r"[A-Za-z]{2,8}\d{3,}[A-Za-z0-9]*", t):  # bank reference codes
        return True
    if re.fullmatch(r"[\d\-/:. ]+", t) or DATE_TOKEN.fullmatch(t):
        return True
    if sum(ch.isdigit() for ch in t) >= max(4, len(t) // 2):
        return True
    if not re.search(r"[A-Za-zÀ-ÿ]{2}", t):
        return True
    return False


def describe(description: str, txn_type: str) -> tuple[str, str, str]:
    """→ (merchant, paymentMethod, category) from the raw narration."""
    d = " ".join((description or "").split())
    low = _strip_accents(d.lower())
    if re.search(r"\batm\b|cash withdrawal|cash wdl|atmwdl|geldautomat|retrait|retiro efectivo|prelievo|geldopname", low):
        return "ATM cash withdrawal", "cash", "other"
    if re.search(r"\bint\.?\s*pd\b|\binterest\b|zinsen|interets|intereses|juros|interessi|rente\b", low):
        return "Bank interest", "bank_transfer", "other"
    method = "bank_transfer"
    if re.search(r"\b(pos|ecom|card|visa|mastercard|maestro|contactless|kartenzahlung|cb|carte|debit card|purchase)\b", low):
        method = "debit_card"
    if re.search(r"credit card|kreditkarte|carte de credit|tarjeta", low):
        method = "credit_card"
    if re.search(r"\bcash\b|bargeld|especes|efectivo", low) and "cashback" not in low:
        method = "cash"
    d_clean = d if d.lower().startswith(("upi", "imps", "neft")) else DATE_FRAGMENT.sub(" ", d)
    tokens = [t for t in re.split(r"[/|*]+|\s[-–—]\s|(?<=\D)-(?=\D)", d_clean) if t and t.strip()]
    if len(tokens) <= 1:
        tokens = [d_clean]
    cands = []
    for tok in tokens:
        words = [w for w in tok.split() if not _looks_like_code(w)]
        cleaned = NOISE.sub(" ", " ".join(words))
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -/.,:")
        if cleaned and re.search(r"[A-Za-zÀ-ÿ]{2}", cleaned) and not _looks_like_code(cleaned):
            cands.append(cleaned)
    merchant = cands[0] if cands else (re.sub(r"[\d/\-:]+", " ", d).strip()[:40] or "Unknown")
    merchant = BANK_WORDS.sub("", merchant).strip(" -/.,")
    merchant = re.sub(r"\s+", " ", merchant)[:48] or "Unknown"
    return merchant, method, categorise(merchant + " " + d)


CATEGORY_RULES = [
    ("food_dining", r"swiggy|zomato|restaurant|ristorante|cafe|caffe|coffee|domino|pizza|kfc|mcdonald|burger|dhaba|"
                    r"bakery|starbucks|subway|chipotle|nando|pret |greggs|costa|deliveroo|doordash|grubhub|uber eats|"
                    r"just eat|takeaway|dunkin|wendy|taco bell|panera|bistro|brasserie|sushi|kebab|diner|pub\b"),
    ("shopping", r"amazon|flipkart|myntra|ajio|meesho|nykaa|dmart|bigbasket|blinkit|zepto|instamart|grocer|supermarket|"
                 r"walmart|target|costco|tesco|sainsbury|asda|aldi|lidl|morrisons|waitrose|carrefour|rewe|edeka|kaufland|"
                 r"mercadona|coles|woolworths|kroger|safeway|whole foods|trader joe|ikea|decathlon|zara|h&m|uniqlo|nike|"
                 r"apple store|best buy|croma|reliance|ebay|etsy|shein|temu|mall|store|mart|boutique|primark|argos|"
                 r"home depot|lowe|b&q|marks|spencer|dm |rossmann|mediamarkt|saturn|fnac|el corte"),
    ("transportation", r"uber|ola|rapido|lyft|bolt|grab|irctc|metro|petrol|fuel|gas station|hpcl|bpcl|iocl|indian oil|"
                       r"shell|bp |exxon|chevron|esso|texaco|total energies|aral|parking|toll|fastag|redbus|bus|cab|taxi|"
                       r"transit|tfl|national rail|trainline|amtrak|sncf|deutsche bahn|db bahn|renfe|trenitalia|ns.nl|"
                       r"citymapper|lime|tier|zipcar|hertz|avis|enterprise"),
    ("bills_utilities", r"electric|bses|tata power|adani|airtel|jio|vodafone|\bvi\b|broadband|fibernet|billdesk|water|"
                        r"gas bill|lpg|indane|recharge|dth|tata sky|postpaid|insurance|lic |premium|rent|society|"
                        r"maintenance|emi|loan|mortgage|verizon|at&t|t-mobile|comcast|xfinity|spectrum|british gas|edf|"
                        r"e.on|octopus energy|ovo|thames water|council tax|o2|ee |three|sky |bt |virgin media|telekom|"
                        r"o2 germany|stadtwerke|vattenfall|orange|free mobile|movistar|iberdrola|endesa|enel|tim |wind|"
                        r"kpn|ziggo|utility|utilities|phone|internet|hydro|pg&e|con edison|duke energy|geico|allstate|"
                        r"state farm|progressive|aviva|axa|allianz"),
    ("entertainment", r"netflix|spotify|prime video|hotstar|disney|hulu|hbo|max\b|paramount|peacock|youtube|apple tv|"
                      r"bookmyshow|pvr|inox|cinema|cinemark|amc |odeon|vue |game|steam|playstation|xbox|nintendo|"
                      r"itunes|apple\.com/bill|twitch|patreon|audible|kindle|ticketmaster|eventbrite|concert|theatre|"
                      r"theater|museum|zoo|gym|fitness|peloton|planet fitness|puregym"),
    ("healthcare", r"pharm|apollo|medplus|hospital|clinic|doctor|dr\.|lab |diagnostic|1mg|pharmeasy|netmeds|dental|"
                   r"dentist|health|cvs|walgreens|rite aid|boots|lloyds pharmacy|apotheke|farmacia|pharmacie|optic|"
                   r"vision|medical|nhs|kaiser|aetna|cigna|blue cross"),
    ("travel", r"makemytrip|goibibo|cleartrip|airline|airways|indigo|air india|vistara|spicejet|hotel|oyo|airbnb|"
               r"booking\.com|agoda|expedia|hotels\.com|yatra|ixigo|flight|railway|ryanair|easyjet|lufthansa|"
               r"british airways|klm|air france|delta|united|american air|southwest|jetblue|emirates|qatar|marriott|"
               r"hilton|hyatt|ihg|accor|holiday inn|premier inn|travelodge|vrbo|trivago|kayak|skyscanner|eurostar"),
]


def _bounded(pattern: str) -> str:
    parts = []
    for alt in pattern.split("|"):
        core = alt.strip()
        if re.fullmatch(r"[a-z&' .]+", core) and len(core.replace(" ", "")) <= 5:
            parts.append(rf"\b{re.escape(core.strip())}\b")
        else:
            parts.append(alt)
    return "|".join(parts)


_CATEGORY_COMPILED = [(cat, re.compile(_bounded(pattern))) for cat, pattern in CATEGORY_RULES]


def categorise(text: str) -> str:
    low = _strip_accents(text.lower())
    for cat, rx in _CATEGORY_COMPILED:
        if rx.search(low):
            return cat
    return "other"


# --- rows ------------------------------------------------------------------------

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
    type_hint: str | None = None      # from a Type column or the section heading
    raw_amount: str = ""


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
    locale: Locale = field(default_factory=Locale)

    @property
    def coverage(self) -> float:
        if not self.rows_seen:
            return 0.0
        return self.rows_parsed / self.rows_seen


SECTION_CREDIT = re.compile(r"\b(deposits?|additions|credits|payments? received|money in|paid in|incoming|receipts|"
                            r"gutschriften?|eingange|abonos|ingresos|entradas|accrediti|bijschrijvingen?)\b", re.I)
SECTION_DEBIT = re.compile(r"\b(withdrawals?|debits|purchases?|payments?(?! received)|fees|charges|money out|paid out|"
                           r"outgoing|spending|belastungen?|lastschriften?|ausgange|cargos|retiros|saidas|addebiti|"
                           r"afschrijvingen?)\b", re.I)


def section_type(text: str) -> str | None:
    t = _strip_accents(text.lower())
    if len(t.split()) > 7 or re.search(r"statement|balance|limit|due|total|kontoauszug|releve", t):
        return None
    if SECTION_CREDIT.search(t) and not SECTION_DEBIT.search(t):
        return "credit"
    if SECTION_DEBIT.search(t) and not SECTION_CREDIT.search(t):
        return "debit"
    if SECTION_DEBIT.search(t) and SECTION_CREDIT.search(t):
        return None
    return None


def type_from_cell(text: str) -> str | None:
    t = _strip_accents((text or "").strip().lower())
    if t in ("dr", "d", "db", "debit", "withdrawal", "purchase", "payment", "soll", "belastung", "cargo", "debito", "af"):
        return "debit"
    if t in ("cr", "c", "credit", "deposit", "haben", "gutschrift", "abono", "credito", "bij"):
        return "credit"
    return None


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


def _row_from_cells(cells: list[str], roles: dict[int, str], page_no: int, loc: Locale, section: str | None) -> Row | None:
    get = lambda role: next((cells[i] for i, r in roles.items() if r == role and i < len(cells)), "")
    desc = get("desc")
    date = parse_date(get("date"), loc)
    debit, _ = parse_amount(get("debit"), loc)
    credit, _ = parse_amount(get("credit"), loc)
    amount, suffix = parse_amount(get("amount"), loc)
    balance, _ = parse_amount(get("balance"), loc, signed=True)
    if not desc:
        desc = " ".join(c for i, c in enumerate(cells) if i not in roles and c)
    if SKIP_ROW.search(desc.strip()) or SKIP_ROW.search(get("date").strip()):
        return Row(None, desc, None, None, None, None, balance, page_no)
    if date is None and debit is None and credit is None and amount is None:
        return None
    hint = type_from_cell(get("type")) or section
    return Row(date, desc, debit, credit, amount, suffix, balance, page_no, hint, get("amount"))


# --- ruled tables ---------------------------------------------------------------

def _line_above(page, top: float) -> str:
    words = [w for w in page.extract_words(x_tolerance=2, y_tolerance=3) if w["bottom"] <= top - 1 and w["bottom"] >= top - 40]
    if not words:
        return ""
    last_top = max(w["top"] for w in words)
    return " ".join(w["text"] for w in sorted(words, key=lambda w: w["x0"]) if abs(w["top"] - last_top) <= 3)


def _rows_from_ruled_tables(page, page_no, roles_prev, loc, section_prev):
    rows: list[Row] = []
    roles = None
    seen = 0
    section = section_prev
    try:
        found = page.find_tables({"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 4, "intersection_tolerance": 6})
    except Exception:
        found = []
    for tbl in found:
        try:
            table = tbl.extract()
        except Exception:
            continue
        heading = _line_above(page, tbl.bbox[1])
        if heading and section_type(heading):
            section = section_type(heading)
        local_roles = None
        for r in table:
            cells = [(c or "").replace("\n", " ").strip() for c in r]
            if local_roles is None:
                local_roles = _roles_from_cells(cells)
                if local_roles:
                    roles = local_roles
                    continue
                if roles_prev and len(cells) >= len(roles_prev) and any(cells):
                    local_roles = roles_prev
                    roles = local_roles
                else:
                    continue
            if not any(cells):
                continue
            row = _row_from_cells(cells, local_roles, page_no, loc, section if not ({"debit", "credit"} & set(local_roles.values())) else None)
            if row:
                if row.date is not None:
                    seen += 1
                rows.append(row)
    return rows, roles, seen, section


# --- unruled tables (words → columns) ---------------------------------------------

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


def _header_bands(line_words):
    words = sorted(line_words, key=lambda w: w["x0"])
    cells = []
    i = 0
    while i < len(words):
        placed = False
        for k in (4, 3, 2, 1):
            if i + k > len(words):
                continue
            chunk = words[i:i + k]
            text = " ".join(w["text"] for w in chunk)
            role = header_role(text)
            if role and (k == 1 or _norm(text) in KEYWORD_PHRASES):
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
    if any(is_amount_like(w["text"]) or DATE_TOKEN.fullmatch(w["text"]) for w in words):
        return None
    labelled = sum(1 for c in cells if c["role"])
    if "date" in roles and ({"debit", "credit", "amount"} & set(roles)) and len(cells) <= 12 and labelled * 2 >= len(cells):
        seen = set()
        for c in cells:
            if c["role"] in seen and c["role"] not in ("ignore", None):
                c["role"] = None
            seen.add(c["role"])
            c["cx"] = (c["x0"] + c["x1"]) / 2
        return cells
    return None


def _merge_fragments(words):
    """Join word runs that only make sense together: "02 Jan 2025",
    "Jan 5, 2025", "5 janv. 2025", "1 234,56"."""
    out = []
    i = 0
    while i < len(words):
        merged = None
        for k in (3, 2):
            if i + k > len(words):
                continue
            chunk = words[i:i + k]
            if any(chunk[j + 1]["x0"] - chunk[j]["x1"] > 6 for j in range(k - 1)):
                continue
            text = " ".join(w["text"] for w in chunk)
            if DATE_TOKEN.fullmatch(text) or re.fullmatch(r"[-−(]?\d{1,3}(?: \d{3})+(?:[.,]\d{2})?\)?", text):
                merged = {"text": text, "x0": chunk[0]["x0"], "x1": chunk[-1]["x1"], "top": chunk[0]["top"], "bottom": chunk[0]["bottom"]}
                i += k
                break
        if merged:
            out.append(merged)
        else:
            out.append(words[i])
            i += 1
    return out


def _numeric_clusters(bands, lines):
    """Where the amounts actually sit: cluster the centres of amount-like
    words in the numeric zone, then map each cluster to the nearest numeric
    header. Returns [(centre, band_index)]."""
    numeric = [i for i, b in enumerate(bands) if b["role"] in NUMERIC_ROLES]
    if not numeric:
        return []
    zone_x0 = min(bands[i]["x0"] for i in numeric) - 30
    centres = []
    for ln in lines:
        for w in _merge_fragments(sorted(ln["words"], key=lambda w: w["x0"])):
            cx = (w["x0"] + w["x1"]) / 2
            if cx >= zone_x0 and is_amount_like(w["text"]) and not DATE_TOKEN.fullmatch(w["text"]):
                centres.append(cx)
    centres.sort()
    clusters = []
    for c in centres:
        if clusters and c - clusters[-1][-1] <= 14:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    kept = [sum(cl) / len(cl) for cl in clusters if not (len(cl) < 2 and len(centres) > 6)]
    if len(kept) == len(numeric):
        return list(zip(kept, numeric))  # one cluster per column, left to right
    out = []
    for centre in kept:
        # values are usually right-aligned under a left-aligned header, so the
        # header's right edge is the better anchor
        band = min(numeric, key=lambda i: min(abs(bands[i]["cx"] - centre), abs(bands[i]["x1"] + 8 - centre)))
        out.append((centre, band))
    return out


def _assign(bands, line_words, clusters=None):
    """Map a data line's words onto header bands. Column ranges run from the
    midpoint between neighbouring header cells; amount-like words go to
    numeric columns by their centre, dates to date columns, everything else
    to the textual columns (so a wide narration never spills into the
    amount column just because its header is narrow)."""
    n = len(bands)
    ranges = []
    for i, b in enumerate(bands):
        left = -1e9 if i == 0 else (bands[i - 1]["x1"] + b["x0"]) / 2
        right = 1e9 if i == n - 1 else (b["x1"] + bands[i + 1]["x0"]) / 2
        ranges.append((left, right))
    numeric = [i for i, b in enumerate(bands) if b["role"] in NUMERIC_ROLES]
    dates = [i for i, b in enumerate(bands) if b["role"] in DATE_ROLES]
    textual = [i for i, b in enumerate(bands) if b["role"] not in NUMERIC_ROLES and b["role"] not in DATE_ROLES and b["role"] != "ignore"]

    def in_range(i, cx):
        return ranges[i][0] <= cx <= ranges[i][1]

    def nearest(cands, cx):
        return min(cands, key=lambda i: 0 if in_range(i, cx) else min(abs(cx - ranges[i][0]), abs(cx - ranges[i][1])))

    out = {}
    prev_band = None
    prev_amount = False
    for w in _merge_fragments(sorted(line_words, key=lambda w: w["x0"])):
        t = w["text"]
        cx = (w["x0"] + w["x1"]) / 2
        is_date = bool(DATE_TOKEN.fullmatch(t))
        amt = is_amount_like(t) and not is_date
        if is_date and dates:
            band = nearest(dates, cx)
        elif amt and numeric and cx >= min(bands[i]["x0"] for i in numeric) - 30 and (clusters or any(in_range(i, cx) for i in numeric)):
            if clusters:
                centre, band = min(clusters, key=lambda c: abs(c[0] - cx))
                if abs(centre - cx) > 40:
                    band = nearest(numeric, cx)
            else:
                band = nearest(numeric, cx)
        elif SUFFIX_WORD.match(t) and prev_amount and prev_band is not None:
            band = prev_band
        elif textual:
            band = nearest(textual, cx)
        else:
            band = nearest(list(range(n)), cx)
        out.setdefault(band, []).append(t)
        prev_band, prev_amount = band, amt
    return out


def _rows_from_words(page, page_no, bands_prev, loc, section_prev):
    words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False)
    lines = _lines(words)
    bands = bands_prev
    rows: list[Row] = []
    seen = 0
    pending_text: list[str] = []
    last_row: Row | None = None
    last_bottom = None
    section = section_prev
    start = 0
    found_here = None
    for i, ln in enumerate(lines[: (len(lines) if bands is None else 8)]):
        cells = _header_bands(ln["words"])
        if cells:
            found_here = cells
            start = i + 1
            break
        txt = " ".join(w["text"] for w in ln["words"])
        if section_type(txt) and len(txt) < 60:
            section = section_type(txt)
    if found_here:
        bands = found_here
    if bands is None:
        return rows, None, 0, section
    roles = {i: b["role"] for i, b in enumerate(bands) if b["role"] and b["role"] not in ("ignore", "valuedate")}
    has_dc = bool({"debit", "credit"} & set(roles.values()))
    clusters = _numeric_clusters(bands, [ln for ln in lines[start:] if not _header_bands(ln["words"])])

    for ln in lines[start:]:
        if _header_bands(ln["words"]):
            continue
        by_band = _assign(bands, ln["words"], clusters)
        cells = [" ".join(by_band.get(i, [])) for i in range(len(bands))]
        text_all = " ".join(c for c in cells if c).strip()
        if not text_all:
            continue
        if SKIP_ROW.search(text_all):
            bal_idx = next((i for i, r in roles.items() if r == "balance"), None)
            bal, _ = parse_amount(cells[bal_idx], loc, signed=True) if bal_idx is not None else (None, None)
            if bal is not None:
                rows.append(Row(None, text_all, None, None, None, None, bal, page_no))
            continue
        row = _row_from_cells(cells, roles, page_no, loc, None if has_dc else section)
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
            if not has_dc and section_type(text_all) and len(text_all) < 60 and not any(ch.isdigit() for ch in text_all):
                section = section_type(text_all)
                pending_text = []
                last_row = None
                continue
            gap = (ln["top"] - last_bottom) if last_bottom is not None else 99
            if last_row is not None and gap < 6:
                last_row.desc = (last_row.desc + " " + text_all).strip()
                last_bottom = ln["bottom"]
            else:
                pending_text.append(text_all)
                if len(pending_text) > 3:
                    pending_text = pending_text[-3:]
    return rows, bands, seen, section


# --- document ---------------------------------------------------------------------

def parse_statement(pdf_bytes: bytes, password: str | None = None) -> ParseResult:
    res = ParseResult()
    with pdfplumber.open(io.BytesIO(pdf_bytes), password=password or "") as pdf:
        res.pages = len(pdf.pages)
        texts = [(p.extract_text() or "") for p in pdf.pages]
        res.text = "\n\n".join(texts)
        loc = detect_locale(res.text)
        res.locale = loc
        all_rows: list[Row] = []
        roles_prev = None
        bands_prev = None
        section = None
        for pno, page in enumerate(pdf.pages, start=1):
            rows, roles, seen, section = _rows_from_ruled_tables(page, pno, roles_prev, loc, section)
            if roles:
                roles_prev = roles
            if not rows:
                rows, bands, seen, section = _rows_from_words(page, pno, bands_prev, loc, section)
                if bands:
                    bands_prev = bands
            res.rows_seen += seen
            all_rows.extend(rows)
    res.header_found = roles_prev is not None or bands_prev is not None
    res.transactions = _finalise(all_rows, res)
    return res


def _finalise(rows: list[Row], res: ParseResult) -> list[dict]:
    txns: list[dict] = []
    loc = res.locale
    prev_balance: Decimal | None = None
    swap_votes = keep_votes = 0
    for r in rows:
        if r.date is None and r.balance is not None and r.debit is None and r.credit is None and r.amount is None:
            prev_balance = r.balance
            continue
        if r.date is None:
            continue
        if r.balance is not None and prev_balance is not None and (r.debit is not None or r.credit is not None):
            debit = r.debit or Decimal(0)
            credit = r.credit or Decimal(0)
            if abs((prev_balance - debit + credit) - r.balance) <= Decimal("0.011"):
                keep_votes += 1
            elif abs((prev_balance + debit - credit) - r.balance) <= Decimal("0.011"):
                swap_votes += 1
        if r.balance is not None:
            prev_balance = r.balance
    swap = swap_votes > keep_votes
    if swap:
        res.warnings.append("debit/credit columns were swapped by the balance check")

    # a signed single amount column: which sign is the debit? Positive means a
    # charge on credit-card statements, a deposit on bank statements.
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
        if debit is not None and debit > 0 and (credit is None or credit == 0):
            txn_type, amount = "debit", debit
        elif credit is not None and credit > 0:
            txn_type, amount = "credit", credit
        elif r.amount is not None:
            amount = r.amount
            raw_neg = bool(re.search(r"^\s*[-−(]|[-−]\s*$", r.raw_amount or ""))
            if r.balance is not None and prev_balance is not None:
                delta = r.balance - prev_balance
                txn_type = "credit" if delta > 0 else "debit"
            elif r.amount_suffix in ("cr", "dr") and not raw_neg:
                txn_type = "credit" if r.amount_suffix == "cr" else "debit"
            elif r.type_hint:
                txn_type = r.type_hint
            elif raw_neg:
                txn_type = "credit" if loc.credit_card else "debit"
            elif loc.credit_card:
                txn_type = "debit"
            else:
                txn_type = "credit" if re.search(r"^\s*\+", r.raw_amount or "") else "debit"
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
        if loc.credit_card and method == "bank_transfer" and txn_type == "debit":
            method = "credit_card"
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


# --- CSV -------------------------------------------------------------------------

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
    loc = detect_locale(text)
    res.locale = loc
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
        row = _row_from_cells([c.strip() for c in r], roles, 1, loc, None)
        if row:
            rows.append(row)
    res.transactions = _finalise(rows, res)
    return res


# --- LLM fallback (extracted text → expense-ai-proxy) ---------------------------------

STATEMENT_PROMPT = """Extract every transaction from this bank or credit-card statement text (any
country, any language). Return ONLY a JSON object {"transactions": [...]} where each item has:
amount (positive number), date (YYYY-MM-DD), merchant, description,
transactionType ("debit" for money leaving the account or a card charge, "credit" for money
received or a card payment/refund — use the Debit/Withdrawal vs Credit/Deposit columns, the sign,
the section heading or the running balance direction), paymentMethod ("bank_transfer",
"debit_card", "credit_card", "cash" or "other"), category (one of food_dining, transportation,
shopping, entertainment, bills_utilities, healthcare, travel, other) and confidence (0-1).
Dates may be day-first or month-first — infer from the whole document. Skip opening/closing
balance, subtotal and total rows."""


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
    cats = {c for c, _ in CATEGORY_RULES} | {"other"}
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
            "category": it.get("category") if it.get("category") in cats else "other",
            "paymentMethod": it.get("paymentMethod") if it.get("paymentMethod") in {"bank_transfer", "debit_card", "credit_card", "cash", "other"} else "other",
            "description": (it.get("description") or "")[:200],
            "transactionType": "credit" if it.get("transactionType") == "credit" else "debit",
            "confidence": min(max(float(it.get("confidence") or 0.6), 0), 1),
            "balanceVerified": None,
            "page": None,
        })
    return out

# Parses raw 10-Q and 10-K HTML into a dict of named sections.
import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

#Trying to find sections in filings using regex patterns. 
PATTERNS_10Q = {
    "financials": [
        r"item\s+1[\.\s\xa0]+financial\s+statements",
        r"part\s+i[\s\.\xa0]+financi",             
    ],
    "mda": [
        r"item\s+2[\.\s\xa0]+management",
        r"management.{0,5}s?\s+discussion\s+and\s+analysis", 
    ],
    "risk_factors": [
        r"item\s+1a[\.\s\xa0]+risk\s+factors",
        r"item\s+1a[\.\s\xa0]+ri\s*sk\s+factors",  
        r"item\s+1a[\.\s\xa0]+ris\s*k\s+factors",   
    ],
    "quantitative": [
        r"item\s+3[\.\s\xa0]+quantitative",
    ],
    "controls": [
        r"item\s+4[\.\s\xa0]+controls",
    ],
}

PATTERNS_10K = {
    "business_overview": [
        r"item\s+1[\.\s\xa0]+business",
        r"item\s+1\b",
    ],
    "risk_factors": [
        r"item\s+1a[\.\s\xa0]+risk\s+factors",
        r"item\s+1a[\.\s\xa0]+ri\s*sk\s+factors",   
        r"item\s+1a[\.\s\xa0]+ris\s*k\s+factors",   
    ],
    "mda": [
        r"item\s+7[\.\s\xa0]+management",
    ],
    "financials": [
        r"item\s+8[\.\s\xa0]+financial\s+statements",
        r"item\s+8[\.\s\xa0]+financial\s+state",   
    ],
    "controls": [
        r"item\s+9a[\.\s\xa0]+controls",
    ],
}

OUTPUT_SECTIONS_10Q = ["financials", "mda", "risk_factors"]
OUTPUT_SECTIONS_10K = ["business_overview", "risk_factors", "mda", "financials"]


_FINANCIALS_STUB_RE = re.compile(
    r"set\s+forth|incorporated\s+by\s+reference|see\s+pages?"
    r"|appears?\s+on\s+pages?|included\s+in\s+this\s+annual",
    re.IGNORECASE,
)


_FINANCIALS_FALLBACK_10K = [
    r"report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm\s*\n\s*to\s+the\b",
    r"index\s+to\s+(?:consolidated\s+)?financial\s+statements",
]

# A found 10-Q financials section shorter than this is treated as missing.
_MIN_FINANCIALS_CHARS_10Q = 1000

# Statement-title anchors for 10-Qs whose "Item 1" heading sits too close to the TOC and gets
# rejected by _is_toc_entry. Order is load-bearing: the operations/income statement carries the
# EPS line — anchoring on comprehensive income or balance sheets would slice past it.
_FINANCIALS_FALLBACK_10Q = [
    r"(?:condensed\s+)?consolidated\s+statements?\s+of\s+operations",
    r"(?:condensed\s+)?consolidated\s+statements?\s+of\s+income",
    r"(?:condensed\s+)?consolidated\s+statements?\s+of\s+comprehensive\s+income",
    r"(?:condensed\s+)?consolidated\s+balance\s+sheets?",
    r"income\s+statements\b",
]

# Heuristic to determine if a section heading is likely a TOC entry, based on the density of other section headings nearby.
def _is_toc_entry(plain: str, pos: int) -> bool:

    forward = plain[pos: pos + 600]
    backward = plain[max(0, pos - 800): pos]
    forward_count = len(re.findall(r"\bitem\s+\d+", forward, re.IGNORECASE))
    backward_count = len(re.findall(r"\bitem\s+\d+", backward, re.IGNORECASE))
    return forward_count >= 5 or (forward_count >= 1 and backward_count >= 4)

# Given a list of regex patterns for a section, find the first match that isn't likely a TOC entry. Returns the position of the match or None if not found.
def _find_section(plain: str, patterns: list[str]) -> int | None:
    """
    Find the first non-TOC occurrence of a section heading.
    Tries each pattern in order, returns the first match that isn't a TOC entry.
    """
    for pattern in patterns:
        for m in re.finditer(pattern, plain, re.IGNORECASE):
            if not _is_toc_entry(plain, m.start()):
                return m.start()
    return None

# For 10-K filings, the financial statements section (Item 8) is often just a stub.This function detects that case and tries to locate the actual financial statements using fallback anchor patterns.
def _fix_financials_stub(plain: str, positions: dict) -> None:
    """
    Mutates `positions` in-place.

    If the 10-K financials section looks like a stub reference (Item 8 just says
    "see pages X-Y" or "information is set forth in..."), locate the actual
    financial statements using fallback anchor patterns.
    """
    fin_pos = positions.get("financials")

    stub_detected = False
    if fin_pos is None:
        stub_detected = True          # Item 8 was filtered as TOC; need fallback
    else:
        # Peek at how much content is actually in this section
        ordered = sorted(positions.items(), key=lambda x: x[1])
        for i, (name, start) in enumerate(ordered):
            if name == "financials":
                end = ordered[i + 1][1] if i + 1 < len(ordered) else len(plain)
                preview = plain[start:end]
                if len(preview) < 2000 and _FINANCIALS_STUB_RE.search(preview):
                    stub_detected = True
                    positions.pop("financials")
                break

    if not stub_detected:
        return

    search_after = (fin_pos + 1) if fin_pos is not None else 0
    for pat in _FINANCIALS_FALLBACK_10K:
        for m in re.finditer(pat, plain, re.IGNORECASE):
            if m.start() > search_after and not _is_toc_entry(plain, m.start()):
                positions["financials"] = m.start()
                return

# For 10-Q filings whose real "Item 1. Financial Statements" heading sits right after the TOC
# (and is therefore rejected by _is_toc_entry), anchor the financials section on the statement
# titles instead. Additive: when a substantial financials section was already found, this is a no-op.
def _fix_financials_missing_10q(plain: str, positions: dict) -> None:
    fin_pos = positions.get("financials")
    if fin_pos is not None:
        ordered = sorted(positions.items(), key=lambda x: x[1])
        for i, (name, start) in enumerate(ordered):
            if name == "financials":
                end = ordered[i + 1][1] if i + 1 < len(ordered) else len(plain)
                if end - start >= _MIN_FINANCIALS_CHARS_10Q:
                    return
                break

    # The anchor must precede every other found section: statements come first in a 10-Q, and
    # prose references ("see our condensed consolidated statements of operations") or exhibit-index
    # lines near EOF must never win.
    others = [pos for name, pos in positions.items() if name != "financials"]
    limit = min(others) if others else len(plain)
    for pat in _FINANCIALS_FALLBACK_10Q:
        for m in re.finditer(pat, plain, re.IGNORECASE):
            if m.start() >= limit:
                break
            if not _is_toc_entry(plain, m.start()):
                positions["financials"] = m.start()
                return


# Main parsing function. Takes raw HTML text and filing type, returns dict of section name to content.
def parse_sections(text: str, filing_type: str) -> dict:
    soup = BeautifulSoup(text, "lxml")
    plain = soup.get_text(separator="\n")

    patterns = PATTERNS_10Q if filing_type == "10-Q" else PATTERNS_10K
    output_keys = OUTPUT_SECTIONS_10Q if filing_type == "10-Q" else OUTPUT_SECTIONS_10K

    # find the first non-TOC match for each section
    positions = {}
    for name, pattern_list in patterns.items():
        pos = _find_section(plain, pattern_list)
        if pos is not None:
            positions[name] = pos

    # for 10-K, replace stub Item 8 with the actual financial statements;
    # for 10-Q, recover a financials section whose heading was rejected as a TOC entry
    if filing_type == "10-K":
        _fix_financials_stub(plain, positions)
    else:
        _fix_financials_missing_10q(plain, positions)

    # sort found sections by position in the document
    ordered = sorted(positions.items(), key=lambda x: x[1])

    # slice content between consecutive section boundaries
    raw_sections = {}
    for i, (name, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(plain)
        raw_sections[name] = plain[start:end].strip()

    return {key: raw_sections.get(key, "") for key in output_keys}

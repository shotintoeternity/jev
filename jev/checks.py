"""Deterministic checks that belong in code, not in a model."""

import re
from difflib import SequenceMatcher
from urllib.parse import urlsplit

from .schema import BAD, QUALIFIERS, Trace, Verdict

_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", " ": " "})
PASSAGE_CHARS = 1200  # context on each side of a matched quote sent to Jev
NEAR_MISS = 0.8  # share of the quote that must match contiguously to count as a paraphrased quote


def normalize(text: str) -> str:
    """Fold curly quotes/dashes and collapse whitespace, so quotes match across line wraps."""
    text = re.sub(r"\\[ntr]", " ", text)  # quotes sometimes carry a literal "\n" for a line break
    return re.sub(r"\s+", " ", text.translate(_QUOTES)).strip()


def _url_key(url: str) -> str:
    parts = urlsplit(url)
    return (parts.netloc.removeprefix("www.") + parts.path.rstrip("/")).lower()


def find_document(url: str | None, documents: dict[str, str]) -> str | None:
    if not url:
        return None
    if url in documents:
        return documents[url]
    want = _url_key(url)
    for u, text in documents.items():
        if _url_key(u) == want:
            return text
    return None


def locate_quote(quote: str, document: str) -> tuple[str, str | None]:
    """Return (status, passage). status is found, near_miss, or missing."""
    doc = normalize(document)
    needle = normalize(quote)
    if not needle:
        return "missing", None
    at = doc.find(needle)
    status = "found"
    if at < 0:
        m = SequenceMatcher(None, doc, needle, autojunk=False).find_longest_match(0, len(doc), 0, len(needle))
        if m.size < NEAR_MISS * len(needle):
            return "missing", None
        at, status = m.a - m.b, "near_miss"
        at = max(at, 0)
    start = max(0, at - PASSAGE_CHARS)
    end = min(len(doc), at + len(needle) + PASSAGE_CHARS)
    return status, doc[start:end]


_NUM = re.compile(r"\d+(?:[.,]\d+)*s?")


def _nums(text: str) -> set[str]:
    text = re.sub(r"(\d\.) (\d)", r"\1\2", text)  # pages sometimes break "29,031. 69"
    return {n.replace(",", "") for n in _NUM.findall(text)}


def missing_numbers(statement: str, passage: str) -> list[str]:
    """Numbers in the statement that never appear in the passage. Jev is weak on numbers, so code checks them."""
    have = {n.rstrip("s") for n in _nums(normalize(passage))}
    missing = []
    for n in _nums(statement):
        if n.endswith("s") and len(n) == 5:  # a decade such as "1950s" is backed by any year in it
            if any(h[:3] == n[:3] and len(h) == 4 for h in have):
                continue
        if n.rstrip("s") not in have:
            missing.append(n)
    return sorted(missing)


def qualifier_gap(qualifier: str, evidence_score: float) -> float:
    """How many levels the stated qualifier exceeds Jev's evidence-strength score (0-4 scale)."""
    return QUALIFIERS.index(qualifier) - evidence_score


def dangling_refs(trace: Trace) -> list[Verdict]:
    """Grounds or answer sentences that point at ids that do not exist."""
    known = {p.id for p in trace.premises}
    out = []
    for c in trace.claims:
        missing = [g for g in c.grounds if g not in known]
        if missing:
            out.append(Verdict(target=c.id, check="structure", status="non_sequitur", note=f"grounds reference unknown ids {missing}"))
        known.add(c.id)
    for i, s in enumerate(trace.answer, 1):
        missing = [g for g in s.claim_ids if g not in known]
        if missing:
            out.append(Verdict(target=f"S{i}", check="structure", status="untraced", note=f"cites unknown ids {missing}"))
    return out


def propagate(trace: Trace, verdicts: list[Verdict]) -> list[Verdict]:
    """Mark claims and answer sentences that rest on a failed premise or claim."""
    failed = {v.target for v in verdicts if v.status in BAD}
    out = []
    for c in trace.claims:  # claims are listed in dependency order
        bad = [g for g in c.grounds if g in failed]
        if bad and c.id not in failed:
            failed.add(c.id)
            out.append(Verdict(target=c.id, check="propagation", status="tainted", note=f"rests on {bad}"))
    for i, s in enumerate(trace.answer, 1):
        bad = [g for g in s.claim_ids if g in failed]
        if bad:
            out.append(Verdict(target=f"S{i}", check="propagation", status="tainted", note=f"asserts {bad}"))
    return out

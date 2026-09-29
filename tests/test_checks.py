from jev import checks
from jev.schema import AnswerSentence, Claim, Premise, Trace, Verdict


def premise(id, **kw):
    base = dict(statement=f"fact {id}", basis_kind="source", source_url="https://example.com/a", quote="x", falsifiable_by="-")
    return Premise(id=id, **(base | kw))


def claim(id, grounds, **kw):
    base = dict(claim=f"claim {id}", warrant="w", backing=None, qualifier="probably", rebuttals=[], falsifiable_by="-")
    return Claim(id=id, grounds=grounds, **(base | kw))


def trace(premises, claims, answer):
    return Trace(premises=premises, claims=claims, answer=[AnswerSentence(text=t, claim_ids=ids) for t, ids in answer])


# ---- quote matching -------------------------------------------------------


def test_quote_matches_across_line_wraps_and_curly_quotes():
    doc = "Intro.\nThe tower was built for the\n  “Exposition Universelle” of 1889.\nMore."
    status, passage = checks.locate_quote('built for the "Exposition Universelle" of 1889', doc)
    assert status == "found"
    assert "1889" in passage


def test_fabricated_quote_is_missing():
    status, passage = checks.locate_quote("completed in 1900 by Gustave Eiffel", "The tower was completed in 1889.")
    assert (status, passage) == ("missing", None)


def test_small_edit_is_near_miss_not_missing():
    doc = "The Eiffel Tower was built for the International Exhibition of Paris of 1889 commemorating the centenary."
    status, passage = checks.locate_quote("The Eiffel Tower was built for the International Exhibition of Paris in 1889", doc)
    assert status == "near_miss"
    assert passage is not None


def test_empty_quote_is_missing():
    assert checks.locate_quote("", "anything")[0] == "missing"


def test_find_document_ignores_www_and_trailing_slash():
    docs = {"https://www.asce.org/landmarks/eiffel-tower/": "text"}
    assert checks.find_document("https://asce.org/landmarks/eiffel-tower", docs) == "text"
    assert checks.find_document("https://asce.org/other", docs) is None
    assert checks.find_document(None, docs) is None


# ---- qualifier ------------------------------------------------------------


def test_qualifier_gap():
    assert checks.qualifier_gap("certainly", 1.0) == 3.0
    assert checks.qualifier_gap("probably", 2.0) == 0.0
    assert checks.qualifier_gap("possibly", 4.0) == -4.0


# ---- structure and propagation --------------------------------------------


def test_dangling_references_are_flagged():
    t = trace([premise("P1")], [claim("C1", ["P1", "P9"])], [("a", ["C1"]), ("b", ["C7"])])
    got = {(v.target, v.status) for v in checks.dangling_refs(t)}
    assert got == {("C1", "non_sequitur"), ("S2", "untraced")}


def test_claim_cannot_cite_a_later_claim():
    t = trace([premise("P1")], [claim("C1", ["C2"]), claim("C2", ["P1"])], [])
    assert [v.target for v in checks.dangling_refs(t)] == ["C1"]


def test_failure_propagates_through_the_chain_to_the_answer():
    t = trace(
        [premise("P1"), premise("P2")],
        [claim("C1", ["P1"]), claim("C2", ["C1", "P2"]), claim("C3", ["P2"])],
        [("uses C2", ["C2"]), ("uses C3", ["C3"])],
    )
    failed = [Verdict(target="P1", check="source", status="contradicted")]
    got = {(v.target, v.status) for v in checks.propagate(t, failed)}
    assert got == {("C1", "tainted"), ("C2", "tainted"), ("S1", "tainted")}


def test_unverified_does_not_propagate():
    t = trace([premise("P1", basis_kind="memory")], [claim("C1", ["P1"])], [("s", ["C1"])])
    failed = [Verdict(target="P1", check="source", status="unverified")]
    assert checks.propagate(t, failed) == []


def test_changed_numbers_are_caught_in_code():
    passage = "Albert Einstein received his Nobel Prize one year later, in 1922, and the prize was worth 121,572 kronor."
    assert checks.missing_numbers("He received it in 1922.", passage) == []
    assert checks.missing_numbers("He received it in 4922.", passage) == ["4922"]
    assert checks.missing_numbers("It was worth 121572 kronor.", passage) == []
    assert checks.missing_numbers("No numbers here.", passage) == []


def test_number_check_tolerates_decades_and_broken_decimals():
    passage = "It converts to 29,031. 69 feet, a height determined in 1954."
    assert checks.missing_numbers("The height is 29,031.69 feet.", passage) == []
    assert checks.missing_numbers("It was set in the 1950s.", passage) == []
    assert checks.missing_numbers("It was set in the 1960s.", passage) == ["1960s"]


def test_dates_with_commas_are_not_merged():
    assert checks.missing_numbers("Announced in December 2020.", "Published December 8, 2020.") == []

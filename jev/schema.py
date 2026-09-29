"""Trace schema (Toulmin model) and verdict types."""

from typing import Literal

from pydantic import BaseModel, Field

# Ordered weakest to strongest; index is compared against Jev's evidence-strength score.
QUALIFIERS = ["possibly", "plausibly", "probably", "very likely", "certainly"]
Qualifier = Literal["possibly", "plausibly", "probably", "very likely", "certainly"]


class Premise(BaseModel):
    """A ground: a fact the argument starts from."""

    id: str = Field(description='Stable id such as "P1".')
    statement: str = Field(description="The fact, stated as one self-contained sentence.")
    basis_kind: Literal["source", "memory", "user", "definition"] = Field(
        description="source = a page you fetched; memory = your own background knowledge; "
        "user = stated in the request; definition = true by definition."
    )
    source_url: str | None = Field(description="URL of the fetched page, when basis_kind is source.")
    quote: str | None = Field(
        description="Verbatim excerpt from the fetched page that establishes the statement, "
        "copied character for character. Null unless basis_kind is source."
    )
    falsifiable_by: str = Field(description="What observation or record would show this premise is false.")


class Claim(BaseModel):
    """A conclusion drawn from premises and/or earlier claims."""

    id: str = Field(description='Stable id such as "C1".')
    claim: str = Field(description="The conclusion, as one self-contained sentence.")
    grounds: list[str] = Field(description="Ids of the premises and earlier claims this rests on.")
    warrant: str = Field(
        description="The specific rule that licenses moving from the grounds to the claim. "
        'Must be specific enough to be false; never "this follows logically".'
    )
    backing: str | None = Field(description="Support for the warrant, if the warrant is not self-evident.")
    qualifier: Qualifier = Field(description="How strongly the claim is asserted.")
    rebuttals: list[str] = Field(description="Conditions under which the claim would not hold.")
    falsifiable_by: str = Field(description="What observation would show this claim is false.")


class AnswerSentence(BaseModel):
    text: str = Field(description="One sentence of the final answer, in order.")
    claim_ids: list[str] = Field(
        description="Ids of the claims or premises this sentence asserts. Empty only for sentences "
        "that assert nothing factual (greetings, transitions, questions back to the user)."
    )


class Trace(BaseModel):
    premises: list[Premise]
    claims: list[Claim]
    answer: list[AnswerSentence] = Field(description="The final answer to the user, split into sentences.")

    def answer_text(self) -> str:
        return " ".join(s.text for s in self.answer)


# ---- verdicts -------------------------------------------------------------

Status = Literal[
    "supported",  # a source backs it
    "contradicted",  # a source says otherwise
    "unsupported",  # cited source says nothing about it
    "fabricated",  # quote not in the cited source
    "unverified",  # memory-based or source not fetched; nothing to check against
    "follows",
    "weak",
    "non_sequitur",
    "overclaimed",  # qualifier stronger than the evidence
    "tainted",  # rests on a failed premise or claim
    "traced",
    "untraced",  # answer asserts something the trace never argued
    "ok",
]

BAD = {"contradicted", "unsupported", "fabricated", "non_sequitur", "tainted", "untraced"}


class Verdict(BaseModel):
    target: str  # "P1", "C2", "S3"
    check: str  # which check produced this
    status: Status
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    note: str = ""
    needs_review: bool = False

"""Domain models — the shared vocabulary for the whole system.

These pydantic types are the contract between every layer: corpus → extraction → graph →
retrieval → MCP → agents → backtest → report. They are all JSON-serializable so they can be
written to the audit log, returned from MCP tools, and round-tripped through the graph store.

Design notes:
* Char **offsets** (``start``/``end``) are the unit of provenance everywhere — a citation is a
  proven ``(start, end)`` span into a document's text, never a vague "see section 3".
* ``DocNode`` (the hierarchical tree) stores **spans + summaries, not full text** — navigation
  sends structure only (token-cheap); text is resolved from spans on demand.
* "Raw" types are what an LLM *claims*; the verifier promotes them to citation-bearing types
  only after independently re-checking the quote against the source (verify, don't trust).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EntityType(StrEnum):
    """Node kinds in the knowledge graph."""

    FACTOR = "factor"      # e.g. value, momentum, size, quality, low_volatility
    ASSET = "asset"        # an asset / asset class / portfolio the factor concerns
    REGIME = "regime"      # a market regime / condition (e.g. high-inflation, recession)
    CLAIM = "claim"        # an assertion made in a document (the unit of evidence)
    AUTHOR = "author"      # who made/authored the claim or document
    DOCUMENT = "document"  # a source document


class RelationType(StrEnum):
    """Edge kinds in the knowledge graph."""

    AUTHORED_BY = "authored_by"    # document  -> author
    PROPOSES = "proposes"          # document  -> claim
    CONCERNS = "concerns"          # claim     -> factor/asset/regime
    AFFECTS = "affects"            # factor/regime -> asset/factor (directional influence)
    EVIDENCED_BY = "evidenced_by"  # claim     -> document (provenance)
    MENTIONS = "mentions"          # document  -> entity
    CONTRADICTS = "contradicts"    # claim     -> claim
    RELATED_TO = "related_to"      # generic association


class PathKind(StrEnum):
    """How a piece of evidence was retrieved (vectorless — every kind is traceable)."""

    GRAPH = "graph"    # graph traversal (entity -> relation -> entity)
    TREE = "tree"      # hierarchical document tree search (PageIndex-style)
    FILTER = "filter"  # structured/metadata filter


class FactorDirection(StrEnum):
    """Long/short construction direction for a cross-sectional factor."""

    HIGH_MINUS_LOW = "high_minus_low"  # long top quantile, short bottom
    LOW_MINUS_HIGH = "low_minus_high"  # long bottom quantile, short top


class Recommendation(StrEnum):
    """Honest research verdicts — never a buy/sell call."""

    PROMISING = "promising"            # survives the overfitting checks; worth further study
    INCONCLUSIVE = "inconclusive"      # mixed / underpowered evidence
    LIKELY_OVERFIT = "likely_overfit"  # fails the overfitting checks; do not trust


# ─────────────────────────────────────────────────────────────────────────────
# Documents & hierarchical structure (corpus + PageIndex tree)
# ─────────────────────────────────────────────────────────────────────────────


class Document(BaseModel):
    """A normalized source document. ``text`` is canonical; all offsets index into it."""

    id: str
    title: str
    source: str = ""          # e.g. "arxiv:q-fin", "sec:edgar", "blog"
    authors: list[str] = Field(default_factory=list)
    published: str = ""       # ISO date string if known
    text: str


class DocNode(BaseModel):
    """A node in a document's structure tree (title + summary + span + children).

    Mirrors the PageIndex shape: ``node_id`` is a stable, zero-padded id; ``start``/``end`` are
    char offsets into the owning document's ``text``; ``summary`` is an LLM/rule-generated gist
    used for navigation; ``children`` nest sub-sections. Text is intentionally *not* stored.
    """

    node_id: str
    title: str
    summary: str = ""
    start: int
    end: int
    level: int = 0
    children: list[DocNode] = Field(default_factory=list)


class DocTree(BaseModel):
    """A document plus its root structure node."""

    doc_id: str
    title: str
    root: DocNode


# ─────────────────────────────────────────────────────────────────────────────
# Citations (proven provenance)
# ─────────────────────────────────────────────────────────────────────────────


class Citation(BaseModel):
    """A verified span: this exact ``quote`` appears at ``doc.text[start:end]``."""

    doc_id: str
    quote: str
    start: int
    end: int
    node_id: str | None = None   # the tree node containing the span, if known
    section: str | None = None   # human-readable section title for display


# ─────────────────────────────────────────────────────────────────────────────
# Raw extraction (what an LLM claims, pre-verification)
# ─────────────────────────────────────────────────────────────────────────────


class RawEntity(BaseModel):
    type: EntityType
    name: str
    quote: str = ""                              # claimed supporting span text
    attributes: dict[str, str] = Field(default_factory=dict)


class RawRelation(BaseModel):
    type: RelationType
    source: str                                  # source entity name
    target: str                                  # target entity name
    quote: str = ""
    attributes: dict[str, str] = Field(default_factory=dict)


class RawExtraction(BaseModel):
    """The unverified output of an extraction call over a document (or node)."""

    entities: list[RawEntity] = Field(default_factory=list)
    relations: list[RawRelation] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge graph (verified entities + relations)
# ─────────────────────────────────────────────────────────────────────────────


class Entity(BaseModel):
    """A graph node. ``id`` is a stable slug (e.g. ``factor:momentum``)."""

    id: str
    type: EntityType
    name: str
    attributes: dict[str, str] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)


class Relation(BaseModel):
    """A graph edge with provenance."""

    type: RelationType
    source: str                                  # source entity id
    target: str                                  # target entity id
    citation: Citation | None = None
    attributes: dict[str, str] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval (vectorless — every result is a traceable path)
# ─────────────────────────────────────────────────────────────────────────────


class PathStep(BaseModel):
    """One hop in an evidence path (a graph edge, a tree descent, or a filter match)."""

    kind: PathKind
    ref: str                 # node_id / entity_id / filter expression
    label: str               # human-readable ("factor:value --affects--> regime:high_inflation")
    detail: str = ""


class EvidencePath(BaseModel):
    """A single traceable retrieval result: the path taken + the cited evidence it yielded."""

    query: str
    kind: PathKind
    steps: list[PathStep] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    text: str = ""           # the retrieved evidence text (resolved from spans)
    score: float = 0.0       # ranking score (relevance/overlap), for ordering only


class RetrievalResult(BaseModel):
    """All evidence paths for a query, plus the token cost of producing them."""

    query: str
    paths: list[EvidencePath] = Field(default_factory=list)
    navigations: int = 0   # hierarchical-doc navigations performed (each is an LLM call in vertex mode)
    input_tokens: int = 0
    output_tokens: int = 0


class NavSelection(BaseModel):
    """Result of a hierarchical tree-search navigation (PageIndex-style).

    ``thinking`` is the navigator's reasoning (the human-readable trace); ``node_ids`` are the
    selected tree nodes — together they form the traceable retrieval path.
    """

    thinking: str = ""
    node_ids: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Factor hypothesis & backtest
# ─────────────────────────────────────────────────────────────────────────────


class FactorHypothesis(BaseModel):
    """A testable cross-sectional factor thesis formed from cited evidence."""

    factor_name: str
    thesis: str
    rank_signal: str                              # which characteristic/column to rank on
    direction: FactorDirection = FactorDirection.HIGH_MINUS_LOW
    quantiles: int = Field(default=5, ge=2, le=20)
    rebalance: str = "M"                          # pandas offset alias (M=month-end)
    universe: list[str] = Field(default_factory=list)  # asset columns; empty = all
    supporting_citations: list[Citation] = Field(default_factory=list)


class BacktestStats(BaseModel):
    """Headline performance stats (the numbers the orchestrator independently verifies)."""

    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float


class BacktestResult(BaseModel):
    """Full deterministic backtest output. All numbers come from the tool, never the model."""

    factor_name: str
    stats: BacktestStats
    n_periods: int
    start: str
    end: str
    information_coefficient: float = 0.0
    turnover: float = 0.0
    data_source: str = "synthetic"


class OverfittingReport(BaseModel):
    """Overfitting diagnostics — the anti-self-deception layer."""

    in_sample_sharpe: float
    out_sample_sharpe: float
    sharpe_decay: float                # IS - OOS
    deflated_sharpe: float             # prob. true Sharpe > 0 after multiple-testing adjustment
    pbo: float                         # probability of backtest overfitting (CSCV)
    n_trials: int                      # number of configurations searched (drives DSR)
    param_sensitivity: float           # coefficient of variation of Sharpe across the param grid
    notes: list[str] = Field(default_factory=list)


class RiskFlag(BaseModel):
    title: str
    severity: Severity
    detail: str


class RiskAssessment(BaseModel):
    """The Critic's judgment over the diagnostics."""

    recommendation: Recommendation
    overfitting_risk: Severity
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    rationale: str


# ─────────────────────────────────────────────────────────────────────────────
# Per-role outputs (the pipeline's typed hand-offs)
# ─────────────────────────────────────────────────────────────────────────────


class Finding(BaseModel):
    """A cited claim surfaced by the Researcher."""

    claim: str
    citations: list[Citation] = Field(default_factory=list)


class ResearcherOutput(BaseModel):
    question: str
    findings: list[Finding] = Field(default_factory=list)
    evidence_paths: list[EvidencePath] = Field(default_factory=list)


class HypothesizerOutput(BaseModel):
    hypothesis: FactorHypothesis


class BacktesterOutput(BaseModel):
    reported_stats: BacktestStats          # what the agent says the tool returned (verified later)
    backtest: BacktestResult


class CriticOutput(BaseModel):
    diagnostics: OverfittingReport
    risk: RiskAssessment


# ─────────────────────────────────────────────────────────────────────────────
# Final report (the auditable artifact)
# ─────────────────────────────────────────────────────────────────────────────

DISCLAIMER = (
    "Research/educational use only. This is NOT financial advice. No claim of real alpha is "
    "made or implied. Backtests use synthetic or historical sample data and are illustrative; "
    "past performance does not indicate future results."
)


class Report(BaseModel):
    """The self-contained, evidence-linked research report."""

    question: str
    backend: str
    thesis: str
    hypothesis: FactorHypothesis
    evidence_paths: list[EvidencePath] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    backtest: BacktestResult
    diagnostics: OverfittingReport
    risk: RiskAssessment
    numbers_verified: bool
    narrative: str = ""
    disclaimer: str = DISCLAIMER

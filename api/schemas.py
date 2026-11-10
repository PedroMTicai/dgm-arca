"""Request and response contracts for the four evaluation endpoints.

These models are the *contract* between your implementation and the grading script.
The grader sends the request bodies defined here and validates the responses against
the response models. You can add optional fields to the responses (extra metadata is
welcome), but do not rename or remove the existing ones.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------- health


class PhaseStatus(BaseModel):
    """What ``/health`` reports for each phase."""

    status: Literal["ready", "pending", "error"]
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    project: str
    team: str = ""
    domain: str = ""
    phases: dict[str, PhaseStatus]


# ---------------------------------------------------------------------- phase 1: /reasoning


class ReasoningRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Problem statement for the model.")
    expected_answer: str | None = Field(
        default=None,
        description="Optional ground truth; when given, the response includes the verifier verdict",
    )
    max_new_tokens: int = Field(default=1024, ge=16, le=8192)


class VerifierVerdict(BaseModel):
    is_correct: bool
    predicted: str | None
    expected: str
    detail: str = ""


class ReasoningResponse(BaseModel):
    thinking: str = Field(..., description="Content of the <think> block (may be empty).")
    answer: str = Field(..., description="Content of the <answer> block, or the best extraction.")
    raw: str = Field(..., description="Full model output, untouched.")
    has_valid_format: bool
    verifier: VerifierVerdict | None = None
    tokens_generated: int = 0
    model: str = ""


# -------------------------------------------------------------------------- phase 2: /tools


class ToolsRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_turns: int = Field(default=5, ge=1, le=20, description="Maximum tool-calling rounds.")


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: Any = None
    ok: bool = True
    error: str | None = None
    latency_ms: float = 0.0


class ToolsResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallRecord]
    turns: int
    model: str = ""
    tokens_generated: int = 0
    finished: bool = Field(
        default=True, description="False when max_turns ran out while the model still called tools."
    )


# ---------------------------------------------------------------------------- phase 3: /rag


class RagRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    retriever: Literal["dense", "bm25", "hybrid"] = "hybrid"


class RetrievedChunk(BaseModel):
    id: str
    source: str = Field(..., description="Document the chunk comes from (file name, URL...).")
    score: float
    text: str


class RagResponse(BaseModel):
    answer: str
    chunks: list[RetrievedChunk]
    citations: list[str] = Field(default_factory=list, description="Chunk ids cited in the answer.")
    invalid_citations: list[str] = Field(
        default_factory=list,
        description="Bracketed ids in the answer that are not among the returned chunks.",
    )
    retriever: Literal["dense", "bm25", "hybrid"]
    model: str = ""


# -------------------------------------------------------------------------- phase 4: /agent


class AgentRequest(BaseModel):
    task: str = Field(..., min_length=1)
    max_steps: int = Field(default=10, ge=1, le=50)
    brain: Literal["base", "rlm", "thinking"] = Field(
        default="rlm",
        description="Model driving the loop: base model, your phase-1 RLM, or a thinking model",
    )
    session_id: str | None = Field(
        default=None,
        description="Reuse the memory of a previous conversation (multi-turn); omit for a new one.",
    )
    allow_actions: bool = Field(
        default=False,
        description="Execute tools with side effects (create_icsr_report). By default they are "
        "not executed and come back in pending_actions for human confirmation.",
    )


class AgentAction(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    thought: str
    action: AgentAction | None = None
    observation: str | None = None


class AgentResponse(BaseModel):
    final_answer: str
    steps: list[AgentStep]
    n_steps: int
    succeeded: bool = True
    model: str = ""
    tokens_used: int = 0
    stop_reason: str = Field(
        default="",
        description="final_answer, answered_without_final_answer, max_steps, loop_detected or "
        "error.",
    )
    pending_actions: list[AgentAction] = Field(
        default_factory=list, description="Side-effect calls that need human confirmation."
    )
    session_id: str | None = None
    memory_compressions: int = 0

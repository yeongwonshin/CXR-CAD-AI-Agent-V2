"""Dynamic MedRAX-style tool loop for CXR-CAD runtime workflows.

The goal of this module is to make the CXR-CAD workbench behave less like a
hard-coded pipeline and more like a lightweight agent: a planner inspects the
current case state and user intent, selects the next ready tools, executes them,
records observations, and replans until enough evidence is available.

This implementation intentionally has no LangGraph/LangChain dependency so the
existing project can keep its current requirements.  The control flow mirrors
MedRAX's core pattern: planner -> tool execution -> observation -> replan.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ToolRunner = Callable[["CXRCaseState"], Mapping[str, Any] | None]


@dataclass
class CXRCaseState:
    """Mutable state passed through the dynamic CXR agent loop."""

    contents: bytes
    filename: str
    content_type: Optional[str]
    model_key: str
    threshold: float
    question: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    executed_tools: List[str] = field(default_factory=list)
    skipped_tools: List[Dict[str, Any]] = field(default_factory=list)
    plan_history: List[Dict[str, Any]] = field(default_factory=list)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)

    def has(self, key: str) -> bool:
        value = self.data.get(key)
        return value is not None

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def question_l(self) -> str:
        return (self.question or "").lower()


@dataclass(frozen=True)
class CXRRuntimeTool:
    """Description and executor for one runtime CXR tool."""

    name: str
    description: str
    requires: Tuple[str, ...]
    provides: Tuple[str, ...]
    run: ToolRunner
    required: bool = False


class DynamicCXRWorkflowAgent:
    """Small dependency-free tool planner/executor inspired by MedRAX.

    The planner is deterministic by default, but it is stateful: it can select
    different tools depending on the uploaded file type, the current tool
    observations, probabilities, image quality, and the user's question.
    """

    QUALITY_TERMS = {
        "quality", "화질", "품질", "흐림", "blur", "sharp", "contrast", "재촬영", "artifact", "아티팩트",
    }
    REPORT_TERMS = {"report", "draft", "판독", "초안", "소견", "impression", "findings"}
    GRADCAM_TERMS = {"grad", "grad-cam", "cam", "heatmap", "히트맵", "근거", "evidence", "explain"}
    ROI_TERMS = {"roi", "위치", "어디", "부위", "해부학", "location", "where", "focus"}
    TRIAGE_TERMS = {"triage", "priority", "우선", "먼저", "응급", "긴급", "위험", "critical", "urgent"}
    COMPARISON_TERMS = {"비교", "변화", "악화", "호전", "compare", "change", "worse", "better", "follow-up"}
    CRITICAL_LABELS = {"Pneumothorax", "Pneumonia", "Edema", "Effusion", "Cardiomegaly"}

    def __init__(
        self,
        tools: Sequence[CXRRuntimeTool],
        *,
        max_iterations: int = 8,
        max_parallel_tools: int = 3,
        default_full_workup: Optional[bool] = None,
    ) -> None:
        self.tools: Dict[str, CXRRuntimeTool] = {tool.name: tool for tool in tools}
        self.max_iterations = max(1, max_iterations)
        self.max_parallel_tools = max(1, max_parallel_tools)
        self.default_full_workup = _env_bool("CXR_AGENT_DEFAULT_FULL_WORKUP", True) if default_full_workup is None else default_full_workup

    def run(self, state: CXRCaseState) -> CXRCaseState:
        """Run planner/tool/observation loop until no more tools are needed."""
        for iteration in range(1, self.max_iterations + 1):
            planned = self.plan_next_tools(state)
            state.plan_history.append(
                {
                    "iteration": iteration,
                    "planned_tools": [name for name, _ in planned],
                    "known_state_keys": sorted(_summarizable_keys(state.data.keys())),
                    "planner_mode": "state_and_intent_rules",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            if not planned:
                break

            for tool_name, rationale in planned:
                tool = self.tools[tool_name]
                started = time.time()
                trace_item: Dict[str, Any] = {
                    "iteration": iteration,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "name": tool.name,
                    "description": tool.description,
                    "requires": list(tool.requires),
                    "provides": list(tool.provides),
                    "rationale": rationale,
                    "status": "running",
                }
                try:
                    output = tool.run(state) or {}
                    if not isinstance(output, Mapping):
                        raise TypeError(f"Tool {tool.name} returned non-mapping output: {type(output)!r}")
                    state.data.update(dict(output))
                    state.executed_tools.append(tool.name)
                    trace_item.update(
                        {
                            "status": "completed",
                            "duration_ms": int((time.time() - started) * 1000),
                            "output_summary": _summarize_tool_output(output),
                        }
                    )
                except Exception as exc:
                    trace_item.update(
                        {
                            "status": "failed",
                            "duration_ms": int((time.time() - started) * 1000),
                            "error": str(exc),
                        }
                    )
                    state.tool_trace.append(trace_item)
                    raise
                state.tool_trace.append(trace_item)
        return state

    def plan_next_tools(self, state: CXRCaseState) -> List[Tuple[str, str]]:
        """Choose the next ready tool calls from current state and question."""
        if "InputRouter" not in state.executed_tools:
            return [("InputRouter", "Start by decoding the upload and routing DICOM versus standard image input.")]
        if "CXRClassifier" not in state.executed_tools:
            return [("CXRClassifier", "Classification probabilities are required before report, ROI, Grad-CAM, and triage tools can run.")]

        desired = self._desired_tools(state)
        planned: List[Tuple[str, str]] = []
        for tool_name in desired:
            if tool_name in state.executed_tools:
                continue
            tool = self.tools.get(tool_name)
            if not tool:
                continue
            missing = [key for key in tool.requires if key not in state.data]
            if missing:
                state.skipped_tools.append({"name": tool_name, "reason": f"Waiting for required state: {missing}"})
                continue
            planned.append((tool_name, self._rationale(tool_name, state)))
            if len(planned) >= self.max_parallel_tools:
                break
        return planned

    def _desired_tools(self, state: CXRCaseState) -> List[str]:
        q = state.question_l
        blank_question = not q.strip()
        full_workup = self.default_full_workup and blank_question
        desired: List[str] = []

        if full_workup or _contains_any(q, self.REPORT_TERMS):
            desired.append("ReportDraftTool")
        if full_workup or _contains_any(q, self.GRADCAM_TERMS):
            desired.append("GradCAMTool")
        if full_workup or _contains_any(q, self.QUALITY_TERMS) or state.get("is_dicom_input"):
            desired.append("QualityCheckTool")
        if full_workup or _contains_any(q, self.ROI_TERMS | self.GRADCAM_TERMS):
            desired.append("AnatomicalROITool")
        if full_workup or _contains_any(q, self.TRIAGE_TERMS):
            desired.append("TriageTool")

        # State-triggered replanning: high-risk or uncertain classifier outputs
        # should prompt quality/ROI/triage even when the user did not ask for them.
        top_probability = _safe_float(state.get("top_probability"), 0.0)
        detected = set(state.get("detected", []) or [])
        if detected & self.CRITICAL_LABELS or top_probability >= 0.75:
            _append_unique(desired, "AnatomicalROITool")
            _append_unique(desired, "TriageTool")
            _append_unique(desired, "ReportDraftTool")
        if 0.25 <= top_probability <= 0.45:
            _append_unique(desired, "QualityCheckTool")
            _append_unique(desired, "TriageTool")
        if state.get("quality_check") and "TriageTool" not in state.executed_tools:
            _append_unique(desired, "TriageTool")

        # Report and triage are safe fallbacks for sparse questions so that the
        # API keeps returning useful clinical scaffolding.
        if not desired:
            desired.extend(["ReportDraftTool", "TriageTool"])
        return desired

    def _rationale(self, tool_name: str, state: CXRCaseState) -> str:
        q = state.question_l
        if tool_name == "ReportDraftTool":
            return "Generate a clinician-editable draft because the case needs a text synthesis or the default workup is active."
        if tool_name == "GradCAMTool":
            return "Produce visual attribution because the question/default workup asks for evidence or model focus."
        if tool_name == "QualityCheckTool":
            return "Assess image quality because quality was requested, DICOM/windowing was used, or probabilities are uncertain."
        if tool_name == "AnatomicalROITool":
            return "Map predicted findings to review ROIs so the next reasoning step has location context."
        if tool_name == "TriageTool":
            if _contains_any(q, self.TRIAGE_TERMS):
                return "Rank review priority because the user asked for triage/priority."
            return "Integrate probabilities, critical labels, placeholder status, and quality observations into review priority."
        return "Tool selected by current state and user intent."


class DynamicBatchWorkflowAgent:
    """Planner for cross-case tools after per-image agent runs finish."""

    def __init__(self, *, max_iterations: int = 3) -> None:
        self.max_iterations = max(1, max_iterations)

    def run(
        self,
        *,
        cases: Sequence[Mapping[str, Any]],
        question: str,
        build_summary: Callable[[Sequence[Mapping[str, Any]], str], Mapping[str, Any]],
    ) -> Dict[str, Any]:
        trace: List[Dict[str, Any]] = []
        q = (question or "").lower()
        case_count = len(cases)
        comparison_requested = case_count > 1 or _contains_any(q, DynamicCXRWorkflowAgent.COMPARISON_TERMS)

        planned = ["BatchSummaryTool"]
        if comparison_requested:
            planned.insert(0, "ComparisonTool")

        summary: Dict[str, Any] = {}
        for iteration, tool_name in enumerate(planned, start=1):
            started = time.time()
            if tool_name == "ComparisonTool":
                output_summary = {
                    "case_count": case_count,
                    "reason": "multi_case_or_comparison_question",
                    "note": "Comparison is materialized inside BatchSummaryTool to avoid duplicating summary computation.",
                }
                status = "completed"
            else:
                summary = dict(build_summary(cases, question))
                output_summary = _summarize_tool_output(summary)
                status = "completed"
            trace.append(
                {
                    "iteration": iteration,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "name": tool_name,
                    "description": "Cross-case agent tool",
                    "rationale": _batch_rationale(tool_name, case_count, question),
                    "status": status,
                    "duration_ms": int((time.time() - started) * 1000),
                    "output_summary": output_summary,
                    "scope": "batch",
                }
            )
        return {"summary": summary, "tool_trace": trace, "planned_tools": planned}


def _batch_rationale(tool_name: str, case_count: int, question: str) -> str:
    if tool_name == "ComparisonTool":
        return "Run only because more than one image is present or the question asks for temporal/case comparison."
    return "Synthesize per-case observations, triage distribution, and optional comparison into a final batch answer."


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    text = text or ""
    return any(term in text for term in terms)


def _append_unique(items: List[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _summarizable_keys(keys: Iterable[str]) -> List[str]:
    hidden = {"contents", "image", "gradcam_b64"}
    return [key for key in keys if key not in hidden]


def _summarize_tool_output(output: Mapping[str, Any], *, limit: int = 6) -> Dict[str, Any]:
    """Summarize tool outputs without leaking images/base64-heavy payloads."""
    summary: Dict[str, Any] = {}
    for key, value in list(output.items())[:limit]:
        if key in {"image", "contents", "gradcam_b64"}:
            summary[key] = "<omitted>"
        elif isinstance(value, Mapping):
            summary[key] = {str(k): _compact_value(v) for k, v in list(value.items())[:6]}
        elif isinstance(value, (list, tuple)):
            summary[key] = [_compact_value(v) for v in list(value)[:6]]
        else:
            summary[key] = _compact_value(value)
    return summary


def _compact_value(value: Any) -> Any:
    if hasattr(value, "size") and hasattr(value, "mode"):
        return f"PIL.Image(size={getattr(value, 'size', None)}, mode={getattr(value, 'mode', None)})"
    if isinstance(value, str) and len(value) > 160:
        return value[:157] + "..."
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, Mapping):
        return {str(k): _compact_value(v) for k, v in list(value.items())[:4]}
    return value

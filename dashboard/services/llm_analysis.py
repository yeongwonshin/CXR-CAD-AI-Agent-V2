"""Fast LLM helpers for the analysis dashboard.

This module intentionally avoids LangChain in the Result Analysis page.  Direct
OpenAI-compatible HTTP calls remove import overhead and make sidebar switching
much faster while still using the same .env configuration.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any


def load_project_env() -> None:
    """Load project-level .env values for Streamlit/FastAPI processes."""
    try:
        from dotenv import load_dotenv
    except Exception:  # pragma: no cover - optional dependency guard
        return

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    seen: set[Path] = set()
    for dotenv_path in candidates:
        dotenv_path = dotenv_path.resolve()
        if dotenv_path in seen:
            continue
        seen.add(dotenv_path)
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)


load_project_env()


def get_configured_api_key() -> str:
    """Return the configured LLM API key without exposing it in the dashboard."""
    load_project_env()
    return (os.getenv("OPENAI_API_KEY") or os.getenv("CXR_AGENT_LLM_API_KEY") or "").strip()


def get_configured_model(default: str = "gpt-4o-mini") -> str:
    """Return the configured model name, preferring the CXR agent variable."""
    load_project_env()
    return (os.getenv("CXR_AGENT_LLM_MODEL") or os.getenv("OPENAI_MODEL") or default).strip()


def get_configured_base_url() -> str:
    """Return an OpenAI-compatible base URL from environment variables."""
    load_project_env()
    return (os.getenv("OPENAI_BASE_URL") or os.getenv("CXR_AGENT_LLM_BASE_URL") or "https://api.openai.com/v1").strip()


def langchain_is_ready() -> tuple[bool, str]:
    """Compatibility shim used by the dashboard.

    The page no longer requires LangChain; direct HTTP is always available as
    long as Python's standard library can run.
    """
    return True, ""


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def _shorten(text: str, limit: int = 5500) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _safe_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _openai_chat(messages: list[dict[str, str]], *, api_key: str, model_name: str, max_tokens: int) -> str:
    api_key = (api_key or get_configured_api_key()).strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 .env 또는 환경변수에 설정되어 있지 않습니다.")

    model = model_name or get_configured_model()
    endpoint = get_configured_base_url().rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": _safe_float_env("CXR_ANALYSIS_LLM_TEMPERATURE", 0.15),
        "max_tokens": int(_safe_float_env("CXR_ANALYSIS_LLM_MAX_TOKENS", max_tokens)),
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    timeout = int(_safe_float_env("CXR_ANALYSIS_LLM_TIMEOUT", 18))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    answer = _normalize_content(answer)
    if not answer:
        raise RuntimeError("LLM 응답이 비어 있습니다.")
    return answer


def generate_metric_summary(
    *,
    metric_title: str,
    metric_context: str,
    api_key: str,
    model_name: str,
) -> str:
    """Generate a grounded but compact summary for the selected metric."""
    messages = [
        {
            "role": "system",
            "content": (
                "당신은 흉부 X-ray AI 대시보드 분석가입니다. 제공된 지표와 수치에만 근거해 한국어로 답하세요. "
                "출력은 핵심 결론, 리스크/한계, 권장 액션을 짧은 bullet로 구성하세요. 데이터에 없는 내용은 추정하지 마세요."
            ),
        },
        {
            "role": "user",
            "content": f"[지표 제목]\n{metric_title}\n\n[지표 컨텍스트]\n{_shorten(metric_context, 5200)}",
        },
    ]
    return _openai_chat(messages, api_key=api_key, model_name=model_name, max_tokens=420)


def ask_metric_question(
    *,
    metric_title: str,
    metric_context: str,
    question: str,
    api_key: str,
    model_name: str,
) -> str:
    """Answer a user question grounded in the selected metric only."""
    messages = [
        {
            "role": "system",
            "content": (
                "당신은 흉부 X-ray AI 성능분석 보조자입니다. 제공된 지표 컨텍스트만 근거로 한국어로 답하세요. "
                "답할 수 없으면 부족한 정보를 분명히 말하고, 답변은 4~7문장으로 간결하게 작성하세요."
            ),
        },
        {
            "role": "user",
            "content": (
                f"[지표 제목]\n{metric_title}\n\n"
                f"[지표 컨텍스트]\n{_shorten(metric_context, 5000)}\n\n"
                f"[사용자 질문]\n{question}"
            ),
        },
    ]
    return _openai_chat(messages, api_key=api_key, model_name=model_name, max_tokens=320)

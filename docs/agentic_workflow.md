# Agentic Workflow

## 1. Purpose

The Agentic Workflow wraps the CXR prediction pipeline with runtime tools that help reviewers reason over one or more cases. It is inspired by tool-using medical imaging agents, but it remains dependency-light and grounded in the project’s own model outputs.

## 2. API endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /agent/status` | Returns agent runtime status and LLM configuration state |
| `POST /agent/analyze` | Runs multi-image case analysis with prediction, quality checks, anatomy assessment, triage, and batch comparison |
| `POST /agent/chat` | Answers follow-up questions using the existing agent result payload and chat history |

## 3. Single-case tool flow

For each uploaded image, the agent can build a case profile that includes:

1. prediction probabilities,
2. detected diseases,
3. top disease and probability,
4. image-quality checks,
5. DICOM metadata summary,
6. Grad-CAM availability,
7. anatomy assessment,
8. triage assessment,
9. report draft,
10. tool trace.

## 4. Batch-level tool flow

After all images are processed, the batch agent creates cross-case summaries.

```text
Case 1 profile
Case 2 profile
Case 3 profile
  -> probability matrix
  -> case overview rows
  -> shared findings and outliers
  -> review priority summary
  -> batch agent trace
```

## 5. Dynamic tool planning

The agent uses the question text and case state to decide which tools are relevant. For example:

| Reviewer question | Likely tool context |
| --- | --- |
| “Which case is most urgent?” | triage assessment, top probabilities, detected disease count |
| “Which image quality is poor?” | image-quality indicators |
| “Is the model looking at the right region?” | Grad-CAM and anatomy assessment |
| “Compare all cases.” | probability matrix and cross-case summary |
| “Draft a report.” | findings, impression, and report-draft fields |

## 6. LLM-backed follow-up chat

The follow-up chat endpoint does not rerun training or modify predictions. It compresses the existing agent result into a compact context and sends it to an OpenAI-compatible chat model when configured.

Relevant environment variables include:

```text
OPENAI_API_KEY
OPENAI_BASE_URL
CXR_AGENT_LLM_MODEL
CXR_AGENT_LLM_ENABLED
CXR_AGENT_LLM_FIRST
CXR_AGENT_TOOL_FIRST_FASTPATH
CXR_AGENT_DEFAULT_FULL_WORKUP
CXR_AGENT_LLM_MAX_TOKENS
CXR_AGENT_LLM_TIMEOUT
CXR_AGENT_LLM_TEMPERATURE
```

## 7. Deterministic fallback

If LLM settings are missing or unavailable, the system can still return deterministic context-based answers for common question types, including:

- case comparison,
- triage priority,
- quality review,
- Grad-CAM availability,
- report draft explanation,
- and disease-specific probability summaries.

## 8. Tool trace

The agent response includes planned tools and executed tool trace. This makes the workflow easier to audit because reviewers can see which context blocks were used for the answer.

## 9. Safety posture

Agent answers should be interpreted as structured review assistance. They are grounded in the model output and metadata, but they are not a final diagnosis.

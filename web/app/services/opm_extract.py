from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import ValidationError

from .opm_validate import humanize_diagram_validation, validate_diagram

logger = logging.getLogger(__name__)

# --- Timeout for each HTTP request (seconds) ---
_LLM_TIMEOUT_S = 30.0


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_truthy(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on")


def _max_llm_rounds() -> int:
    """JSON parse retries + validation repair rounds (each round is at least one LLM call)."""
    n = _env_int("OPM_MAX_LLM_ROUNDS", 4)
    return max(1, min(20, n))


def _llm_temperature() -> float:
    """Sampling temperature for main extraction (default 0). Set OPM_TEMPERATURE e.g. 0.2 for more varied recall."""
    return _env_float("OPM_TEMPERATURE", 0.0)


def _expand_llm_temperature() -> float:
    """Temperature for optional expand pass (default 0.15 if OPM_EXPAND_TEMPERATURE unset)."""
    t = os.environ.get("OPM_EXPAND_TEMPERATURE", "").strip()
    if t:
        try:
            return float(t)
        except ValueError:
            pass
    return max(_llm_temperature(), 0.15)


class OPMExtractionError(Exception):
    """Normalized failure from the LLM extraction layer (router maps this to HTTP)."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        raw_response: str | None = None,
        *,
        stage: str = "llm",
        technical_detail: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.detail = detail
        self.raw_response = raw_response
        self.stage = stage
        self.technical_detail = technical_detail
        super().__init__(detail)


def parse_json(raw: str) -> dict:
    """
    Parse LLM output into a Python dict.

    - If JSON decoding fails, raise ValueError("Invalid JSON")
    - If decoded JSON is not a dict, raise ValueError("JSON must be object")
    """
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise ValueError("Invalid JSON") from exc

    if not isinstance(data, dict):
        raise ValueError("JSON must be object")

    return data


def normalize_llm_output(raw: str) -> str:
    """Strip optional markdown fences if the model wrapped JSON in ``` blocks."""
    s = raw.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if not lines:
        return s
    if lines[0].strip().startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def build_system_prompt() -> str:
    return """You are an OPM diagram extraction engine.

**Coverage first:** For any non-trivial passage, maximize **recall** of what the text actually says. Prefer **many nodes and links** over a tiny toy graph. Do **not** summarize the whole passage into one main entity + one generic process unless the input is truly that short.

**Labels:** Reuse **important words and phrases from the source** in `label` fields (concise; keep distinctive terms from the input). Each `label` is short (aim ≤160 characters); split very long ideas into **multiple nodes** instead of dropping detail.

**Before you output JSON, mentally scan the text for items you have not yet represented** (entities, actions, conditions, constraints). Add nodes/links for them using valid triples only.

Density rule: short notes → small graph; long structured text → **large** graph with separate nodes for distinct requirements, entities, and steps.

Return exactly one JSON object with this schema:
{
  "version": "1.0",
  "nodes": [
    { "id": "<string>", "kind": "<object|process|state>", "label": "<string>" }
  ],
  "links": [
    { "id": "<string>", "source": "<node-id>", "target": "<node-id>", "relation": "<agent|instrument|consumption|result|effect|aggregation|specialization|characterization>" }
  ]
}

**Node `id` format (required):** lowercase letters and digits only, with single hyphens or underscores between segments — e.g. `glp1_receptor`, `gs-alpha`, `step_3`. No spaces, no uppercase, no dots.

Node kinds:
- object = entity, actor, thing, or other noun-like referent in the domain
- process = verb-like action or step
- state = named condition, risk, or outcome phrase when modeled as a condition

Preferred (source_kind, relation, target_kind) triples — stick to these patterns:
- (object, agent|instrument|consumption, process)
- (process, result, object)
- (process, effect, state)
- (object, characterization, state)
- (object, aggregation|specialization, object) with consistent direction

How to use relations:
- agent / instrument / consumption → source kind should be "object", target kind "process"
- result → process to object
- effect → process to state (not object→process; not state as source for effect)
- characterization → object to state
- aggregation / specialization → object to object

**Mandatory pattern check before you output:** For EVERY link with relation `agent`, `instrument`, or `consumption`, look up `target` in `nodes` and confirm that node's `kind` is **exactly** `"process"`. If you planned object→object, use `aggregation` or `specialization` instead, or introduce a `process` node and point `instrument`/`agent` at that process.

**Tiny valid template (structure only — replace labels with text from the source):**
{"version":"1.0","nodes":[
  {"id":"prod","kind":"object","label":"…"},
  {"id":"adj","kind":"object","label":"…"},
  {"id":"give","kind":"process","label":"…"},
  {"id":"out","kind":"state","label":"…"}
],"links":[
  {"id":"l1","source":"prod","target":"give","relation":"agent"},
  {"id":"l2","source":"adj","target":"give","relation":"instrument"},
  {"id":"l3","source":"give","target":"out","relation":"effect"}
]}
Here `give` MUST be `"kind":"process"` so `l1` and `l2` targets are valid. **Never** set `"target"` of `instrument` to an object-only id.

**Quick check:** `result` always means **process → object** (target must be `kind: "object"`). `effect` always means **process → state** (target must be `kind: "state"`). If the outcome is a **state** node, the edge from the relevant **process** to that state **must** be `effect`, never `result`.

**Direction guard (very common errors):**
- `effect` and `result` edges always **start from a process** (`source` must be `kind: "process"`). Never draw object→state or state→anything with `effect` or `result`.
- `agent`, `instrument`, and `consumption` edges always **end at a process** (`target` must be `kind: "process"`). Never point them at an **object** or **state**—only at **process**. **Never** use `instrument` or `agent` between two objects (object→object); for relations among objects use `aggregation` or `specialization`, or add an intermediate **process** and link object→process.
- Secondary entities named in the text (supports, prerequisites, co-measures) must link as object--instrument|agent--> some **process**, not object--agent--> state and not object--instrument--> object.

**Pattern (placeholders, not a fixed scenario):**
- INVALID: object(Oa) --result--> state(Sx) — states are never `result` targets.
- INVALID: object(Oa) --effect--> state(Sx) — `effect` must be **process**→state, not object→state.
- INVALID: object(Ob) --agent--> state(Sx) — `agent` must be object→**process**.
- VALID: object(Oa) --agent|instrument--> process(P1) --effect--> state(Sx); add further object→process links for other entities as the text requires; use process--result--> object(Oc) only when the target is typed as object.

Avoid confusing edges:
- Do not use agent/instrument/consumption when the source is not an object.
- States are not "consumed"; avoid consumption where the source is a state node.
- **Never point `result` at a state node** — `result` is process→object. If the outcome reads like a condition phrase and you type it as state, use **`effect`** (process→state), not `result`.
- Do not point consumption/instrument/agent at a state id; the target should be a process. Supporting entities that are not the main subject link via instrument/agent to an appropriate **process**, or via aggregation among objects—not "consumption" into a state.
- Do not use consumption for "because of X, do Y"; use objects plus valid relations or omit.

For **effect** (process→state): the process should plausibly influence the state as the text states. If the text says an action is taken *because of* a risk, do not reverse direction (mitigation should not read as causing the risk).

Structured or multi-part texts:
- Prefer explicit actor objects with agent→process when the text names who does what.
- Keep risk or adverse states separate from procedural steps unless a valid relation applies.

Dense passages (many sentences, lists, or constraints):
- **Materialize lists:** distinct items in the source should become distinct nodes or clearly distinct labels, not one vague node.
- Capture separate clauses: who/what/when/where/how/forbidden/allowed when the text states them.
- Use separate process nodes when the text describes clearly different actions or phases.
- Use objects for entities and tools named in text; use states for conditions/outcomes when you model them as states.
- Put salient qualifiers in **process or object labels** so they appear on the graph.
- Do **not** omit an explicit fact because the graph is getting large—valid OPM structure is required, but **sparsity is a failure** for dense source text.

**Multi-entity rule:** Whenever the text names **several distinct entities or requirements**, represent them with extra **object** and/or **process** nodes and **valid triples only**. Do not collapse unrelated ideas into one process plus one outcome. Introduce intermediate **process** nodes when needed so adjuncts and co-measures connect object→process, not object→state.

General:
- Every link must have a unique "id" across "links".
- Extract only what the text supports; do not invent facts.
- For long structured passages, prefer richer graphs when the text enumerates requirements.
- nodes and links must be arrays (use [] if empty).

Output raw JSON only. No markdown. No explanation."""


def _dense_passage_followup(text: str) -> str:
    """Extra instructions when input looks like a multi-clause passage (not a one-liner)."""
    t = text.strip()
    if len(t) < 160:
        return ""
    return """

The input has multiple clauses or constraints. Your graph must reflect **several** objects and processes—not a minimal chain with one main entity, one process, and one outcome.

Where the source text states them, add nodes and valid links for: supporting or parallel measures; timing or modality qualifiers; restrictions or prohibitions; handling or storage instructions; population or eligibility; distinct steps that are not the same action. Use generic labels taken from the text, not a canned template.

If your first draft has only three nodes but the passage enumerates more, expand until those mentions are represented.
"""


def build_user_prompt(text: str) -> str:
    return f"""Extract an OPM diagram from the following text.

Requirements:
1. Include salient **object** nodes for entities the text names (main subject, supporting measures, tools, locations, etc.).
2. Use separate **process** nodes when the text describes clearly different actions or phases.
3. Use **state** nodes for conditions/outcomes where appropriate; connect with **effect** (process→state) only.
4. Connect objects that play a supporting role to a **process** via instrument/agent (never object→state for agent/instrument). **Each** instrument/agent/consumption link's `target` node must have `"kind":"process"` in `nodes`.
5. Do not output a minimal skeleton if the passage lists multiple requirements—expand the graph.
{_dense_passage_followup(text)}
<text>
{text}
</text>"""


def _openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _call_llm_retry_502_once(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float | None = None,
) -> str:
    """Single LLM call; on 502 only, retry once with the same prompts."""
    try:
        return call_llm(system_prompt, user_prompt, temperature=temperature)
    except OPMExtractionError as exc:
        if exc.status_code == 502:
            return call_llm(system_prompt, user_prompt, temperature=temperature)
        raise


def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float | None = None,
) -> str:
    """
    One chat completion. 30s timeout. Raises OPMExtractionError on transport/empty/API errors.
    """
    model = os.environ.get("OPM_MODEL", "").strip()
    if not model:
        raise OPMExtractionError(502, "OPM_MODEL is not set", None)

    if temperature is None:
        temperature = _llm_temperature()

    client = _openai_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=4096,
            timeout=_LLM_TIMEOUT_S,
        )
    except APITimeoutError as exc:
        raise OPMExtractionError(502, "LLM request timed out", None) from exc
    except APIConnectionError as exc:
        raise OPMExtractionError(502, "LLM connection failed", None) from exc
    except APIStatusError as exc:
        code = getattr(exc, "status_code", None)
        if code == 429:
            raise OPMExtractionError(502, "LLM rate limited", None) from exc
        raise OPMExtractionError(502, f"LLM error ({code}): {exc}", None) from exc
    except OPMExtractionError:
        raise
    except Exception as exc:
        raise OPMExtractionError(502, f"LLM request failed: {exc}", None) from exc

    if not response.choices:
        raise OPMExtractionError(502, "Empty LLM response", None)

    message = response.choices[0].message
    content = message.content if message else None
    if content is None or not str(content).strip():
        raise OPMExtractionError(502, "Empty LLM response", None)

    return str(content).strip()


def _should_auto_expand(text: str, data: dict) -> bool:
    """
    If enabled, run a second LLM pass when the text is long but the graph is still tiny.
    Enable with OPM_AUTO_EXPAND=1 (opt-in so tests and default behavior stay unchanged).
    """
    if not _env_truthy("OPM_AUTO_EXPAND"):
        return False
    min_chars = _env_int("OPM_EXPAND_MIN_CHARS", 200)
    max_nodes = _env_int("OPM_EXPAND_MAX_NODES", 5)
    t = text.strip()
    if len(t) < min_chars:
        return False
    nodes = data.get("nodes") or []
    return len(nodes) <= max_nodes


def _build_expand_user_prompt(text: str, diagram: dict) -> str:
    return f"""The OPM JSON below is valid but is sparse relative to the source text length.

Your task: output one **new** complete OPM JSON that **keeps** the relationships that are still correct and **adds** nodes and links for every distinct entity, step, constraint, supporting measure, or qualifier mentioned in the source. Do not drop facts already modeled unless you must fix structure. Prefer **more** nodes over fewer. Use labels from the source text. Valid OPM triples only.

Source text:
{text}

Current diagram:
{json.dumps(diagram, ensure_ascii=False, indent=2)}

Return raw JSON only. No markdown."""


def _try_expand_diagram_once(
    text: str,
    diagram: dict,
    system_prompt: str,
) -> dict | None:
    """One expansion pass; returns None if parse/validate fails."""
    expand_prompt = _build_expand_user_prompt(text, diagram)
    t_exp = _expand_llm_temperature()
    try:
        raw = _call_llm_retry_502_once(system_prompt, expand_prompt, temperature=t_exp)
        normalized = normalize_llm_output(raw)
        data = parse_json(normalized)
        validate_diagram(data)
        return data
    except (ValueError, ValidationError, OPMExtractionError):
        return None


def _validation_repair_suffix(exc: Exception) -> str:
    return (
        "\n\n---\nYour previous JSON failed server validation:\n"
        f"{exc}\n\n"
        "Return one corrected JSON object only. Reminders:\n"
        "- `result`: only process→object (never to state).\n"
        "- `effect`: only process→state (never use `result` for a state target).\n"
        "- `agent` / `instrument` / `consumption`: **only object→process** (target must be `kind: \"process\"`). "
        "If the error says instrument/agent got object→object: your `target` id points to an object node — retarget to a process id "
        "(e.g. administration / storage / therapy process) or use `aggregation`/`specialization` between objects.\n"
        "- `characterization`: only object→state.\n"
        "- If the source text mentions several entities or steps, add those nodes and valid links—"
        "do not return only one main entity + one process + one outcome node.\n"
    )


def extract_opm_diagram(text: str) -> dict:
    """
    LLM-backed extraction: parse JSON, validate OPM rules; retry on bad JSON or validation errors.

    On validation failure, sends the error back to the model (bounded attempts).

    Environment (optional):
    - OPM_MAX_LLM_ROUNDS: max extraction/repair loops (default 4, max 20). Lower for faster failure on bad inputs.
    - OPM_TEMPERATURE: sampling temperature for the main pass (default 0). Try 0.15–0.3 if graphs stay too small.
    - OPM_AUTO_EXPAND=1: after a valid but tiny graph, one extra LLM pass to add nodes (see OPM_EXPAND_*).
    - OPM_EXPAND_MIN_CHARS (default 200), OPM_EXPAND_MAX_NODES (default 5): when to run auto-expand.
    - OPM_EXPAND_TEMPERATURE: temperature for expand pass (default max(OPM_TEMPERATURE, 0.15)).
    """
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(text)
    last_raw: str | None = None
    repair_suffix = ""
    prev_validation_err: str | None = None

    max_rounds = _max_llm_rounds()
    for attempt in range(max_rounds):
        combined = user_prompt + repair_suffix
        try:
            raw = _call_llm_retry_502_once(system_prompt, combined)
        except OPMExtractionError:
            raise
        last_raw = raw
        try:
            normalized = normalize_llm_output(raw)
            data = parse_json(normalized)
        except ValueError as exc:
            if str(exc) == "JSON must be object":
                raise OPMExtractionError(422, str(exc), last_raw, stage="validation") from exc
            repair_suffix = f"\n\nInvalid JSON ({exc}). Return one JSON object only.\n"
            if attempt == max_rounds - 1:
                raise OPMExtractionError(422, str(exc), last_raw, stage="validation") from exc
            continue
        try:
            validate_diagram(data)
            if _should_auto_expand(text, data):
                expanded = _try_expand_diagram_once(text, data, system_prompt)
                if expanded is not None:
                    return expanded
            return data
        except (ValidationError, ValueError) as exc:
            vmsg = str(exc)
            # Model often repeats the same invalid graph; do not burn the full round budget on identical errors.
            if prev_validation_err == vmsg and attempt >= 1:
                logger.warning(
                    "OPM validation error unchanged after a repair round; stopping early (%s/%s): %s",
                    attempt + 1,
                    max_rounds,
                    exc,
                )
                raise OPMExtractionError(
                    422,
                    humanize_diagram_validation(exc),
                    last_raw,
                    stage="validation",
                    technical_detail=vmsg,
                ) from exc
            prev_validation_err = vmsg
            repair_suffix = _validation_repair_suffix(exc)
            if attempt == max_rounds - 1:
                logger.warning("OPM diagram validation failed (final attempt): %s", exc)
                raise OPMExtractionError(
                    422,
                    humanize_diagram_validation(exc),
                    last_raw,
                    stage="validation",
                    technical_detail=vmsg,
                ) from exc
            continue

    raise OPMExtractionError(422, "Invalid JSON", last_raw, stage="validation")

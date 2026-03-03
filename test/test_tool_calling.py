"""
Unit tests for tool-calling: model must output a JSON tool call; we execute it
and compare result to the expected output of output_every_func_return_type(__file__).

Requires: ollama running locally with model llama3.1:8b, and optionally
K_SHOT_SYSTEM_PROMPT (or TOOL_CALLING_SYSTEM_PROMPT) describing the tool and format.
"""

import ast
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
from dotenv import load_dotenv

load_dotenv()

NUM_RUNS_TIMES = 3
YOUR_SYSTEM_PROMPT = os.getenv("TOOL_CALLING_SYSTEM_PROMPT", os.getenv("K_SHOT_SYSTEM_PROMPT", ""))


# ==========================
# Tool implementation (the "executor")
# ==========================


def _annotation_to_str(annotation: Optional[ast.AST]) -> str:
    if annotation is None:
        return "None"
    try:
        return ast.unparse(annotation)  # type: ignore[attr-defined]
    except Exception:
        if isinstance(annotation, ast.Name):
            return annotation.id
        return type(annotation).__name__


def _list_function_return_types(file_path: str) -> List[Tuple[str, str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    results: List[Tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return_str = _annotation_to_str(node.returns)
            results.append((node.name, return_str))
    results.sort(key=lambda x: x[0])
    return results


def output_every_func_return_type(file_path: str = None) -> str:
    """Tool: Return a newline-delimited list of "name: return_type" for each top-level function."""
    path = file_path or __file__
    if not os.path.isabs(path):
        candidate = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(candidate):
            path = candidate
    pairs = _list_function_return_types(path)
    return "\n".join(f"{name}: {ret}" for name, ret in pairs)


def add(a: int, b: int) -> int:
    return a + b


def greet(name: str) -> str:
    return f"Hello, {name}!"


TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "output_every_func_return_type": output_every_func_return_type,
}


# ==========================
# Helpers for model + execution
# ==========================


def resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    here = os.path.dirname(__file__)
    c1 = os.path.join(here, p)
    if os.path.exists(c1):
        return c1
    return p


def extract_tool_call(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json\n"):
            text = text[5:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("Model did not return valid JSON for the tool call")


def run_model_for_tool_call(system_prompt: str) -> Dict[str, Any]:
    from ollama import chat

    response = chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Call the tool now."},
        ],
        options={"temperature": 0.3},
    )
    content = response.message.content
    return extract_tool_call(content)


def execute_tool_call(call: Dict[str, Any]) -> str:
    name = call.get("tool")
    if not isinstance(name, str):
        raise ValueError("Tool call JSON missing 'tool' string")
    func = TOOL_REGISTRY.get(name)
    if func is None:
        raise ValueError(f"Unknown tool: {name}")
    args = call.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("Tool call JSON 'args' must be an object")

    if "file_path" in args and isinstance(args["file_path"], str):
        args["file_path"] = resolve_path(args["file_path"]) if str(args["file_path"]) != "" else __file__
    elif "file_path" not in args:
        args["file_path"] = __file__

    return func(**args)


def compute_expected_output() -> str:
    return output_every_func_return_type(__file__)


# ==========================
# Test
# ==========================


def test_your_prompt_tool_call():
    """
    Run up to NUM_RUNS_TIMES: get a JSON tool call from the model, execute it,
    and pass if the tool output matches the expected (this file's function return types).
    """
    if not YOUR_SYSTEM_PROMPT.strip():
        pytest.skip(
            "TOOL_CALLING_SYSTEM_PROMPT (or K_SHOT_SYSTEM_PROMPT) not set. "
            "Set in .env or environment to run this test."
        )

    expected = compute_expected_output()
    last_exc = None

    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        try:
            call = run_model_for_tool_call(YOUR_SYSTEM_PROMPT)
        except ImportError as e:
            pytest.skip("ollama package not installed")
        except Exception as exc:
            last_exc = exc
            print(f"Failed to parse tool call: {exc}")
            continue

        print(call)
        try:
            actual = execute_tool_call(call)
        except Exception as exc:
            last_exc = exc
            print(f"Tool execution failed: {exc}")
            continue

        if actual.strip() == expected.strip():
            print(f"Generated tool call: {call}")
            print(f"Generated output: {actual}")
            print("SUCCESS")
            assert actual.strip() == expected.strip()
            return

        print("Expected output:\n" + expected)
        print("Actual output:\n" + actual)

    if last_exc is not None:
        raise last_exc
    pytest.fail(
        f"None of {NUM_RUNS_TIMES} runs produced tool output matching expected."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

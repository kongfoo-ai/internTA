"""
Unit tests for k-shot prompting: reverse-word task via Ollama chat.

Requires: ollama running locally with model mistral-nemo:12b, and optionally
a non-empty YOUR_SYSTEM_PROMPT for the reverse-word instruction.
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()

NUM_RUNS_TIMES = 5
# TODO: Fill this in for the reverse-word task
# YOUR_SYSTEM_PROMPT = os.getenv("K_SHOT_SYSTEM_PROMPT", "")
YOUR_SYSTEM_PROMPT = """You are a word reversal assistant. Given a word, reverse the order of its letters.

Example 1:
Input: "hello"
Output: "olleh"

Example 2:
Input: "world"
Output: "dlrow"

Example 3:
Input: "python"
Output: "nohtyp"

For any word given, reverse all its letters and output only the reversed word."""

USER_PROMPT = """
Reverse the order of letters in the following word. Only output the reversed word, no other text:

httpstatus
"""

EXPECTED_OUTPUT = "sutatsptth"


def _run_prompt_once(system_prompt: str):
    """Call Ollama chat once; returns (output_text, None) or (None, error)."""
    try:
        from ollama import chat
    except ImportError:
        return None, ImportError("ollama package not installed")
    try:
        response = chat(
            model="mistral-nemo:12b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.5},
        )
        return (response.message.content or "").strip(), None
    except Exception as e:
        return None, e


def test_your_prompt_reverse_word():
    """
    Run the reverse-word prompt up to NUM_RUNS_TIMES; pass if any output
    matches EXPECTED_OUTPUT. Prints SUCCESS when a match is found.
    """
    if not YOUR_SYSTEM_PROMPT.strip():
        pytest.skip(
            "K_SHOT_SYSTEM_PROMPT not set. Set it in .env or environment to run this test."
        )

    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        output_text, err = _run_prompt_once(YOUR_SYSTEM_PROMPT)
        if err is not None:
            if isinstance(err, ImportError):
                pytest.skip("ollama package not installed")
            raise err
        if (output_text or "").strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            assert output_text.strip() == EXPECTED_OUTPUT.strip()
            return
        print(f"Expected output: {EXPECTED_OUTPUT}")
        print(f"Actual output: {output_text}")

    pytest.fail(
        f"None of {NUM_RUNS_TIMES} runs produced expected output {EXPECTED_OUTPUT!r}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

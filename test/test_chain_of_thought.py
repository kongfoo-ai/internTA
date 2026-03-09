"""
Unit tests for chain-of-thought style prompting: extract final answer and optional
Ollama integration (modular exponentiation 3^12345 mod 100).

Requires: ollama running locally with model llama3.1:8b for the integration test.
"""

import os
import re
import unittest
from dotenv import load_dotenv

load_dotenv()

NUM_RUNS_TIMES = 5

YOUR_SYSTEM_PROMPT = os.getenv(
    "COT_SYSTEM_PROMPT",
    (
        "You are a careful mathematical reasoning assistant.\n\n"
        "Solve the problem using step-by-step reasoning.\n\n"
        "Follow this structure:\n"
        "1. Understand the problem.\n"
        "2. Break it into smaller steps.\n"
        "3. Perform calculations carefully.\n"
        "4. Verify the result if possible.\n\n"
        "After finishing the reasoning, output the final answer on a new line "
        "using exactly this format:\n\n"
        "Answer: <number>"
    ),
)

USER_PROMPT = """
Solve this problem, then give the final answer on the last line as "Answer: <number>".

what is 3^{12345} (mod 100)?
"""

EXPECTED_OUTPUT = "Answer: 43"


def extract_final_answer(text: str) -> str:
    """Extract the final 'Answer: ...' line from a verbose reasoning trace.

    - Finds the LAST line that starts with 'Answer:' (case-insensitive)
    - Normalizes to 'Answer: <number>' when a number is present
    - Falls back to returning the matched content if no number is detected
    """
    matches = re.findall(r"(?mi)^\s*answer\s*:\s*(.+)\s*$", text)
    if matches:
        value = matches[-1].strip()
        num_match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if num_match:
            return f"Answer: {num_match.group(0)}"
        return f"Answer: {value}"
    return text.strip()


def run_cot_prompt(system_prompt: str) -> bool:
    """Run up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    try:
        from ollama import chat  # type: ignore[import-untyped]
    except ImportError:
        raise unittest.SkipTest("ollama package not installed")

    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.3},
        )
        output_text = response.message.content
        final_answer = extract_final_answer(output_text)
        if final_answer.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        print(f"Expected output: {EXPECTED_OUTPUT}")
        print(f"Actual output: {final_answer}")
    return False


class TestExtractFinalAnswer(unittest.TestCase):
    """Unit tests for extract_final_answer."""

    def test_single_answer_with_number(self):
        text = "Some reasoning.\nAnswer: 43"
        self.assertEqual(extract_final_answer(text), "Answer: 43")

    def test_multiple_answers_uses_last(self):
        text = "Step 1.\nAnswer: 99\nStep 2.\nAnswer: 43"
        self.assertEqual(extract_final_answer(text), "Answer: 43")

    def test_answer_case_insensitive(self):
        text = "Reasoning here.\nanswer: 43"
        self.assertEqual(extract_final_answer(text), "Answer: 43")

    def test_answer_with_extra_whitespace(self):
        text = "Reasoning.\n  Answer :  43  "
        self.assertEqual(extract_final_answer(text), "Answer: 43")

    def test_answer_negative_number(self):
        text = "Result.\nAnswer: -7"
        self.assertEqual(extract_final_answer(text), "Answer: -7")

    def test_answer_decimal(self):
        text = "Result.\nAnswer: 3.14"
        self.assertEqual(extract_final_answer(text), "Answer: 3.14")

    def test_answer_with_comma_in_number(self):
        text = "Result.\nAnswer: 1,234"
        self.assertEqual(extract_final_answer(text), "Answer: 1234")

    def test_no_answer_returns_stripped_text(self):
        text = "  Only reasoning, no Answer line.  "
        self.assertEqual(extract_final_answer(text), "Only reasoning, no Answer line.")

    def test_empty_string(self):
        self.assertEqual(extract_final_answer(""), "")

    def test_expected_output_format(self):
        text = "So the answer is 43.\nAnswer: 43"
        self.assertEqual(extract_final_answer(text).strip(), EXPECTED_OUTPUT.strip())


class TestChainOfThoughtPrompt(unittest.TestCase):
    """Integration test: run Ollama with USER_PROMPT and check final answer matches EXPECTED_OUTPUT."""

    def test_your_prompt_matches_expected(self):
        if not YOUR_SYSTEM_PROMPT.strip():
            self.skipTest(
                "COT_SYSTEM_PROMPT not set. Set it in .env or environment to run this test."
            )
        result = run_cot_prompt(YOUR_SYSTEM_PROMPT)
        self.assertTrue(result, f"None of {NUM_RUNS_TIMES} runs produced {EXPECTED_OUTPUT!r}")


if __name__ == "__main__":
    unittest.main()

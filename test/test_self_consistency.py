"""
Unit tests for self-consistency evaluation: answer extraction and majority-vote
over multiple model runs. Uses mocked Ollama chat for deterministic tests;
optional integration test can call real Ollama when run explicitly.
"""

import os
import re
import unittest
from collections import Counter
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

load_dotenv()

NUM_RUNS_TIMES = 5
EXPECTED_OUTPUT = "Answer: 25"

# TODO: Fill this in! Try to get as close to 100% correctness across all runs as possible.
YOUR_SYSTEM_PROMPT = ""

USER_PROMPT = """
Solve this problem, then give the final answer on the last line as "Answer: <number>".

Henry made two stops during his 60-mile bike trip. He first stopped after 20
miles. His second stop was 15 miles before the end of the trip. How many miles
did he travel between his first and second stops?
"""


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


def run_majority_vote(answers: list[str], expected: str) -> bool:
    """Given a list of extracted answers, return True if majority equals expected."""
    if not answers:
        return False
    counts = Counter(a.strip() for a in answers)
    majority_answer, _ = counts.most_common(1)[0]
    return majority_answer == expected.strip()


class TestExtractFinalAnswer(unittest.TestCase):
    """Unit tests for extract_final_answer()."""

    def test_single_answer_with_number(self):
        text = "So we add 20 and 5. The answer is 25.\nAnswer: 25"
        self.assertEqual(extract_final_answer(text), "Answer: 25")

    def test_verbose_reasoning_takes_last_answer(self):
        text = """
        First I thought 30. Then I recalculated.
        Answer: 30
        Actually 20 + 5 = 25.
        Answer: 25
        """
        self.assertEqual(extract_final_answer(text), "Answer: 25")

    def test_case_insensitive_answer_keyword(self):
        text = "So the result is 25.\nanswer: 25"
        self.assertEqual(extract_final_answer(text), "Answer: 25")

    def test_answer_with_whitespace_around_colon(self):
        # Regex requires line to start with optional whitespace then "answer"
        text = "So the result is 25.\n  Answer :  25"
        self.assertEqual(extract_final_answer(text), "Answer: 25")

    def test_number_with_decimal(self):
        text = "Answer: 25.0"
        self.assertEqual(extract_final_answer(text), "Answer: 25.0")

    def test_negative_number(self):
        text = "Answer: -10"
        self.assertEqual(extract_final_answer(text), "Answer: -10")

    def test_number_with_comma(self):
        text = "Answer: 1,000"
        self.assertEqual(extract_final_answer(text), "Answer: 1000")

    def test_no_answer_line_returns_stripped_text(self):
        text = "I think it's 25 miles."
        self.assertEqual(extract_final_answer(text), "I think it's 25 miles.")

    def test_empty_string(self):
        self.assertEqual(extract_final_answer(""), "")


class TestMajorityVote(unittest.TestCase):
    """Unit tests for majority-vote logic (no Ollama)."""

    def test_majority_matches_expected_success(self):
        answers = ["Answer: 25", "Answer: 25", "Answer: 25", "Answer: 30", "Answer: 20"]
        self.assertTrue(run_majority_vote(answers, EXPECTED_OUTPUT))

    def test_majority_differs_fails(self):
        answers = ["Answer: 30", "Answer: 30", "Answer: 25", "Answer: 20", "Answer: 20"]
        self.assertFalse(run_majority_vote(answers, EXPECTED_OUTPUT))

    def test_unanimous_correct(self):
        answers = ["Answer: 25"] * NUM_RUNS_TIMES
        self.assertTrue(run_majority_vote(answers, EXPECTED_OUTPUT))

    def test_empty_answers_fails(self):
        self.assertFalse(run_majority_vote([], EXPECTED_OUTPUT))

    def test_whitespace_normalized(self):
        answers = ["Answer: 25  ", "  Answer: 25", "Answer: 25"]
        self.assertTrue(run_majority_vote(answers, EXPECTED_OUTPUT))


try:
    import ollama as _ollama  # noqa: F401
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False


@unittest.skipUnless(_OLLAMA_AVAILABLE, "ollama package not installed")
class TestSelfConsistencyFlow(unittest.TestCase):
    """Integration-style tests with mocked Ollama chat."""

    @patch("ollama.chat")
    def test_flow_success_when_majority_correct(self, mock_chat):
        # Simulate 3/5 runs returning correct answer
        responses = [
            "Reasoning... Answer: 25",
            "Reasoning... Answer: 30",
            "Reasoning... Answer: 25",
            "Reasoning... Answer: 25",
            "Reasoning... Answer: 20",
        ]
        mock_chat.side_effect = [
            MagicMock(message=MagicMock(content=content)) for content in responses
        ]

        from ollama import chat

        answers = []
        for _ in range(NUM_RUNS_TIMES):
            resp = chat(
                model="llama3.1:8b",
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": USER_PROMPT},
                ],
                options={"temperature": 1},
            )
            final = extract_final_answer(resp.message.content)
            answers.append(final.strip())

        self.assertTrue(run_majority_vote(answers, EXPECTED_OUTPUT))
        self.assertEqual(mock_chat.call_count, NUM_RUNS_TIMES)

    @patch("ollama.chat")
    def test_flow_failure_when_majority_wrong(self, mock_chat):
        responses = ["Answer: 30"] * 3 + ["Answer: 25"] * 2
        mock_chat.side_effect = [
            MagicMock(message=MagicMock(content=content)) for content in responses
        ]

        from ollama import chat

        answers = []
        for _ in range(NUM_RUNS_TIMES):
            resp = chat(
                model="llama3.1:8b",
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": USER_PROMPT},
                ],
                options={"temperature": 1},
            )
            final = extract_final_answer(resp.message.content)
            answers.append(final.strip())

        self.assertFalse(run_majority_vote(answers, EXPECTED_OUTPUT))


if __name__ == "__main__":
    unittest.main()

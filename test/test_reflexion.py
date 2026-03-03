"""
Unit tests for reflexion flow: code extraction, loading, and evaluation of
generated is_valid_password implementations. Uses mocked Ollama chat for the
reflexion flow so tests are deterministic without a running Ollama service.
"""

import re
import unittest
from typing import Callable, List, Tuple
from unittest.mock import patch

# --- Logic under test (from reflexion script) ---

SPECIALS = set("!@#$%^&*()-_")
TEST_CASES: List[Tuple[str, bool]] = [
    ("Password1!", True),       # valid
    ("password1!", False),      # missing uppercase
    ("Password!", False),       # missing digit
    ("Password1", False),       # missing special
]


def extract_code_block(text: str) -> str:
    m = re.findall(r"```python\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip()
    m = re.findall(r"```\n([\s\S]*?)```", text)
    if m:
        return m[-1].strip()
    return text.strip()


def load_function_from_code(code_str: str) -> Callable[[str], bool]:
    namespace: dict = {}
    exec(code_str, namespace)  # noqa: S102
    func = namespace.get("is_valid_password")
    if not callable(func):
        raise ValueError("No callable is_valid_password found in generated code")
    return func


def evaluate_function(func: Callable[[str], bool]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    for pw, expected in TEST_CASES:
        try:
            result = bool(func(pw))
        except Exception as exc:
            failures.append(f"Input: {pw} → raised exception: {exc}")
            continue

        if result != expected:
            reasons = []
            if len(pw) < 8:
                reasons.append("length < 8")
            if not any(c.islower() for c in pw):
                reasons.append("missing lowercase")
            if not any(c.isupper() for c in pw):
                reasons.append("missing uppercase")
            if not any(c.isdigit() for c in pw):
                reasons.append("missing digit")
            if not any(c in SPECIALS for c in pw):
                reasons.append("missing special")
            if any(c.isspace() for c in pw):
                reasons.append("has whitespace")

            failures.append(
                f"Input: {pw} → expected {expected}, got {result}. "
                f"Failing checks: {', '.join(reasons) or 'unknown'}"
            )

    return (len(failures) == 0, failures)


# --- Correct implementation matching TEST_CASES / SPECIALS ---

def _reference_is_valid_password(password: str) -> bool:
    if len(password) < 8:
        return False
    if not any(c.islower() for c in password):
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    if not any(c in SPECIALS for c in password):
        return False
    if any(c.isspace() for c in password):
        return False
    return True


CORRECT_CODE = '''
SPECIALS = set("!@#$%^&*()-_")

def is_valid_password(password: str) -> bool:
    if len(password) < 8:
        return False
    if not any(c.islower() for c in password):
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    if not any(c in SPECIALS for c in password):
        return False
    if any(c.isspace() for c in password):
        return False
    return True
'''


# --- Tests ---

class TestExtractCodeBlock(unittest.TestCase):
    """Unit tests for extract_code_block()."""

    def test_python_fenced_block(self):
        text = "Here is the code:\n```python\ndef foo():\n    return 1\n```"
        self.assertEqual(extract_code_block(text), "def foo():\n    return 1")

    def test_python_fenced_case_insensitive(self):
        text = "```Python\ndef bar(): pass\n```"
        self.assertEqual(extract_code_block(text), "def bar(): pass")

    def test_takes_last_block_when_multiple(self):
        text = "```python\nfirst\n```\n\n```python\nsecond\n```"
        self.assertEqual(extract_code_block(text), "second")

    def test_generic_fenced_no_lang(self):
        text = "```\ndef baz(): return 0\n```"
        self.assertEqual(extract_code_block(text), "def baz(): return 0")

    def test_no_fence_returns_stripped_text(self):
        text = "  def qux(): return 2  "
        self.assertEqual(extract_code_block(text), "def qux(): return 2")

    def test_empty_content_strips(self):
        text = "```python\n   \n```"
        self.assertEqual(extract_code_block(text), "")


class TestLoadFunctionFromCode(unittest.TestCase):
    """Unit tests for load_function_from_code()."""

    def test_loads_valid_implementation(self):
        code = "def is_valid_password(password: str) -> bool:\n    return len(password) >= 8"
        func = load_function_from_code(code)
        self.assertTrue(callable(func))
        self.assertIs(func("Password1!"), True)
        self.assertIs(func("short"), False)

    def test_raises_when_function_missing(self):
        code = "x = 42"
        with self.assertRaises(ValueError) as ctx:
            load_function_from_code(code)
        self.assertIn("is_valid_password", str(ctx.exception))

    def test_raises_on_syntax_error(self):
        code = "def is_valid_password(:\n    return True"
        with self.assertRaises(SyntaxError):
            load_function_from_code(code)


class TestEvaluateFunction(unittest.TestCase):
    """Unit tests for evaluate_function()."""

    def test_correct_implementation_passes_all(self):
        passed, failures = evaluate_function(_reference_is_valid_password)
        self.assertTrue(passed, msg=failures)
        self.assertEqual(failures, [])

    def test_wrong_implementation_fails_with_diagnostics(self):
        def wrong(pw: str) -> bool:
            return len(pw) >= 8  # ignores upper/digit/special

        passed, failures = evaluate_function(wrong)
        self.assertFalse(passed)
        self.assertGreater(len(failures), 0)
        for f in failures:
            self.assertIn("expected", f)

    def test_exception_in_func_recorded_as_failure(self):
        def raises(pw: str) -> bool:
            raise RuntimeError("oops")

        passed, failures = evaluate_function(raises)
        self.assertFalse(passed)
        self.assertEqual(len(failures), len(TEST_CASES))
        self.assertTrue(any("raised exception" in f for f in failures))

    def test_test_cases_cover_all_four_inputs(self):
        self.assertEqual(len(TEST_CASES), 4)
        self.assertEqual(TEST_CASES[0], ("Password1!", True))
        self.assertEqual(TEST_CASES[1], ("password1!", False))
        self.assertEqual(TEST_CASES[2], ("Password!", False))
        self.assertEqual(TEST_CASES[3], ("Password1", False))


class TestEvaluateFunctionWithGeneratedCode(unittest.TestCase):
    """Integration-style tests using load_function_from_code + evaluate_function."""

    def test_correct_code_string_passes_evaluation(self):
        func = load_function_from_code(CORRECT_CODE)
        passed, failures = evaluate_function(func)
        self.assertTrue(passed, msg=failures)

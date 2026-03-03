"""
Unit tests for RAG flow and reflexion-style helpers from the provided scripts:
- RAG: load_corpus_from_files, make_user_prompt, extract_code_block, REQUIRED_SNIPPETS check.
- Reflexion helpers: extract_code_block, load_function_from_code, evaluate_function.
Uses temp files and mocks; no live Ollama required.
"""

import os
import re
import tempfile
import unittest
from typing import Callable, List, Tuple

# --- RAG script logic under test ---

def load_corpus_from_files(paths: List[str]) -> List[str]:
    corpus: List[str] = []
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    corpus.append(f.read())
            except Exception as exc:
                corpus.append(f"[load_error] {p}: {exc}")
        else:
            corpus.append(f"[missing_file] {p}")
    return corpus


def make_user_prompt(question: str, context_docs: List[str]) -> str:
    if context_docs:
        context_block = "\n".join(f"- {d}" for d in context_docs)
    else:
        context_block = "(no context provided)"
    return (
        f"Context (use ONLY this information):\n{context_block}\n\n"
        f"Task: {question}\n\n"
        "Requirements:\n"
        "- Use the documented Base URL and endpoint.\n"
        "- Send the documented authentication header.\n"
        "- Raise for non-200 responses.\n"
        "- Return only the user's name string.\n\n"
        "Output: A single fenced Python code block with the function and necessary imports.\n"
    )


REQUIRED_SNIPPETS = [
    "def fetch_user_name(",
    "requests.get",
    "/users/",
    "X-API-Key",
    "return",
]


def extract_code_block(text: str) -> str:
    m = re.findall(r"```python\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip()
    m = re.findall(r"```\n([\s\S]*?)```", text)
    if m:
        return m[-1].strip()
    return text.strip()


# --- Reflexion script logic under test ---

SPECIALS = set("!@#$%^&*()-_")
TEST_CASES: List[Tuple[str, bool]] = [
    ("Password1!", True),
    ("password1!", False),
    ("Password!", False),
    ("Password1", False),
]


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


# --- Tests: RAG ---


class TestLoadCorpusFromFiles(unittest.TestCase):
    """Unit tests for load_corpus_from_files()."""

    def test_reads_existing_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Base URL: https://api.example.com\n/users/{id}")
            path = f.name
        try:
            corpus = load_corpus_from_files([path])
            self.assertEqual(len(corpus), 1)
            self.assertIn("https://api.example.com", corpus[0])
            self.assertIn("/users/", corpus[0])
        finally:
            os.unlink(path)

    def test_missing_file_returns_placeholder(self):
        corpus = load_corpus_from_files(["/nonexistent/path/doc.txt"])
        self.assertEqual(len(corpus), 1)
        self.assertIn("[missing_file]", corpus[0])
        self.assertIn("doc.txt", corpus[0])

    def test_multiple_paths_mix_existing_and_missing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("content")
            path = f.name
        try:
            corpus = load_corpus_from_files([path, "/missing/a.txt"])
            self.assertEqual(len(corpus), 2)
            self.assertEqual(corpus[0], "content")
            self.assertIn("[missing_file]", corpus[1])
        finally:
            os.unlink(path)

    def test_empty_paths_returns_empty_list(self):
        self.assertEqual(load_corpus_from_files([]), [])


class TestMakeUserPrompt(unittest.TestCase):
    """Unit tests for make_user_prompt()."""

    def test_with_context_docs(self):
        question = "Write fetch_user_name(...)"
        context_docs = ["Doc1", "Doc2"]
        out = make_user_prompt(question, context_docs)
        self.assertIn("Context (use ONLY this information):", out)
        self.assertIn("- Doc1", out)
        self.assertIn("- Doc2", out)
        self.assertIn("Task: Write fetch_user_name(...)", out)
        self.assertIn("Requirements:", out)
        self.assertIn("Output: A single fenced Python code block", out)

    def test_without_context_shows_no_context_placeholder(self):
        out = make_user_prompt("Q?", [])
        self.assertIn("(no context provided)", out)
        self.assertIn("Task: Q?", out)


class TestExtractCodeBlock(unittest.TestCase):
    """Unit tests for extract_code_block() (shared by RAG and reflexion)."""

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


class TestRequiredSnippets(unittest.TestCase):
    """Unit tests for REQUIRED_SNIPPETS validation (snippet-in-code check)."""

    def test_code_with_all_snippets_passes(self):
        code = """
def fetch_user_name(user_id: str, api_key: str) -> str:
    r = requests.get("/users/" + user_id, headers={"X-API-Key": api_key})
    return r.json()["name"]
"""
        missing = [s for s in REQUIRED_SNIPPETS if s not in code]
        self.assertEqual(missing, [])

    def test_code_missing_snippet_fails(self):
        code = "def fetch_user_name(): return 'x'"
        missing = [s for s in REQUIRED_SNIPPETS if s not in code]
        self.assertIn("requests.get", missing)
        self.assertIn("/users/", missing)
        self.assertIn("X-API-Key", missing)


# --- Tests: Reflexion helpers ---


class TestLoadFunctionFromCode(unittest.TestCase):
    """Unit tests for load_function_from_code() (reflexion script)."""

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


class TestEvaluateFunction(unittest.TestCase):
    """Unit tests for evaluate_function() (reflexion script)."""

    def test_correct_implementation_passes_all(self):
        passed, failures = evaluate_function(_reference_is_valid_password)
        self.assertTrue(passed, msg=failures)
        self.assertEqual(failures, [])

    def test_wrong_implementation_fails_with_diagnostics(self):
        def wrong(pw: str) -> bool:
            return len(pw) >= 8

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


if __name__ == "__main__":
    unittest.main()

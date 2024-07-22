import unittest
from unittest.mock import patch
from generate_data import extract_atomic_claims, generate_questions
import re

def mock_extract_atomic_claims(entities):
    if "自适应元件" in entities:
        return ["自适应元件(adaptive generic parts)是合成生物学领域的一个重要的调控元件， 是为了协调其他元件的环境不稳定性，使整个合成生物学模块的代谢效率达到最高。"]
    return []

def mock_generate_questions(claims):
    if "自适应元件(adaptive generic parts)是合成生物学领域的一个重要的调控元件， 是为了协调其他元件的环境不稳定性，使整个合成生物学模块的代谢效率达到最高。" in claims:
        return ["什么是自适应元件？", "自适应元件是什么？"]
    return []

class TestGenerateData(unittest.TestCase):
    def setUp(self):
        self.entities = "自适应元件"
        self.expected_claims = ["自适应元件(adaptive generic parts)是合成生物学领域的一个重要的调控元件， 是为了协调其他元件的环境不稳定性，使整个合成生物学模块的代谢效率达到最高。"]
        self.expected_question_pattern = r"自适应元件.*\？"

    @patch('generate_data.extract_atomic_claims', side_effect=mock_extract_atomic_claims)
    @patch('generate_data.generate_questions', side_effect=mock_generate_questions)
    def test_claim_extraction_and_question_generation(self, mock_generate, mock_extract):
        # Test atomic claim extraction
        claims = extract_atomic_claims(self.entities)
        self.assertEqual(claims, self.expected_claims, "The extracted claims do not match the expected claims.")
        
        # Test question generation
        questions = generate_questions(self.expected_claims)
        
        self.assertTrue(any(re.match(self.expected_question_pattern, q) for q in questions), "The generated questions do not match the expected pattern.")

        mock_extract.assert_called_once_with(self.entities)
        mock_generate.assert_called_once_with(self.expected_claims)
    
    @patch('generate_data.extract_atomic_claims', side_effect=mock_extract_atomic_claims)
    @patch('generate_data.generate_questions', side_effect=mock_generate_questions)
    def test_empty_entities(self, mock_generate, mock_extract):
        entities = ""
        claims = extract_atomic_claims(entities)
        self.assertEqual(claims, [], "The extracted claims should be empty for empty entities.")
        
        questions = generate_questions(claims)
        self.assertEqual(questions, [], "The generated questions should be empty for empty claims.")
        
        mock_extract.assert_called_once_with(entities)
        mock_generate.assert_called_once_with([])

if __name__ == "__main__":
    unittest.main()
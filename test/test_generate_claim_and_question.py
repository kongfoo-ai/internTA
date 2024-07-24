'''
Unit tests for generate_claim module.
'''

import unittest
import os
from generate_claim import extract_atomic_claims, generate_questions
current_path = os.path.dirname(os.path.abspath(__file__))
main_folder_path = os.path.join(current_path, '..')

class TestGenerateData(unittest.TestCase):
    """Tests for the data generation functions in generate_claim module."""
    def setUp(self):
        self.entities = "自适应元件"
        self.text = (
            "自适应元件（adaptive generic parts）是合成生物学领域中至关重要的调控元件。"
            "它们的设计目的是为了应对和协调其他生物元件在不同环境条件下的表现和稳定性，从而确保整个合成生物学模块能够在各种外部变化中保持最佳的代谢效率。"
            "具体而言，自适应元件可以通过多种机制调节基因表达、蛋白质功能或代谢途径，以适应外界的温度、pH值、营养物质浓度等变化。"
            "这些元件的应用不仅可以提高生物系统的鲁棒性和灵活性，还能显著增强合成生物学装置在工业、医疗、农业等领域中的实际应用潜力。"
        )
        self.expected_claim_keywords = ["自适应元件", "调控元件", "效率"]
        self.expected_question_keywords = ["自适应元件", "？"]

    def test_claim_extraction_and_question_generation(self):
        """Test the claim extraction and question generation functions."""
        claims = extract_atomic_claims(self.text, self.entities)
        # claims contain expected keywords
        for keyword in self.expected_claim_keywords:
            self.assertTrue(
                any(keyword in claim for claim in claims),
                f"The extracted claims do not contain the expected keyword: {keyword}") 
        # claims should not be empty
        self.assertGreater(len(claims), 0, "Claims extraction resulted in no claims.")
        for claim in claims:
            self.assertTrue(
                isinstance(claim, str) and len(claim.strip()) > 0,
                f"Extracted claim is not valid: {claim}"
            )

        # Test question generation
        questions = generate_questions(claims)
        for keyword in self.expected_question_keywords:
            self.assertTrue(
                any(keyword in q for q in questions),
                f"The generated questions do not contain the expected keyword: {keyword}"
            )

        # generated questions should not be empty
        self.assertGreater(len(questions), 0, "Question generation resulted in no questions.")
        for question in questions:
            self.assertTrue(
                isinstance(question, str) and len(question.strip()) > 0,
                f"Generated question is not valid: {question}"
            )

    def test_empty_entities(self):
        """Test the functions with empty entities."""
        entities = ""
        claims = extract_atomic_claims('', entities)
        self.assertEqual(claims, [], "The extracted claims should be empty for empty entities.")
        questions = generate_questions(claims)
        self.assertEqual(questions, [], "The generated questions should be empty for empty claims.")

if __name__ == "__main__":
    unittest.main()

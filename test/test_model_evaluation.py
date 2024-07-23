from data.model_evaluation import process_data, get_model_answer, compute_similarity_rouge
import unittest
import pandas as pd
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
main_folder_path = os.path.join(current_path, '..')

class TestRougeEvaluation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.filename = os.path.join(main_folder_path, 'data/SynBio-Bench.json')
    
    def test_process_data(self):
        results = process_data(self.filename)

        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))  
        
        # Save results to a CSV file
        df_results.to_csv('test_results.csv', index=False)

        self.assertGreater(len(results), 0, "No results found. Check your input data.")
        
        for result in results:
            self.assertIn('topic', result)
            self.assertIn('correct', result)
            self.assertIn('incorrect', result)
            self.assertIn('percent_correct', result)
            self.assertGreaterEqual(result['correct'], 0)
            self.assertGreaterEqual(result['incorrect'], 0)
            self.assertGreaterEqual(result['percent_correct'], 0)
        first_topic = results[0]
        self.assertEqual(first_topic['correct'], 1)
        self.assertEqual(first_topic['incorrect'], 2)
        self.assertAlmostEqual(first_topic['percent_correct'], 33.333333, places=6)
           
    def test_get_model_answer(self):
        sample_prompt = "什么是合成生物学?"
        response = get_model_answer(sample_prompt)
        self.assertIsNotNone(response, "Model answer should not be None.")
        self.assertIsInstance(response, str, "Model answer should be a string.")

    def test_compute_similarity_rouge(self):
        reference = "合成生物学是一门新兴的学科，结合了生物学、工程学、计算机科学等领域的知识。"
        generated = "合成生物学是一门新兴的学科，结合了生物学和计算机科学等领域的知识。"
        
        scores = compute_similarity_rouge(reference, generated)
        
        self.assertEqual(len(scores), 9, "ROUGE scores should return a tuple with 9 elements.")
        self.assertTrue(all(score >= 0 for score in scores if score != -1.0), "ROUGE scores should be non-negative or -1.0 in case of error.")

if __name__ == "__main__":
    unittest.main()

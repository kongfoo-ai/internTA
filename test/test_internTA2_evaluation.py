import unittest
from unittest.mock import patch, MagicMock
from data.internTA2_evaluation_utils import process_data, llm_as_judge, get_answer
import os

current_path = os.path.dirname(os.path.abspath(__file__))
main_folder_path = os.path.join(current_path, '..')

class TestModelEvaluationUtils(unittest.TestCase):

    def test_process_data(self):
        # 测试 process_data 函数
        input_data = [{'answer': 'This is a test answer.</think> More text.'}]
        expected_output = [{'answer': 'More text.'}]
        self.assertEqual(process_data(input_data), expected_output)

    @patch('data.internTA2_evaluation_utils.requests.request')
    def test_llm_as_judge(self, mock_request):
        # 测试 llm_as_judge 函数
        mock_response = MagicMock()
        mock_response.json.return_value = {'choices': [{'message': {'content': 'Verdict: EQUIVALENT'}}]}
        mock_request.return_value = mock_response

        data = [{'question': 'What is DNA?', 'answer': 'DNA is...', 'solution': 'DNA is a molecule...'}]
        token = 'fake_token'
        acc, eq_count, diff_count, amb_count = llm_as_judge(data, token)

        self.assertEqual(eq_count, 1)
        self.assertEqual(diff_count, 0)
        self.assertEqual(amb_count, 0)

if __name__ == '__main__':
    unittest.main()
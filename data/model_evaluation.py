"""
This module provides functions to interact with internTA model API, compute ROUGE scores,
and process data from a JSON file.
"""

import json
import requests
import jieba
from rouge_chinese import Rouge

def get_model_answer(prompt):
    """
    Sends a prompt to the internTA API and retrieves the model's generated answer.
    """
    url = "http://i-2.gpushare.com:50259/api/generate"
    data = {
        "model": "kongfoo-16:latest",
        "prompt": prompt,
        "stream": False
    }
    json_data = json.dumps(data)
    try:
        response = requests.post(
            url, data=json_data, headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    response_data = response.json()
    return response_data.get('response', None)

def compute_similarity_rouge(reference, generated):
    """
    Computes ROUGE F1 scores to evaluate the similarity between reference and generated text.
    """
    if not generated:
        return -1.0, -1.0, -1.0  # F1 scores for rouge-1, rouge-2, and rouge-l
    reference = ' '.join(jieba.cut(reference))
    generated = ' '.join(jieba.cut(generated))
    rouge = Rouge()
    try:
        scores = rouge.get_scores(generated, reference)[0]
        return (
            scores['rouge-1']['f'],  # ROUGE-1 F1 score
            scores['rouge-2']['f'],  # ROUGE-2 F1 score
            scores['rouge-l']['f']   # ROUGE-L F1 score
        )
    except (ValueError, KeyError) as e:
        print(f"Error calculating ROUGE score: {e}")
        return -1.0, -1.0, -1.0

def process_data(filename):
    """
    Processes a JSON file containing topics and questions, 
    compares model answers with reference answers, 
    and calculates correctness percentages.
    """
    def process_topic(topic):
        """
        Processes a single topic to compute correctness statistics.
        """
        topic_name = topic['topic']
        correct_count = 0
        incorrect_count = 0
        for question in topic['questions']:
            reference_answers = topic['answers'].get(question, [])
            model_answer = get_model_answer(question)
            if model_answer:
                max_f1 = get_max_f1(reference_answers, model_answer)
                if max_f1 >= 0.5:  # Assume response is correct if max F1 score >= 0.5
                    correct_count += 1
                else:
                    incorrect_count += 1
            else:
                print(f"Failed to generate answer for question: {question}")
        percent_correct = calculate_percent_correct(correct_count, incorrect_count)
        return {
            'topic': topic_name,
            'correct': correct_count,
            'incorrect': incorrect_count,
            'percent_correct': percent_correct
        }

    def get_max_f1(reference_answers, model_answer):
        """
        Computes the maximum F1 score between the model answer and reference answers.
        """
        max_f1 = -1.0
        for reference_answer in reference_answers:
            rouge_scores = compute_similarity_rouge(reference_answer, model_answer)
            rouge1_f1, rouge2_f1, rouge_l_f1 = rouge_scores
            max_f1 = max(max_f1, rouge1_f1, rouge2_f1, rouge_l_f1)
        return max_f1

    def calculate_percent_correct(correct_count, incorrect_count):
        """
        Calculates the percentage of correct answers.
        """
        if correct_count + incorrect_count > 0:
            return (correct_count / (correct_count + incorrect_count)) * 100
        return 0.0

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = [process_topic(topic) for topic in data]

    return results

import json
import requests
import jieba
from rouge_chinese import Rouge
import pandas as pd

def get_model_answer(prompt):
    url = "http://i-2.gpushare.com:50259/api/generate"
    data = {
        "model": "kongfoo-16:latest",
        "prompt": prompt,
        "stream": False
    }
    json_data = json.dumps(data)
    try:
        response = requests.post(url, data=json_data, headers={"Content-Type": "application/json"})
        response.raise_for_status()  
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
    response_data = response.json()
    return response_data.get('response', None)

def compute_similarity_rouge(reference, generated):
    if not generated:
        return -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
    
    reference = ' '.join(jieba.cut(reference))
    generated = ' '.join(jieba.cut(generated))
    
    rouge = Rouge()
    
    try:
        scores = rouge.get_scores(generated, reference)[0]
        
        return (
            scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'],
            scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'],
            scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f']
        )
    except Exception as e:
        print(f"Error calculating ROUGE score: {e}")
        return -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0

def process_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for topic in data:
        topic_name = topic['topic']
        correct_count = 0
        incorrect_count = 0
        
        for question in topic['questions']:
            reference_answers = topic['answers'].get(question, [])
            model_answer = get_model_answer(question)
            
            if model_answer:
                max_f1 = -1.0  # Keep track of the maximum F1 score across all reference answers
                max_rouge1_f1 = -1.0
                max_rouge2_f1 = -1.0
                max_rougeL_f1 = -1.0

                for reference_answer in reference_answers:
                    rouge1_p, rouge1_r, rouge1_f1, rouge2_p, rouge2_r, rouge2_f1, rougeL_p, rougeL_r, rougeL_f1 = compute_similarity_rouge(reference_answer, model_answer)
                    
                    max_rouge1_f1 = max(max_rouge1_f1, rouge1_f1)
                    max_rouge2_f1 = max(max_rouge2_f1, rouge2_f1)
                    max_rougeL_f1 = max(max_rougeL_f1, rougeL_f1)

                    max_f1 = max(max_f1, max_rouge1_f1, max_rouge2_f1, max_rougeL_f1)
                
                if max_f1 >= 0.5:  # Assume response is correct if max F1 score >= 0.5
                    correct_count += 1
                else:
                    incorrect_count += 1
            else:
                print(f"Failed to generate answer for question: {question}")
        
        if correct_count + incorrect_count > 0:
            percent_correct = (correct_count / (correct_count + incorrect_count)) * 100
        else:
            percent_correct = 0.0
        
        results.append({
            'topic': topic_name,
            'correct': correct_count,
            'incorrect': incorrect_count,
            'percent_correct': percent_correct
        })
    
    return results

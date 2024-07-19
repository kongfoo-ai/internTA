import json
import requests
import jieba
from rouge_chinese import Rouge

def get_model_answer(prompt):
    url = "http://i-2.gpushare.com:50259/api/generate"
    data = {
        "model": "kongfoo-16:latest",
        "prompt": prompt,
        "stream": False
    }
    json_data = json.dumps(data)
    response = requests.post(url, data=json_data, headers={"Content-Type": "application/json"})
    
    if response.status_code == 200:
        response_data = response.json()
        return response_data.get('response', None)
    else:
        print(f"Failed to connect. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return None

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
    
    for topic in data:
        topic_name = topic['topic']
        totals = {
            'f1': 0,
            'p1': 0,
            'r1': 0,
            'f2': 0,
            'p2': 0,
            'r2': 0,
            'fL': 0,
            'pL': 0,
            'rL': 0
        }
        num_questions = 0
        
        for question in topic['questions']:
            reference_answers = topic['answers'].get(question, [])
            model_answer = get_model_answer(question)
            
            if model_answer:
                accumulators = {
                    'f1': 0,
                    'p1': 0,
                    'r1': 0,
                    'f2': 0,
                    'p2': 0,
                    'r2': 0,
                    'fL': 0,
                    'pL': 0,
                    'rL': 0
                }
                
                for reference_answer in reference_answers:
                    p1, r1, f1, p2, r2, f2, pL, rL, fL = compute_similarity_rouge(reference_answer, model_answer)
                    accumulators['p1'] += p1
                    accumulators['r1'] += r1
                    accumulators['f1'] += f1
                    accumulators['p2'] += p2
                    accumulators['r2'] += r2
                    accumulators['f2'] += f2
                    accumulators['pL'] += pL
                    accumulators['rL'] += rL
                    accumulators['fL'] += fL

                num_references = len(reference_answers)
                averages = {key: accumulators[key] / num_references for key in accumulators}
                
                f_score = (averages['f1'] + averages['f2'] + averages['fL']) / 3
                p_score = (averages['p1'] + averages['p2'] + averages['pL']) / 3
                r_score = (averages['r1'] + averages['r2'] + averages['rL']) / 3
                
                totals['f1'] += f_score
                totals['p1'] += p_score
                totals['r1'] += r_score
                num_questions += 1
            else:
                print(f"Failed to generate answer for question: {question}")
        
        if num_questions > 0:
            average_scores = {key: totals[key] / num_questions for key in totals}
        else:
            average_scores = {key: 0 for key in totals}
        
        print(f"Similarity scores for topic '{topic_name}': F1 = {average_scores['f1']:.2f}, Precision = {average_scores['p1']:.2f}, Recall = {average_scores['r1']:.2f}")

if __name__ == "__main__":
    process_data('SynBio-Bench.json')

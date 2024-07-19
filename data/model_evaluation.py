import json
import requests
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import numpy as np

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
        if 'response' in response_data:
            return response_data['response']
        else:
            print("Response does not contain 'response' key.")
            return None
    else:
        print("Failed to connect. Status code:", response.status_code)
        print("Response:", response.text)
        return None

def compute_similarity_tfidf(reference, generated):
    try:
        words1 = jieba.lcut(reference)
        words2 = jieba.lcut(generated)
        
        processed_sentence1 = " ".join(words1)
        processed_sentence2 = " ".join(words2)
        
        corpus = [processed_sentence1, processed_sentence2]
        
        vectorizer = TfidfVectorizer(stop_words=['的', '了', '和', '是', '就', '都', '而', '及', '与', '或', '之'])
        
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        
        normalized_similarity = (cosine_sim + 1) / 2
        
        return normalized_similarity
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0

with open('SynBio-Bench.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

table_data = []

for topic in data['topics']:
    topic_name = topic['topic']
    total_score = 0
    num_questions = 0
    
    for question in topic['questions']:
        reference_answer = topic['answers'][question][0] 
        
        model_answer = get_model_answer(question)
        
        if model_answer:
            score = compute_similarity_tfidf(reference_answer, model_answer)
            total_score += score
            num_questions += 1
            table_data.append([topic_name, question, reference_answer, model_answer, f"{score:.2f}"])
        else:
            print(f"Failed to generate answer for question: {question}")
    
    if num_questions > 0:
        average_score = total_score / num_questions
    else:
        average_score = 0
    
    print(f"Average similarity score for topic '{topic_name}': {average_score:.2f}")
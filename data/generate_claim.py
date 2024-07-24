"""
This module processes entities from a JSON file and extracts atomic claims from 
textbook documents in DOCX format. It then generates questions and answers based 
on the extracted claims using the OpenAI GPT-3.5-turbo model and saves the results 
in a JSON file.
"""

import os
import json
import openai
from docx import Document
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL")
openai.default_headers = {"x-foo": "true"}

def read_docx_files(docx_folder):
    """
    从文件中读取textbook内容。
    """
    combined_text = ""
    for filename in os.listdir(docx_folder):
        if filename.endswith(".docx") and not filename.startswith("~$"):
            doc_path = os.path.join(docx_folder, filename)
            try:
                doc = Document(doc_path)
                for para in doc.paragraphs:
                    combined_text += para.text + "\n"
            except FileNotFoundError as e:
                print(f"文件 {doc_path} 未找到: {e}")
            except OSError as e:
                print(f"读取 {doc_path} 时出错，操作系统错误: {e}")
            except ValueError as e:
                print(f"读取 {doc_path} 时出错，值错误: {e}")
    return combined_text

def split_text(text, max_length=4000):
    """
    将长文本拆分成多个较小的文本块。
    """
    words = text.split()
    chunks = []
    chunk = []
    length = 0

    for word in words:
        length += len(word) + 1
        if length > max_length:
            chunks.append(" ".join(chunk))
            chunk = []
            length = len(word) + 1
        chunk.append(word)

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

def extract_atomic_claims(text, entity):
    """
    使用 API 提取与 entity 相关的atomic claim。
    """
    if not entity:
        return []

    claims = []
    text_chunks = split_text(text)

    for chunk in text_chunks:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"从以下文本中提取与关键词 '{entity}' 相关的知识点，"
                               f"并在找到相关定义后停止处理，确保知识点包含关键词：\n\n{chunk}"
                },
            ],
        )
        response = completion.choices[0].message.content

        if response:
            claims.extend([
                    claim.strip() for claim in response.strip().split('\n') if claim.strip()
                ])
        else:
            print("接收到的响应为空")
    return list(set(claims))

def generate_questions(claims):
    """
    将atomic claim转换为问题。
    """
    if not claims:
        return []

    questions = []
    for claim in claims:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"将以下知识点直接转换为测试问题，返回独立的问题：\n\n{claim}"
                },
            ],
        )
        response = completion.choices[0].message.content

        if response:
            questions.extend([q.strip() for q in response.strip().split('\n')
                              if q.strip() and not q.strip().startswith('单独的知识测试问题：')])
        else:
            print("接收到的响应为空")
    return questions[:3]

def generate_answers(question, claims, num_answers=3):
    """
    根据atomic claim生成多个reference answers
    """
    answers = []
    for _ in range(num_answers):
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"根据以下知识点提供答案：\n\n问题: {question}\n\n知识点:\n"
                               f"{', '.join(claims)}\n\n请确保答案直接回答问题，避免重复和无关的内容。"
                },
            ],
        )
        response = completion.choices[0].message.content

        if response:
            answers.append(response.strip())
        else:
            print("接收到的响应为空")
    return answers[:3]

def process_entities(entities_filename, docx_folder):
    """
    提取claim，生成问题和答案。
    """
    entities = read_entities(entities_filename)
    textbook = read_docx_files(docx_folder)

    synbio_bench = []
    topic_id = 1

    for entity in entities:
        # 提取与entity相关的atomic claim
        claims = extract_atomic_claims(textbook, entity)

        # 生成问题
        questions = generate_questions(claims)

        topic_data = {
            "topic": entity,
            "topic_id": topic_id,
            "questions": questions,
            "answers": {}
        }

        for question in questions:
            topic_data["answers"][question] = generate_answers(question, claims)

        synbio_bench.append(topic_data)
        topic_id += 1

    return synbio_bench

def read_entities(filename):
    """
    从 JSON 文件中读取entities。
    """
    with open(filename, 'r', encoding='utf-8') as json_file:
        entities = json.load(json_file)
    return entities

if __name__ == "__main__":
    ENTITIES_FILENAME = 'entities.json'
    DOCX_FOLDER = '合成生物第七章'

    SYNBIO_BENCH = process_entities(ENTITIES_FILENAME, DOCX_FOLDER)

    with open('SynBio-Bench.json', 'w', encoding='utf-8') as file:
        json.dump(SYNBIO_BENCH, file, ensure_ascii=False, indent=4)

    print("SynBio-Bench JSON 文件已生成。")

import json
import requests
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

def get_answer(prompt, fine_tuned_model, tokenizer, max_seq_length=8000, load_in_4bit=True):
    import os
    from datetime import datetime
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # fine_tuned_model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=load_in_4bit, device_map="auto")

    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Prepare the model for inference
    fine_tuned_model.eval()

    # prompt = f"<|start_header_id|>system<|end_header_id|> Please reason step by step .<|eot_id|><|start_header_id|> User: <|end_header_id|>{prompt}<|eot_id|>\n\n"
    prompt = f"<|im_start|>system\nPlease reason step by step.\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

    # Tokenizing the input and generating the output
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = fine_tuned_model.generate(**inputs, max_new_tokens=max_seq_length, use_cache=True)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # 去掉 prompt 部分
    answer = answer.replace(prompt, "").strip()

    return answer

def inference_with_latest_model(model_path, max_seq_length=8000, valid_dataset_path='valid_dataset.json'):
    # 新增导入
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import os
    import json
    from tqdm import tqdm

    # 加载模型
    load_in_4bit = True
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    if model_path:
        print(f"最新的模型目录是: {model_path}")
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=load_in_4bit, bnb_4bit_quant_type="nf4", device_map="auto")
    else:
        print("未找到符合条件的模型目录。")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    fine_tuned_model.eval()

    # 加载 valid_dataset.json
    with open(valid_dataset_path, 'r', encoding='utf-8') as f:
        valid_data = json.load(f)

    results = []  # 用于存储结果
    for index, item in tqdm(enumerate(valid_data)):
        question = item['prompt']  # 假设问题在 'prompt' 键下
        solution = item['solution']  # 假设解决方案在 'solution' 键下

        # 使用 get_answer 函数生成答案
        answer = get_answer(question, fine_tuned_model, tokenizer, max_seq_length)

        # 保存结果
        results.append({
            "question": question,
            "solution": solution,
            "answer": answer
        })

    # 将结果写入 JSON 文件
    output_file_path = os.path.join(model_path, f"results_valid_{max_seq_length}.json")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"结果已保存到: {output_file_path}")

def process_data(data):
    for item in data:
        answer = item.get('answer', '')
        last_think_index = answer.rfind('</think>')
        item['answer'] = answer[last_think_index + len('</think>'):].strip() if last_think_index != -1 else answer
    return data


def llm_as_judge(data, token, model="deepseek-ai/DeepSeek-V3", url="https://api.siliconflow.cn/v1/chat/completions", output_path="all_scores.json", error_path='error_list.json'):
    all_responses = []

    # Iterate through each element in data
    for idx in range(len(data)):
        question = data[idx]['question']
        answer = data[idx]['answer']
        solution = data[idx]['solution']

        # Construct evaluation prompt
        prompt = f"""
        You are a synthetic biology answer validator. You will be provided with a synthetic biology problem, and you need to compare the answer in the reference solution and the final answer in the model's solution while also evaluating whether the model's reasoning process is correct. Your evaluation should consider both the equivalence of the final answer and the soundness of the reasoning process.

        PROBLEM:

        {question}

        REFERENCE SOLUTION:

        {solution}

        MODEL'S SOLUTION:

        {answer}

        If the model's answer is nonsensical or its reasoning is flawed, return "Verdict: AMBIGUOUS".

        Start with a brief explanation of your comparison (2-3 sentences). Then output your final answer in one of the following formats:

        "Verdict: EQUIVALENT"
        "Verdict: DIFFERENT"
        "Verdict: AMBIGUOUS"
        """

        url = url

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "n": 1,
            "stream": False
        }
        headers = {
            "Authorization": token,
            "Content-Type": "application/json"
        }

        # Check if response_file exists and is not empty
        try:
            response = requests.request("POST", url, json=payload, headers=headers)
            all_responses.append(response.json())
        except Exception as e:
            print(f"Request failed: {e}")

    # Save all responses to a JSON file
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(all_responses, json_file, ensure_ascii=False, indent=4)

    # with open('all_scores.json', "r", encoding="utf-8") as file:
    #     all_scores = json.load(file)

    all_scores = all_responses

    # with open('processed_answers.json', "r", encoding="utf-8") as file:
    #     all_ques_ans = json.load(file)

    all_ques_ans = data

    # Initialize counters
    equivalent_count = 0
    different_count = 0
    ambiguous_count = 0

    # Initialize error list
    list_of_errors = []

    # Iterate through all responses
    for idx, response in enumerate(all_scores):
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            
            # Count each verdict
            if "Verdict: EQUIVALENT" in content:
                equivalent_count += 1
            elif "Verdict: DIFFERENT" in content:
                different_count += 1
                list_of_errors.append(all_ques_ans[idx])
            elif "Verdict: AMBIGUOUS" in content:
                ambiguous_count += 1

    # Save error list to list_of_errors.json file
    with open(error_path, "w", encoding="utf-8") as json_file:
        json.dump(list_of_errors, json_file, ensure_ascii=False, indent=4)

    acc = equivalent_count/(equivalent_count+different_count+ambiguous_count)

    # Print statistics
    return acc, equivalent_count, different_count, ambiguous_count
    # print(f"EQUIVALENT: {equivalent_count}, DIFFERENT: {different_count}, AMBIGUOUS: {ambiguous_count}, acc: {equivalent_count/(equivalent_count+different_count+ambiguous_count)}")

def main():
    import argparse

    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Test the model evaluation utility.')
    parser.add_argument('--input_file_path', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--llm_token', type=str, required=True, help='Token for the model')
    parser.add_argument('--output_path', type=str, default='all_scores.json', help='Path to the output JSON file.')
    parser.add_argument('--llm_url', type=str, default="https://api.siliconflow.cn/v1/chat/completions", help='Link to do LLM-as-a-judge')
    parser.add_argument('--model_choice', type=str, default="deepseek-ai/DeepSeek-V3", help='Model name to do LLM-as-a-judge')
    parser.add_argument('--error_path', type=str, default='list_of_errors.json', help='Path to save error exercises')
    args = parser.parse_args()

    # 读取 JSON 文件
    with open(args.input_file_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    # 处理每个项目的答案
    data = process_data(data)

    # 调用 llm_as_judge 函数
    acc, equivalent_count, different_count, ambiguous_count = llm_as_judge(data, args.llm_token, args.model_choice, args.llm_url, args.output_path, args.error_path)

    # 打印统计信息
    print(f"EQUIVALENT: {equivalent_count}, DIFFERENT: {different_count}, AMBIGUOUS: {ambiguous_count}, acc: {acc}")

if __name__ == "__main__":
    main()
"""
InternTA API Server
==================

English:
--------
This file implements a REST API server for the InternTA (Synthetic Biology Teaching Assistant) 
based on the DeepSeek-R1-Distill-Qwen-7B large language model with QLoRA fine-tuning.

The API server provides a OpenAI-compatible chat completions endpoint (/v1/chat/completions) 
that can be used to interact with the InternTA model. The server loads a 4-bit quantized version 
of the model to reduce memory usage while maintaining performance.

Key components:
- FastAPI framework for the REST API
- Hugging Face Transformers for model loading and inference
- PEFT (Parameter-Efficient Fine-Tuning) for loading the QLoRA adapter
- BitsAndBytes for 4-bit quantization

Chinese:
--------
此文件实现了基于 DeepSeek-R1-Distill-Qwen-7B 大语言模型并通过 QLoRA 微调的合成生物学助教 InternTA 的 REST API 服务器。

API 服务器提供了与 OpenAI 兼容的聊天完成端点 (/v1/chat/completions)，可用于与 InternTA 模型进行交互。
服务器加载模型的 4 位量化版本，以减少内存使用同时保持性能。

主要组件：
- FastAPI 框架用于 REST API
- Hugging Face Transformers 用于模型加载和推理
- PEFT (参数高效微调) 用于加载 QLoRA 适配器
- BitsAndBytes 用于 4 位量化
"""

import os
import copy
import warnings
from dataclasses import asdict, dataclass
from typing import List, Optional
from dotenv import load_dotenv

import streamlit as st
import torch
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from modelscope import snapshot_download, AutoModel, AutoTokenizer
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import uvicorn
from peft import PeftModel


logger = logging.get_logger(__name__)

# Load environment variables from .env file
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN", "")  # Default to empty string if not found

def init():
    # 设置 HF 镜像环境变量（如果需要）
    model_dir = snapshot_download('Kongfoo-ai/internTAv2.0_test', cache_dir='./')
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.system('huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir DeepSeek-R1-Distill-Qwen-7B --cache-dir DeepSeek-R1-Distill-Qwen-7B')

@dataclass
class GenerationConfig:
    # 用于对话生成的配置
    max_length: int = 32768
    max_new_tokens: Optional[int] = 8000  # 新增生成新token的最大数量
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005

    def update(self, **kwargs):
        config = asdict(self)
        config.update(kwargs)
        # 移除不必要的键
        config.pop("cache_position", None)
        return config


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    **kwargs,
):
    # 将 prompt 编码为输入张量，同时生成 attention_mask
    inputs = tokenizer([prompt], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()

    if generation_config is None:
        # 若模型本身有 generation_config，则使用
        generation_config = model.generation_config
    # 合并生成配置与额外参数
    gen_kwargs = generation_config.update(**kwargs)
    # 如果没有设置 pad_token_id，则设置为 tokenizer.pad_token_id，否则为 tokenizer.eos_token_id
    if "pad_token_id" not in gen_kwargs or gen_kwargs["pad_token_id"] is None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # 调用 generate 方法，直接生成完整回答，注意将 max_new_tokens 参数传入
    outputs = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
    # 解码生成的 token，去除特殊符号
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 如果生成结果包含输入 prompt，可根据需要进行切分
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]
    return generated_text


@st.cache_resource
def load_model():
    # ========== 量化配置 (支持 4-bit 量化) ==========
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # ========== 加载基础模型 ==========
    base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,  # 4-bit 量化
        device_map="auto",
        trust_remote_code=True
    )
    # ========== 加载 QLoRA 适配器 ==========
    lora_adapter_path = "Kongfoo-ai/internTAv2.0_test"
    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    # ========== 加载 Tokenizer ==========
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
    return lora_model, tokenizer


user_prompt = "<|im_start|>user\n{user}<|im_end|>\n"
robot_prompt = "<|im_start|>assistant\n{robot}<|im_end|>\n"
cur_query_prompt = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


# 定义 API 请求数据结构
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]

class ChatCompletionResponse(BaseModel):
    choices: List[dict]


# 创建 FastAPI app
app_api = FastAPI(title="InternTA Chat Completions API")


@st.cache_resource
def get_model_and_tokenizer():
    return load_model()


# Authentication dependency
async def verify_token(authorization: str = Header(None)):
    if not API_TOKEN:
        # If no token is set in .env, skip authentication
        return True
    
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401, 
                detail="Invalid authentication scheme",
                headers={"WWW-Authenticate": "Bearer"}
            )
        if token != API_TOKEN:
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"}
            )
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return True


@app_api.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        model, tokenizer = get_model_and_tokenizer()
        # 格式化消息，构造 prompt
        total_prompt = "<s>"
        for msg in request.messages:
            if msg.role == "system":
                total_prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
            elif msg.role == "user":
                total_prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
            elif msg.role == "assistant":
                total_prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
        total_prompt += "<|im_start|>assistant\n"

        generation_config = GenerationConfig()
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=total_prompt,
            use_cache=False,  # 关闭缓存
            **asdict(generation_config)
        )
        return ChatCompletionResponse(
            choices=[{
                "message": {
                    "role": "assistant",
                    "content": response
                }
            }]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import sys
    uvicorn.run(app_api, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    init()
    main()

"""
This script refers to the dialogue example of streamlit, the interactive generation code of chatglm2 and transformers.
We mainly modified part of the code logic to adapt to the generation of our model.
Please refer to these links below for more information:
    1. streamlit chat example: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    2. chatglm2: https://github.com/THUDM/ChatGLM2-6B
    3. transformers: https://github.com/huggingface/transformers
Please run with the command `streamlit run path/to/web_demo.py --server.address=0.0.0.0 --server.port 7860`.
Using `python path/to/web_demo.py` may cause unknown problems.
"""

import os
import copy
import warnings
from dataclasses import asdict, dataclass
from typing import List, Optional

import streamlit as st
import torch
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from modelscope import snapshot_download, AutoModel, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from peft import PeftModel


logger = logging.get_logger(__name__)

def init():
    # 设置 HF 镜像环境变量（如果需要）
    model_dir = snapshot_download('Kongfoo-ai/internTAv2.0_test', cache_dir='./')
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.system('huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir DeepSeek-R1-Distill-Qwen-7B')

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


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider("Max Length", min_value=8, max_value=32768, value=32768)
        # 增加 max_new_tokens 的设置，比如默认生成 256 个新 token
        max_new_tokens = st.slider("Max New Tokens", min_value=1, max_value=4096, value=256)
        top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01)
        st.button("Clear Chat History", on_click=lambda: st.session_state.pop("messages", None))
    generation_config = GenerationConfig(
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        temperature=temperature
    )
    return generation_config


user_prompt = "<|im_start|>user\n{user}<|im_end|>\n"
robot_prompt = "<|im_start|>assistant\n{robot}<|im_end|>\n"
cur_query_prompt = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def combine_history(prompt):
    messages = st.session_state.messages
    meta_instruction = (
        "你是一个由躬富科技提供支持开发的聊天机器人E.CoPI。现在你是我的助教，我有一些关于学习《合成生物学》课本的问题，请你用专业的知识帮我解决。"
    )
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    for message in messages:
        if message["role"] == "user":
            total_prompt += user_prompt.format(user=message["content"])
        elif message["role"] == "robot":
            total_prompt += robot_prompt.format(robot=message["content"])
    total_prompt += cur_query_prompt.format(user=prompt)
    return total_prompt


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


@app_api.post("/v1/chat/completions")
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
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        # 运行 FastAPI 服务器
        uvicorn.run(app_api, host="0.0.0.0", port=8000)
    else:
        # 运行 Streamlit 界面
        print("load model begin.")
        model, tokenizer = load_model()
        print("load model end.")

        user_avator = "statics/momo.png"
        robot_avator = "statics/robot.png"

        st.title("我是E.CoPI老师，你的《合成生物学》助教~")

        generation_config = prepare_generation_config()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message.get("avatar")):
                st.markdown(message["content"])

        if prompt := st.chat_input("What is up?"):
            with st.chat_message("user", avatar=user_avator):
                st.markdown(prompt)
            real_prompt = combine_history(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt, "avatar": user_avator})

            with st.chat_message("robot", avatar=robot_avator):
                response = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    **asdict(generation_config)
                )
                st.markdown(response)
            st.session_state.messages.append({
                "role": "robot",
                "content": response,
                "avatar": robot_avator,
            })
            torch.cuda.empty_cache()


if __name__ == "__main__":
    init()
    main()

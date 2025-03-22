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
    # Set the HF mirror environment variable (if needed)
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    model_dir = snapshot_download('Kongfoo-ai/internTAv2.0_test', cache_dir='./')
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.system('huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir DeepSeek-R1-Distill-Qwen-7B')

@dataclass
class GenerationConfig:
    # Configuration for dialogue generation
    max_length: int = 32768
    max_new_tokens: Optional[int] = None  # Maximum number of new tokens to generate
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005

    def update(self, **kwargs):
        config = asdict(self)
        config.update(kwargs)
        # Remove unnecessary keys
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
    # Encode the prompt into input tensors and generate attention_mask
    inputs = tokenizer([prompt], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()

    if generation_config is None:
        # If the model has its own generation_config, use it
        generation_config = model.generation_config
    # Combine generation configuration with additional arguments
    gen_kwargs = generation_config.update(**kwargs)
    # If pad_token_id is not set, set it to tokenizer.pad_token_id or tokenizer.eos_token_id
    if "pad_token_id" not in gen_kwargs or gen_kwargs["pad_token_id"] is None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Call the generate method to generate a full response
    outputs = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
    # Decode the generated tokens, skipping special tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # If the generated text starts with the input prompt, trim it
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]
    generated_text = generated_text.replace("\\\\(", "$").replace("\\\\)", "$").replace("\\(", "$").replace("\\)", "$")
    generated_text = generated_text.replace("\\\\[", "$").replace("\\\\]", "$").replace("\\[", "$").replace("\\]", "$")
    return generated_text


@st.cache_resource
def load_model():
    # ========== Quantization Configuration (Supports 4-bit Quantization) ==========
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # ========== Load Base Model ==========
    base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,  # 4-bit quantization
        device_map="auto",
        trust_remote_code=True
    )
    # ========== Load QLoRA Adapter ==========
    lora_adapter_path = "Kongfoo-ai/internTAv2.0_test"
    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    # ========== Load Tokenizer ==========
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
    return lora_model, tokenizer


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider("Max Length", min_value=8, max_value=32768, value=32768)
        # Add a setting for max_new_tokens, e.g., default to 256 new tokens
        max_new_tokens = st.slider("Max New Tokens", min_value=1, max_value=8192, value=8192)
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
        "You are a chatbot powered by Gongfu Technology, E.CoPI. You are my teaching assistant, and I have some questions about the textbook 'Synthetic Biology'. Please help me with your professional knowledge."
    )
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    for message in messages:
        if message["role"] == "user":
            total_prompt += user_prompt.format(user=message["content"])
        elif message["role"] == "robot":
            total_prompt += robot_prompt.format(robot=message["content"])
    total_prompt += cur_query_prompt.format(user=prompt)
    return total_prompt


# Define API request data structure
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]

class ChatCompletionResponse(BaseModel):
    choices: List[dict]


# Create FastAPI app
app_api = FastAPI(title="InternTA Chat Completions API")


@st.cache_resource
def get_model_and_tokenizer():
    return load_model()


@app_api.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        model, tokenizer = get_model_and_tokenizer()
        # Format the messages and construct the prompt
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
            use_cache=False,  # Disable cache
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


def display_thinking_and_answer(generated_text):
    # If the generated text contains </think>, split the thinking part and the answer part
    if '</think>' in generated_text:
        # Split at the last occurrence of </think>
        last_think_pos = generated_text.rfind('</think>')

        # Get the part before the last </think>
        thinking_part = generated_text[:last_think_pos]
        # Get the part after the last </think>
        answer_part = generated_text[last_think_pos + len('</think>'):]

        # Remove all occurrences of </think> from the thinking part
        thinking_part = thinking_part.replace('</think>', '')

        # Display thinking part in small text
        st.markdown(f"<small>{thinking_part}</small>", unsafe_allow_html=True)
        # Display the answer part normally
        st.markdown(answer_part)
    else:
        # Display the entire text if no thinking part
        st.markdown(generated_text)


def display_in_chunks(generated_text):
    # Split the text into chunks by newline characters
    chunks = generated_text.split("\n")
    
    for chunk in chunks:
        # Render each chunk, if it contains math formulas, render as LaTeX
        if chunk.startswith("$") and chunk.endswith("$"):  # Math formula, render as LaTeX
            st.markdown(f"$$ {chunk} $$", unsafe_allow_html=True)
        else:
            st.markdown(chunk)


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        # Run the FastAPI server
        uvicorn.run(app_api, host="0.0.0.0", port=8091)
    else:
        # Run the Streamlit interface
        print("load model begin.")
        model, tokenizer = load_model()
        print("load model end.")

        user_avator = "statics/momo.png"
        robot_avator = "statics/robot.png"

        st.title("I am E.CoPI, your teaching assistant for 'Synthetic Biology'~")

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
                # Display long responses in chunks and render the thinking part
                display_thinking_and_answer(response)
            st.session_state.messages.append({
                "role": "robot",
                "content": response,
                "avatar": robot_avator,
            })
            torch.cuda.empty_cache()


if __name__ == "__main__":
    init()
    main()

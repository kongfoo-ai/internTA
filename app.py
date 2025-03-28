"""
InternTA: Synthetic Biology Teaching Assistant
==============================================

This application provides a conversational AI assistant specialized in synthetic biology education.
It supports two operation modes:
1. Remote API mode - Connects to a remote API endpoint (api.ecopi.chat) using an API key
2. Local model mode - Loads and runs a fine-tuned language model locally (if available)

Features:
- Interactive chat interface with Streamlit
- Support for streaming responses (local model only)
- LaTeX equation rendering in markdown
- Special handling for thinking/reasoning sections with </think> tags
- Configurable generation parameters (temperature, top_p, etc.)
- GPU memory management for local model operation

Usage:
- Run with `--local` flag to default to local model mode
- Run with `--show-local-option` to allow users to switch between local and remote modes

Dependencies:
- streamlit for the web interface
- transformers, torch, and peft for local model loading and inference
- requests for API communication

"""

import streamlit as st
import requests
import json
import os
import re
from datetime import datetime
import sys
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel  # Import PeftModel for loading LoRA adapters
import time
import argparse  # Add argparse for command line arguments


# Function to process content for display
def process_content_for_display(content, is_user_message=False):
    """
    Process content for display with special formatting for </think> tags,
    strip <|im_end|> tokens and backslashes, and ensure LaTeX equations display correctly
    """
    if not content:
        return ""
    
    # Strip <|im_end|> tokens
    content = content.replace("<|im_end|>", "")
    
    # Replace double backslashes with single backslashes
    if is_user_message:
        content = content.replace("\\\\", "\\")
    
    # Store all LaTeX expressions temporarily to preserve them
    latex_expressions = []
    
    # Function to replace LaTeX with placeholders
    def store_latex(match):
        latex = match.group(0)  # Get the entire match including delimiters
        placeholder = f"LATEX_PLACEHOLDER_{len(latex_expressions)}"
        latex_expressions.append(latex)
        return placeholder
    
    # Process different LaTeX patterns:
    # 1. Single dollar sign $...$
    content = re.sub(r'\$(.*?)\$', store_latex, content)
    
    # 2. Double dollar sign $$...$$
    content = re.sub(r'\$\$(.*?)\$\$', store_latex, content)
    
    # 3. \(...\) notation
    content = re.sub(r'\\\((.*?)\\\)', store_latex, content)
    
    # 4. \[...\] notation
    content = re.sub(r'\\\[(.*?)\\\]', store_latex, content)

    # 5. [...] notation
    content = re.sub(r'\[(.*?)\]', store_latex, content)
    
    # Restore LaTeX expressions with proper formatting for Streamlit markdown
    for i, latex in enumerate(latex_expressions):
        placeholder = f"LATEX_PLACEHOLDER_{i}"
        
        # Convert \(...\) to $...$ format
        if latex.startswith('\\(') and latex.endswith('\\)'):
            inner_content = latex[2:-2]  # Remove \( and \)
            latex = f'${inner_content}$'
        
        # Convert \[...\] to $$...$$ format
        elif latex.startswith('\\[') and latex.endswith('\\]'):
            inner_content = latex[2:-2]  # Remove \[ and \]
            latex = f'$${inner_content}$$'

        # Convert [...] to $...$ format
        elif latex.startswith('[') and latex.endswith(']'):
            inner_content = latex[2:-2]  # Remove [ and ]
            latex = f'${inner_content}$'
        
        # Replace the placeholder with the properly formatted LaTeX
        content = content.replace(placeholder, latex)
    
    # Find the position of the </think> tag
    think_pos = content.find('</think>')
    
    if think_pos == -1:
        # No </think> tag found, return content as is
        return content
    
    # Split the content into thinking part and regular part
    thinking_part = content[:think_pos + len('</think>')]
    regular_part = content[think_pos + len('</think>'):]
    
    # Apply styling to the thinking part using HTML with explicit dark font color for dark mode compatibility
    styled_thinking_part = f'''<div style="background-color: #f0f7ff; 
                                     border-left: 3px solid #4a88e5; 
                                     padding: 10px; 
                                     margin-bottom: 10px; 
                                     border-radius: 5px; 
                                     font-style: italic;
                                     color: #333333;">
                                {thinking_part}
                              </div>'''
    
    # Combine the styled thinking part with the regular part
    return styled_thinking_part + regular_part

# Function to load local model
@st.cache_resource
def load_local_model(lora_adapter_path="internTAv2.0_test", base_model_path="DeepSeek-R1-Distill-Qwen-7B"):
    """Load the base model and LoRA adapter separately, then merge them in memory"""
    print(f"Loading base model from {base_model_path} and LoRA adapter from {lora_adapter_path}...")
    
    # Check if CUDA is available and set the device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup quantization config for 4-bit if using CUDA
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # Load the base model with quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,  # 4-bit quantization
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # Load the base model without quantization for CPU
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Load the LoRA adapter and apply it to the base model
    try:
        print(f"Applying LoRA adapter from {lora_adapter_path} to base model...")
        lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        print("LoRA adapter applied successfully!")
        
        # Load tokenizer from the adapter path (it contains the specific tokenizer settings for the model)
        tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
        
        # Fix for attention mask warning - ensure pad_token is properly set
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Add a new pad token if neither pad nor eos token exists
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Resize model embeddings to match the new vocabulary size
                lora_model.resize_token_embeddings(len(tokenizer))
        
        print("Model loaded successfully!")
        return lora_model, tokenizer
    except Exception as e:
        print(f"Error applying LoRA adapter: {str(e)}")
        # Clean up resources
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

# Function to clear CUDA cache safely
def clear_cuda_cache():
    """Clear CUDA cache if available"""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all CUDA operations to finish
            print("CUDA cache cleared")
            return True
        except Exception as e:
            print(f"Error clearing CUDA cache: {str(e)}")
            return False
    return False

# Function to generate streaming response from local model
def stream_generate(model, tokenizer, input_ids, attention_mask=None, **gen_kwargs):
    """Generate text in a streaming fashion using the approach from run.py"""
    try:
        # Remove attention_mask from gen_kwargs if it's there to avoid duplication
        streamer_kwargs = {k: v for k, v in gen_kwargs.items() if k != 'attention_mask'}
        
        # Get generation parameters
        max_new_tokens = streamer_kwargs.get("max_new_tokens", 1000)
        do_sample = streamer_kwargs.get("do_sample", True)
        temperature = streamer_kwargs.get("temperature", 0.2)
        top_p = streamer_kwargs.get("top_p", 0.5)
        repetition_penalty = streamer_kwargs.get("repetition_penalty", 1.0)
        pad_token_id = streamer_kwargs.get("pad_token_id", tokenizer.eos_token_id)
        eos_token_id = tokenizer.eos_token_id
        
        # Setup for generation
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        input_length = input_ids.shape[1]  # Remember initial input length
        
        # Build model kwargs for prepare_inputs_for_generation
        model_kwargs = {
            "attention_mask": attention_mask,
        }
        
        # Main generation loop
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Prepare model inputs
                model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                
                # Forward pass
                outputs = model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                
                # Get next token logits
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply repetition penalty if needed
                if repetition_penalty > 1.0:
                    for i in range(input_ids.shape[0]):
                        for previous_token in input_ids[i]:
                            # Exponential penalty
                            next_token_logits[i, previous_token] /= repetition_penalty
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Filter with top_p
                if top_p < 1.0 and top_p > 0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for i in range(next_token_logits.shape[0]):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_token_logits[i, indices_to_remove] = -float("Inf")
                
                # Sample or greedy select
                if do_sample:
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Append next tokens to input_ids
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                
                # Update model kwargs for next generation step
                if attention_mask is not None:
                    # Extend attention mask for new tokens
                    attention_mask = torch.cat([
                        attention_mask, 
                        torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                    ], dim=1)
                
                # Update model_kwargs with updated attention_mask
                model_kwargs["attention_mask"] = attention_mask
                
                # Check if any sequences are finished
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
                
                # Decode the generated tokens so far (skip initial input)
                output_ids = input_ids[0, input_length:].cpu().tolist()
                text_generated = tokenizer.decode(output_ids, skip_special_tokens=True)
                
                # Free GPU memory
                del next_token_logits, next_tokens
                if 'sorted_logits' in locals(): del sorted_logits
                if 'sorted_indices' in locals(): del sorted_indices
                if 'cumulative_probs' in locals(): del cumulative_probs
                if 'probs' in locals(): del probs
                
                # Yield the generated text
                yield text_generated
                
                # Exit if all sequences are finished
                if unfinished_sequences.max() == 0:
                    break
    finally:
        # Always clean up CUDA memory after generation, even if there's an error
        del model_inputs, outputs
        if 'next_token_logits' in locals(): del next_token_logits
        if 'next_tokens' in locals(): del next_tokens
        
        # Optional additional cleanup for CUDA memory
        #clear_cuda_cache()

# Function to generate response using local model
def generate_local_response(model, tokenizer, messages, temperature=0.2, top_p=0.15, 
                           repetition_penalty=1.05, max_tokens=4096, do_sample=True, stream=True):
    """Generate a response using the local model with optional streaming"""
    try:
        # Format the conversation history for the model
        prompt = ""
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            prompt += f"{role}: {msg['content']}\n"
        
        prompt += "assistant: "
        
        # Tokenize the prompt with explicit attention mask
        tokenized_input = tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        input_ids = tokenized_input["input_ids"].to(model.device)
        attention_mask = tokenized_input["attention_mask"].to(model.device)
        
        # Set generation parameters - use all parameters from the UI
        gen_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": do_sample,  # Use the parameter from UI
            "top_p": top_p,  # Use the parameter from UI
            "repetition_penalty": repetition_penalty,  # Use the parameter from UI
            "pad_token_id": tokenizer.eos_token_id
        }
        
        if stream:
            # Streaming generation
            response = ""
            for new_text in stream_generate(model, tokenizer, input_ids, attention_mask=attention_mask, **gen_config):
                response = new_text
                yield response
        else:
            # Non-streaming generation
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_config
            )
            response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            yield response
    finally:
        # Always clean up to avoid memory leaks
        if 'tokenized_input' in locals(): del tokenized_input
        if 'input_ids' in locals(): del input_ids
        if 'attention_mask' in locals(): del attention_mask
        if 'outputs' in locals(): del outputs
        
        # Final CUDA cleanup
        #clear_cuda_cache()

# Setup command line arguments and default model source
def parse_args():
    parser = argparse.ArgumentParser(description="InternTA: Synthetic Biology Teaching Assistant")
    parser.add_argument("--local", action="store_true", help="Use local model instead of remote API")
    parser.add_argument("--show-local-option", action="store_true", help="Show local model as an option in the UI")
    return parser.parse_args()

# Main function to start the app
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set default model source based on command line arguments
    if "model_source" not in st.session_state:
        if args.local:
            st.session_state.model_source = "本地模型 | Local Model"
        else:
            st.session_state.model_source = "远程 API | Remote API"
    
    # Store show_local_option flag in session state
    if "show_local_option" not in st.session_state:
        st.session_state.show_local_option = args.show_local_option

    # Start the Streamlit app (the existing code continues from here)
    # Page configuration
    st.set_page_config(
        page_title="InternTA: 合成生物学助教 | Synthetic Biology Teaching Assistant",
        page_icon="🧬",
        layout="wide"
    )

    # Initialize messages if not in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Fixed API URL - no need for user input
    API_BASE_URL = "https://api.ecopi.chat/v1/chat/completions"

    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    # Fixed model name - no need for user input
    MODEL_NAME = "internta"

    # Sidebar for configuration
    with st.sidebar:
        st.title("InternTA: 合成生物学助教 | Synthetic Biology TA")

        # Model source selection - depends on whether to show local option
        if st.session_state.show_local_option:
            # Show both options
            model_source = st.radio(
                "模型来源 | Model Source",
                ["本地模型 | Local Model", "远程 API | Remote API"],
                index=0 if st.session_state.model_source == "本地模型 | Local Model" else 1
            )
        else:
            # Only show remote API option
            model_source = "远程 API | Remote API"
            
        # Update session state if user changes selection
        if model_source != st.session_state.model_source:
            st.session_state.model_source = model_source
        
        if model_source == "远程 API | Remote API":
            # Display the fixed API URL (read-only)
            st.info(f"获取 API 密钥地址 | Obtain API Key from here: https://docs.ecopi.chat")
            
            # API Key input
            api_key = st.text_input("Please enter an API Key | 请输入 API 密钥", 
                                   value=st.session_state.api_key,
                                   placeholder="sk-...",
                                   type="password")
            
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key
            
            # 远程API模式下的参数设置
            st.write("参数设置 | Parameters:")
            temperature = st.slider("温度 | Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
            top_p = st.slider("Top P (核采样阈值 | Nucleus sampling threshold)", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
            max_tokens = st.number_input("最大生成长度 | Max Tokens", min_value=1, max_value=100000, value=4096, step=100)
            
            # 显示流式响应不可用的提示
            st.warning("远程API不支持流式响应，将使用非流式模式 | Remote API does not support streaming, will use non-streaming mode")
            use_streaming = False  # 远程API强制非流式
        else:
            # Path to base model and LoRA adapter
            with st.expander("模型设置 | Model Settings", expanded=True):
                base_model_path = st.text_input(
                    "基础模型路径 | Base Model Path",
                    value="DeepSeek-R1-Distill-Qwen-7B",
                    help="基础大语言模型的路径 | Path to base language model"
                )
                
                lora_adapter_path = st.text_input(
                    "LoRA适配器路径 | LoRA Adapter Path",
                    value="internTAv2.0_test",
                    help="LoRA微调适配器的路径 | Path to LoRA fine-tuned adapter"
                )
            
            # Check if we need to load or reload the model
            model_changed = (
                "base_model_path" not in st.session_state or 
                "lora_adapter_path" not in st.session_state or
                st.session_state.get("base_model_path", "") != base_model_path or
                st.session_state.get("lora_adapter_path", "") != lora_adapter_path
            )
            
            if model_changed or "local_model" not in st.session_state:
                # Only load the model if it's not loaded or the paths have changed
                with st.spinner("加载模型中... | Loading model..."):
                    try:
                        st.session_state.local_model, st.session_state.local_tokenizer = load_local_model(
                            lora_adapter_path=lora_adapter_path,
                            base_model_path=base_model_path
                        )
                        # Store current paths in session state
                        st.session_state.base_model_path = base_model_path
                        st.session_state.lora_adapter_path = lora_adapter_path
                        st.success("模型加载成功！| Model loaded successfully!")
                    except Exception as e:
                        st.error(f"加载模型失败 | Failed to load model: {str(e)}")
                        if "cuda" in str(e).lower() and "out of memory" in str(e).lower():
                            st.warning("GPU内存不足，请考虑使用更小的模型或清理GPU内存 | GPU out of memory, consider using a smaller model or freeing GPU memory")
            
            # 本地模型的参数设置 - Enhanced with parameters from run.py
            st.write("生成参数设置 | Generation Parameters:")
            temperature = st.slider("温度 | Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05, 
                                  help="控制生成文本的随机性。较高的值 (如 0.8) 会使输出更加多样化，较低的值 (如 0.2) 使输出更加确定和集中 | Controls randomness in generation. Higher (0.8) is more diverse, lower (0.2) more focused")
            
            top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.15, step=0.05,
                             help="核采样阈值 - 模型只考虑概率总和达到此值的候选词 | Nucleus sampling threshold - model only considers tokens that make up this probability mass")
            
            repetition_penalty = st.slider("重复惩罚 | Repetition Penalty", min_value=1.0, max_value=2.0, value=1.05, step=0.05,
                                         help="控制重复内容的惩罚力度，较高的值会减少重复 | Controls penalty for repetition, higher reduces repetition")
            
            max_tokens = st.number_input("最大生成长度 | Max Tokens", min_value=100, max_value=100000, value=4096, step=100,
                                       help="响应中生成的最大标记数 | Maximum number of tokens to generate in response")
            
            # 高级选项 (可折叠) | Advanced options (collapsible)
            with st.expander("高级选项 | Advanced Options"):
                do_sample = st.checkbox("使用采样 | Use Sampling", value=True, 
                                      help="启用从概率分布采样，关闭则使用贪婪搜索 | Enable sampling from probability distribution, disable for greedy search")
                
            # 本地模型支持流式响应选项
            use_streaming = st.checkbox("流式响应 | Streaming Response", value=True, 
                                      help="逐步显示生成的文本 | Display generated text incrementally")
        
        # Memory management
        with st.expander("内存管理 | Memory Management"):
            if st.button("清理GPU缓存 | Clear GPU Cache"):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    st.success("GPU缓存已清理 | GPU cache cleared")
                else:
                    st.info("未检测到GPU | No GPU detected")
        
        # Clear chat button
        if st.button("清空对话 | Clear Chat", help="清除所有对话历史 | Clear all chat history"):
            st.session_state.messages = []
            st.rerun()

    # Main chat interface
    #st.title("InternTA: 合成生物学助教 | Synthetic Biology Teaching Assistant")
    #st.caption("基于 InternLM2 大模型，帮助学生更好地学习《合成生物学》 | Based on InternLM2 model, helping students better learn Synthetic Biology")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Process the content for display with all necessary transformations
            processed_content = process_content_for_display(message["content"])
            st.markdown(processed_content, unsafe_allow_html=True)

    # Get user input
    if prompt := st.chat_input("请输入您的问题... | Enter your question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            processed_prompt = process_content_for_display(prompt, is_user_message=True)
            st.markdown(processed_prompt, unsafe_allow_html=True)
        
        # Prepare model call
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("思考中... | Thinking...")
            
            try:
                # Determine which model to use
                if model_source == "本地模型 | Local Model":
                    # Check if model is loaded
                    if "local_model" not in st.session_state:
                        message_placeholder.error("本地模型未加载，请检查模型路径 | Local model not loaded, please check model path")
                        st.stop()
                    
                    # Use local model
                    if use_streaming:
                        # Process streaming response
                        full_response = ""
                        
                        # Stream the response with all parameters from UI
                        for response_chunk in generate_local_response(
                            st.session_state.local_model,
                            st.session_state.local_tokenizer,
                            st.session_state.messages,
                            temperature=temperature,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty,
                            max_tokens=max_tokens,
                            do_sample=do_sample if 'do_sample' in locals() else True,
                            stream=True
                        ):
                            # Update the full response
                            full_response = response_chunk
                            # Update the placeholder with the processed content
                            processed_content = process_content_for_display(full_response)
                            message_placeholder.markdown(processed_content + "▌", unsafe_allow_html=True)
                            # Short sleep to reduce CPU usage and improve UI responsiveness
                            time.sleep(0.01)
                        
                        # Final update without the cursor
                        if full_response:
                            processed_content = process_content_for_display(full_response)
                            message_placeholder.markdown(processed_content, unsafe_allow_html=True)
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        else:
                            message_placeholder.error("无响应内容 | No response content")
                    else:
                        # Non-streaming generation with all parameters from UI
                        full_response = ""
                        for response in generate_local_response(
                            st.session_state.local_model,
                            st.session_state.local_tokenizer,
                            st.session_state.messages,
                            temperature=temperature,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty,
                            max_tokens=max_tokens,
                            do_sample=do_sample if 'do_sample' in locals() else True,
                            stream=False
                        ):
                            full_response = response
                            break  # Only need the first (and only) result
                        
                        if full_response:
                            processed_content = process_content_for_display(full_response)
                            message_placeholder.markdown(processed_content, unsafe_allow_html=True)
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        else:
                            message_placeholder.error("无响应内容 | No response content")
                else:
                    # Use remote API
                    if not st.session_state.api_key:
                        message_placeholder.error("请在侧边栏输入 API 密钥 | Please enter an API key in the sidebar")
                        st.stop()
                    
                    # 远程API模式下强制使用非流式响应，不管用户在界面上选择什么
                    # For remote API, always force non-streaming mode regardless of UI selection
                    actual_streaming = False  # 强制设置为False | Force to False
                    
                    if use_streaming:
                        message_placeholder.warning("远程API不支持流式响应，已自动切换为非流式模式 | Remote API doesn't support streaming, automatically switched to non-streaming mode")
                    
                    # Prepare the payload for the API call
                    payload = {
                        "model": MODEL_NAME,
                        "messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                        "temperature": float(temperature),
                        "max_tokens": int(max_tokens),
                        "top_p": float(top_p),
                        "stream": actual_streaming  # 使用强制的非流式模式 | Use forced non-streaming mode
                    }
                    
                    # Add optional parameters if provided from UI
                    if "repetition_penalty" in locals() and repetition_penalty > 1.0:
                        payload["repetition_penalty"] = float(repetition_penalty)
                    
                    # If there's an n parameter (number of completions), ensure it's an integer
                    payload["n"] = 1  # Set to 1 as we just want a single response
                    
                    headers = {
                        "Content-Type": "application/json"
                    }
                    
                    # Add Authorization header if API key is provided
                    headers["Authorization"] = f"Bearer {st.session_state.api_key}"
                    
                    # Make the API call
                    response = requests.post(
                        API_BASE_URL,
                        headers=headers,
                        json=payload,
                        stream=actual_streaming,  # 使用强制的非流式模式 | Use forced non-streaming mode
                        timeout=600  # 10-minute timeout
                    )
                    
                    if response.status_code == 200:
                        # Since we've forced non-streaming mode, we only need to handle non-streaming responses
                        try:
                            response_json = response.json()
                            if "choices" in response_json and len(response_json["choices"]) > 0:
                                content = response_json["choices"][0]["message"]["content"]
                                processed_content = process_content_for_display(content)
                                message_placeholder.markdown(processed_content, unsafe_allow_html=True)
                                # Add assistant response to chat history
                                st.session_state.messages.append({"role": "assistant", "content": content})
                            else:
                                message_placeholder.error("无响应内容 | No response content")
                        except Exception as e:
                            message_placeholder.error(f"处理响应时出错 | Error processing response: {str(e)}")
                            print(f"Response processing error: {str(e)}")
                            print(f"Response content: {response.text[:500]}")  # Print first 500 chars of response for debugging
                    else:
                        error_message = f"错误 | Error: {response.status_code} - {response.text}"
                        message_placeholder.error(error_message)
                        print(f"API Error: {error_message}")
                        
                        # If it's the specific "ids" error, suggest a solution
                        if "ids" in response.text and "list" in response.text and "integer" in response.text:
                            message_placeholder.warning("尝试关闭流式响应并重试 | Try turning off streaming response and retry")
                        
                        # If it's a streaming-related error, inform the user again
                        if "stream" in response.text.lower() or "not allowed" in response.text.lower():
                            message_placeholder.warning("远程API不支持流式响应，请确保请求设置为非流式模式 | Remote API does not support streaming, please ensure requests are set to non-streaming mode")
                
                # Clear GPU cache if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                message_placeholder.error(f"错误 | Error: {str(e)}")
                print(f"Generation error: {str(e)}")
                
                # If it's a CUDA out of memory error, suggest a solution
                if "CUDA out of memory" in str(e):
                    message_placeholder.warning("GPU内存不足，请尝试降低最大生成长度或使用CPU模式 | GPU out of memory, try reducing max tokens or using CPU mode")
                    
                # Clear GPU cache if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Display some helpful information at the bottom
    st.markdown("---")
    if model_source == "本地模型 | Local Model":
        st.caption(f"使用模型 | Using model: Base={base_model_path}, LoRA={lora_adapter_path}")
    else:
        st.caption("默认 API 端点 ｜ Default API Endpoint: https://api.ecopi.chat - 请在侧边栏输入 API 密钥开始对话。| Enter your API key in the sidebar to begin.")

# Run the main function when executed directly
if __name__ == "__main__":
    main()

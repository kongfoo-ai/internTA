import streamlit as st
import requests
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="InternTA: 合成生物学助教 | Synthetic Biology Teaching Assistant",
    page_icon="🧬",
    layout="wide"
)

# Initialize session state variables if they don't exist
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

    # Display the fixed API URL (read-only)
    st.info(f"获取 API 密钥地址 | Obtain API Key from here: https://docs.ecopi.chat")
    
    # API Key input
    api_key = st.text_input("Please enter an API Key | 请输入 API 密钥", 
                           value=st.session_state.api_key,
                           placeholder="sk-...",
                           type="password")
    
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
    
    # Parameters
    temperature = st.slider("温度 | Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    max_tokens = st.number_input("最大生成长度 | Max Tokens", min_value=1, max_value=4096, value=1024, step=1)
    
    # Clear chat button
    if st.button("清空对话 | Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

# Main chat interface
#st.title("InternTA: 合成生物学助教 | Synthetic Biology Teaching Assistant")
#st.caption("基于 InternLM2 大模型，帮助学生更好地学习《合成生物学》 | Based on InternLM2 model, helping students better learn Synthetic Biology")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
if prompt := st.chat_input("请输入您的问题... | Enter your question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Prepare API call
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("思考中... | Thinking...")
        
        try:
            # Prepare the payload for the API call
            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # Add Authorization header if API key is provided
            if st.session_state.api_key:
                headers["Authorization"] = f"Bearer {st.session_state.api_key}"
            else:
                message_placeholder.error("请在侧边栏输入 API 密钥 | Please enter an API key in the sidebar")
                st.stop()
            
            # Make the API call
            response = requests.post(
                API_BASE_URL,
                headers=headers,
                json=payload,
                timeout=120  # 2-minute timeout
            )
            
            if response.status_code == 200:
                # Parse the response
                response_data = response.json()
                
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    assistant_response = response_data["choices"][0]["message"]["content"]
                    
                    # Update the placeholder with the response
                    message_placeholder.markdown(assistant_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                else:
                    message_placeholder.error("API 响应格式无效 | Invalid API response format")
            else:
                message_placeholder.error(f"错误 | Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            message_placeholder.error(f"错误 | Error: {str(e)}")

# Display some helpful information at the bottom
st.markdown("---")
st.caption("默认 API 端点 ｜ Default API Endpoint: https://api.ecopi.chat - 请在侧边栏输入 API 密钥开始对话。| Enter your API key in the sidebar to begin.")

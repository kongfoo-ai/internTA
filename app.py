import streamlit as st
import requests
import json
import os
import re
from datetime import datetime

# Function to process content for display
def process_content_for_display(content):
    """
    Process content for display with special formatting for </think> tags,
    strip <|im_end|> tokens and backslashes, and ensure LaTeX equations display correctly
    """
    if not content:
        return ""
    
    # Strip <|im_end|> tokens
    content = content.replace("<|im_end|>", "")
    
    # Replace double backslashes with single backslashes
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
    temperature = st.slider("温度 | Temperature", min_value=0.0, max_value=1.0, value=0.25, step=0.1)
    max_tokens = st.number_input("最大生成长度 | Max Tokens", min_value=1, max_value=100000, value=4096, step=1)
    
    # Set streaming to false by default and hide the option
    use_streaming = False
    
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
        # Process the content for display with all necessary transformations
        processed_content = process_content_for_display(message["content"])
        st.markdown(processed_content, unsafe_allow_html=True)

# Get user input
if prompt := st.chat_input("请输入您的问题... | Enter your question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        processed_prompt = process_content_for_display(prompt)
        st.markdown(processed_prompt, unsafe_allow_html=True)
        #st.markdown(prompt)
    
    # Prepare API call
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("思考中... | Thinking...")
        
        try:
            # Prepare the payload for the API call
            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "stream": use_streaming  # Always False now
            }
            
            # Add optional parameters with proper typing
            # If there's an n parameter (number of completions), ensure it's an integer
            payload["n"] = 1  # Set to 1 as we just want a single response
            
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
                stream=use_streaming,  # Always False now
                timeout=600  # 10-minute timeout
            )
            
            if response.status_code == 200:
                if use_streaming:
                    # Process the streaming response
                    full_response = ""
                    
                    # Iterate through the streaming response
                    for chunk in response.iter_lines():
                        if chunk:
                            # Decode the chunk
                            chunk_decoded = chunk.decode('utf-8')
                            
                            # Skip the "data: " prefix and empty lines
                            if chunk_decoded.startswith('data: '):
                                chunk_data = chunk_decoded[6:]  # Remove 'data: ' prefix
                                
                                # Skip "[DONE]" message
                                if chunk_data == "[DONE]":
                                    continue
                                    
                                try:
                                    # Parse the JSON chunk
                                    chunk_json = json.loads(chunk_data)
                                    
                                    # Extract the content delta if available
                                    if "choices" in chunk_json and len(chunk_json["choices"]) > 0:
                                        choice = chunk_json["choices"][0]
                                        if "delta" in choice and "content" in choice["delta"]:
                                            content_delta = choice["delta"]["content"]
                                            if content_delta:
                                                full_response += content_delta
                                                # Update the placeholder with the processed content
                                                processed_content = process_content_for_display(full_response)
                                                message_placeholder.markdown(processed_content + "▌", unsafe_allow_html=True)
                                except json.JSONDecodeError as je:
                                    # Log the issue for debugging
                                    print(f"JSON decode error: {je}, Data: {chunk_data[:100]}")
                                    continue
                                except Exception as e:
                                    # Catch any other errors in processing chunks
                                    print(f"Error processing chunk: {e}, Data: {chunk_data[:100]}")
                                    continue
                    
                    # Final update without the cursor
                    if full_response:
                        processed_content = process_content_for_display(full_response)
                        message_placeholder.markdown(processed_content, unsafe_allow_html=True)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        message_placeholder.error("无响应内容 | No response content")
                else:
                    # Process the non-streaming response
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
                
        except Exception as e:
            message_placeholder.error(f"错误 | Error: {str(e)}")

# Display some helpful information at the bottom
st.markdown("---")
st.caption("默认 API 端点 ｜ Default API Endpoint: https://api.ecopi.chat - 请在侧边栏输入 API 密钥开始对话。| Enter your API key in the sidebar to begin.")

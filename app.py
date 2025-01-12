from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import jwt
import datetime

app = Flask(__name__)
CORS(app)

SECRET_KEY = "your-secret-key"

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        
        if not token:
            return jsonify({'message': 'Missing authentication token'}), 401
            
        try:
            jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except:
            return jsonify({'message': 'Invalid token'}), 401
            
        return f(*args, **kwargs)
    return decorated

@app.route('/v1/chat/completions', methods=['POST'])
@token_required
def create_chat_completion():
    data = request.get_json()
    
    # Validate required fields
    if not data.get('model') or not data.get('messages'):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Mock response
    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response. Based on your input, I generated this example response."
                }
            }
        ]
    }
    
    return jsonify(mock_response), 200

# 用于测试的创建令牌的端点
@app.route('/get_token', methods=['GET'])
def get_token():
    token = jwt.encode(
        {'user': 'test', 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)},
        SECRET_KEY,
        algorithm="HS256"
    )
    return jsonify({'token': token})

if __name__ == '__main__':
    app.run(debug=True)

'''
pip install flask flask-cors pyjwt

# 获取测试用的 token
curl http://localhost:5000/get_token

# 使用 token 调用聊天接口
curl -X POST \
  http://localhost:5000/v1/chat/completions \
  -H 'Authorization: Bearer <your_token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "internta-v02",
    "messages": [
        {
            "role": "user",
            "content": "你好"
        }
    ]
}'

# Generate SECRET_KEY
# 方法1:使用 secrets
import secrets
secret_key = secrets.token_hex(32)
print(secret_key)

# 方法2:使用 os.urandom
import os
secret_key = os.urandom(32).hex()
print(secret_key)
'''
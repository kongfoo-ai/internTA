'''
The script reads conversation data from an Excel file, structures it into a format suitable for
OpenAI, and then repeats the data a specified number of times before saving it to a JSON file.
'''
import json
import pandas as pd

# 设置用户的名字
NAME = 'internTA'
# 设置需要重复添加的数据次数
N = 1000
EXCEL_FILE = 'examples.xlsx'
df = pd.read_excel(EXCEL_FILE, engine='openpyxl')

# 初始化OpenAI格式的数据结构
conversations = {}
con_ls = []

for index, row in df.iterrows():
    conv_id = row['conversation']
    if conv_id not in con_ls:
        conversations[conv_id] = {"messages": []}
        con_ls.append(conv_id)
    conversations[conv_id]["messages"].append(
            {
                "role": "user",
                "content": row['Q']
            })
    conversations[conv_id]["messages"].append(
            {
                "role": "assistant",
                "content": row['A']
            }
    )

# 将OpenAI格式的数据结构合并到一个列表中
data = [value for key, value in conversations.items()]

# 通过循环，将初始化的对话数据重复添加到data列表中
output = []
for i in range(N):
    for d in data:
        num = len(d['messages'])
        for j in range(num):
            output.append(d)

# 将data列表中的数据写入到一个名为'personal_assistant.json'的文件中
with open('personal_assistant.json', 'w', encoding='utf-8') as f:
    # 使用json.dump方法将数据以JSON格式写入文件
    # ensure_ascii=False 确保中文字符正常显示
    # indent=4 使得文件内容格式化，便于阅读
    json.dump(output, f, ensure_ascii=False, indent=4)

from datasets import load_dataset,load_from_disk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments

import wandb

import os
os.environ['WANDB_INIT_TIMEOUT'] = '600'
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# find the dataset
import os
from datetime import datetime
import json
import argparse

# 添加参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True, help='模型名称')
parser.add_argument('--max_seq_length', type=int, default=8192, help='最大序列长度')
parser.add_argument('--model_save_path', type=str, required=True, help='模型保存路径')
parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称')  # 新增数据集名称参数
args = parser.parse_args()

# def find_latest_directory(base_path):
#     latest_time = None
#     latest_directory = None

#     # 遍历上层目录
#     parent_directory = os.path.dirname(base_path)
#     for entry in os.listdir(parent_directory):
#         entry_path = os.path.join(parent_directory, entry)
#         # 检查是否是目录且以 'dataset_' 开头
#         if os.path.isdir(entry_path) and entry.startswith("dataset_"):
#             # 提取时间部分并转换为 datetime 对象
#             try:
#                 dir_time = datetime.strptime(entry, "dataset_%Y%m%d_%H%M%S")
#                 # 更新最新目录
#                 if latest_time is None or dir_time > latest_time:
#                     latest_time = dir_time
#                     latest_directory = entry_path
#             except ValueError:
#                 continue  # 如果格式不匹配，跳过

#     return latest_directory

# 使用示例
# current_file_path = "/ailab/user/hantao_dispatch/project/internTA/datasets/"
# latest_dir = find_latest_directory(current_file_path)

# if latest_dir:
#     print(f"最新的目录是: {latest_dir}")
# else:
#     print("未找到符合条件的目录。")

# Load the dataset
dataset_name = args.dataset_path
dataset = load_from_disk(dataset_name) 
# Device map
device_map = 'auto'  # for PP and running with `python test_sft.py`

# Load the model + tokenizer
model_name = args.model_name  # 从命令行参数获取模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache=False,
    device_map=device_map
)
# PEFT config
lora_alpha = 16
lora_dropout = 0.1
lora_r = 32  # 64
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
    modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"],
)
# Args
max_seq_length = args.max_seq_length  # 从命令行参数获取最大序列长度
output_dir = os.path.join(args.model_save_path, "results")
os.makedirs(output_dir, exist_ok=True)
per_device_train_batch_size = 1  # reduced batch size to avoid OOM
gradient_accumulation_steps = 8
optim = "adamw_torch"
save_steps = 10
logging_steps = 1
learning_rate = 5e-5
max_grad_norm = 0.2
max_steps = 400  
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=False,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,  # gradient checkpointing
    # fsdp="full_shared auto_wrap",
    #report_to="wandb",
)
# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="context",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    # ddp_find_unused_parameters=False,
)

# Train :)
trainer.train()

# 定义保存模型的绝对路径
model_save_path = args.model_save_path  # 从命令行参数获取模型保存路径
# 获取当前日期
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
# 定义模型名称
model_name = f"synBiology_model_{current_date}"

os.makedirs(model_save_path, exist_ok=True)

# 保存模型
trainer.save_model(os.path.join(model_save_path, model_name))
tokenizer.save_pretrained(os.path.join(model_save_path, model_name))

# 记录信息
record = {
    "initial_model_name": model_name,  # 初始模型名称
    "trained_model_name": model_name,   # 训练后的模型名称
    "dataset_directory": args.dataset_path,     # 数据集的目录
    "training_arguments": {               # 添加训练参数
        "max_seq_length": max_seq_length,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optim": optim,
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "max_steps": max_steps,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
    }
}

# 保存记录到 JSON 文件
model_save_path_name = os.path.join(model_save_path, model_name)
record_file_path = os.path.join(model_save_path_name, "model_record.json")
with open(record_file_path, "w", encoding="utf-8") as record_file:
    json.dump(record, record_file, ensure_ascii=False, indent=4)

print(f"model has been saved to '{model_save_path_name}'。")
print(f"record has been saved to '{record_file_path}'。")

import json

train_logs = trainer.state.log_history  # 获取训练日志

# 仅提取 loss、learning_rate 和 epoch
loss_data = [
    {
        "step": i + 1,
        "loss": entry["loss"],
        "learning_rate": entry["learning_rate"],
        "epoch": entry["epoch"],
        "grad_norm": entry.get("grad_norm", None),
    }
    for i, entry in enumerate(train_logs) if "loss" in entry
]

# 保存到 JSON 文件
with open(os.path.join(model_save_path_name, "loss_log.json"), "w", encoding="utf-8") as f:
    json.dump(loss_data, f, ensure_ascii=False, indent=4)

# print("Loss 记录已保存到 loss_log.json")
print("Training is done")


"""
Model Merger Utility for InternTA
=================================

This script merges a base language model with a LoRA (Low-Rank Adaptation) adapter
to create a standalone model with the fine-tuned weights incorporated directly.

Purpose:
- Combines a quantized base model with a LoRA adapter fine-tuned for synthetic biology
- Creates a single merged model that can be deployed without requiring separate adapter loading
- Useful for deployment scenarios where loading adapters separately is inconvenient

Usage:
    python merge.py --base-model <BASE_MODEL_PATH> --lora-adapter <LORA_PATH> --output-path <OUTPUT_PATH>

Arguments:
    --base-model: Path to the base language model (default: DeepSeek-R1-Distill-Qwen-7B)
    --lora-adapter: Path to the LoRA adapter (default: internTAv2.0_test)
    --output-path: Path to save the merged model (default: merged_model)

Technical details:
- Uses 4-bit quantization (NF4) to load the base model efficiently
- Preserves the tokenizer configuration from the LoRA adapter
- Compatible with the InternTA application's model loading requirements

This is a companion utility to the main InternTA application (app.py).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import argparse

def load_and_merge_models(base_model_path, lora_adapter_path, output_path):
    # ========== 量化配置 (支持 4-bit 量化) ==========
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # ========== 加载基础模型 ==========
    print(f"加载基础模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,  # 4-bit 量化
        device_map="auto",
        trust_remote_code=True
    )
    
    # ========== 加载 QLoRA 适配器 ==========
    print(f"加载 LoRA 适配器: {lora_adapter_path}")
    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    # ========== 加载 Tokenizer ==========
    print("加载 Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
    
    # ========== 合并模型 ==========
    print("合并模型中...")
    merged_model = lora_model.merge_and_unload()
    
    # ========== 保存合并后的模型 ==========
    print(f"保存合并后的模型到: {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("模型合并完成!")
    return merged_model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="合并基础模型和LoRA适配器")
    parser.add_argument("--base-model", type=str, default="DeepSeek-R1-Distill-Qwen-7B", help="基础模型路径")
    parser.add_argument("--lora-adapter", type=str, default="internTAv2.0_test", help="LoRA适配器路径")
    parser.add_argument("--output-path", type=str, default="merged_model", help="输出模型路径")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_path, exist_ok=True)
    
    # 加载、合并并保存模型
    load_and_merge_models(args.base_model, args.lora_adapter, args.output_path)

if __name__ == "__main__":
    main()
    
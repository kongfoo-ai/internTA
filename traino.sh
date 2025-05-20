#!/bin/bash

python train/train_agent.py \
  --base_model_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --data_dir "train_data" \
  --val_dataset "val_dataset" \
  --output_root "output" \
  --batch_size 1 \
  --grad_accum 8 \
  --learning_rate 5e-5 \
  --max_steps 120 \
  --warmup_ratio 0.15 \
  --save_steps 100 \
  --logging_steps 1 \
  --lr_scheduler "cosine" \
  --lora_r 32 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --target_accuracy 0.90 \
  --max_rounds 5 \
  --llm_judge_token "your_api_token" \
  --llm_judge_model "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" \
  --llm_judge_url "https://api.siliconflow.cn/v1/chat/completions"

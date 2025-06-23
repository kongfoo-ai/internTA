#!/bin/bash

python train/train_agent.py \
    --base_model_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --data_dir "dataset/training" \
    --val_dataset "dataset/validation.json" \
    --output_root "training_output" \
    --batch_size 1 \
    --grad_accum 8 \
    --learning_rate 5e-5 \
    --max_steps 30 \
    --warmup_ratio 0.15 \
    --save_steps 100 \
    --logging_steps 1 \
    --lr_scheduler "cosine" \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --target_accuracy 0.5 \
    --max_rounds 3 \
    --llm_judge_token "your tokens" \
    --llm_judge_model "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" \
    --llm_judge_url "https://api.siliconflow.cn/v1/chat/completions"
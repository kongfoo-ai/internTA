import os
import argparse
import sklearn
import torch
from transformers import modeling_utils

# Ensure ALL_PARALLEL_STYLES is set
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") \
   or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_from_disk
import re
import json
import requests

SYSTEM_PROMPT = '''
You are a Training Planner agent.
Your job is to decide, based on the latest training metrics, whether to adjust the learning rate, extend the number of epochs, or stop training.

Input metrics:
- epoch: integer, the current epoch number.
- val_loss_history: list of floats, the validation loss for the last N epochs (newest last).
- judge_score_history: list of floats between 0-1, the Judge LLM score for the last N epochs (newest last).
- current_lr: float, the current learning rate.
- max_epochs: integer, the configured maximum number of epochs.

Rules:
1. If the latest `judge_score_history[-1]` ≥ `0.9`, then stop training.
2. If `val_loss_history` has not decreased by ≥1% over the last 3 epochs, set `new_lr = current_lr * 0.5`.
3. If judge score is increasing but still <0.9, and epoch is within 80% of `max_epochs`, then extend epochs by +5.
4. Otherwise, make no change.

Respond **only** with a JSON object following the schema:
```json
{
  "stop": <bool>,
  "new_lr": <float|null>,
  "extend_epochs": <int|null>,
  "reason": <string>
}
```'''

# Utility to check format

def check_format(answer: str) -> bool:
    pattern = r".*</think>.*</think>.*"
    return bool(re.search(pattern, answer, flags=re.DOTALL))

# Add project root to path for data utils
import sys
def add_project_root():
    root = os.path.abspath(os.path.join(__file__, '..', '..'))
    if root not in sys.path:
        sys.path.insert(0, root)
add_project_root()

from data.internTA2_evaluation_utils import llm_as_judge, process_data

from peft import LoraConfig, PeftModel

class TrainingAgent:
    def __init__(
        self,
        base_model_path,
        data_dir,
        val_dataset,
        output_root,
        initial_hyperparams,
        llm_judge_token,
        llm_judge_model,
        llm_judge_url,
        target_accuracy=0.90,
        max_rounds=5
    ):
        # Quantization config
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        # Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=self.bnb_config,
            trust_remote_code=True,
            use_cache=False,
            device_map={"": 0}
        )
        # PEFT config
        self.peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            bias="none",
            r=initial_hyperparams['lora_r'],
            lora_alpha=initial_hyperparams['lora_alpha'],
            lora_dropout=initial_hyperparams['lora_dropout'],
            target_modules=["q_proj","k_proj","v_proj","up_proj","down_proj","gate_proj"],
        )
        # Save settings
        self.base_model_path = base_model_path
        self.data_dir = data_dir
        self.val_dataset = val_dataset
        self.output_root = output_root
        self.target_accuracy = target_accuracy
        self.max_rounds = max_rounds
        self.hyperparams = initial_hyperparams
        self.llm_judge_token = llm_judge_token
        self.llm_judge_model = llm_judge_model
        self.llm_judge_url = llm_judge_url
        self.train_loss_history = []
        self.judge_score_history = []
        self.current_epoch = 0
        self.current_step = 0
        self.trainer = None

        # self.base = AutoModelForCausalLM.from_pretrained(
        #     self.base_model_path,
        #     quantization_config=self.bnb_config,
        #     trust_remote_code=True,
        #     use_cache=False,
        #     device_map={"": 0}
        # )

    def _create_trainer(self, round_idx):
        output_dir = os.path.join(self.output_root, f"round_{round_idx}")
        os.makedirs(output_dir, exist_ok=True)

        if round_idx == 1:
            # 第一轮：用base model + LoRA配置
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=self.bnb_config,
                trust_remote_code=True,
                use_cache=False,
                device_map={"": 0}
            )
            peft_config = self.peft_config
            self.trainer = SFTTrainer(
                model=model,
                train_dataset=load_from_disk(os.path.join(self.data_dir, "train_dataset")),
                peft_config=peft_config,
                args=TrainingArguments(
                    output_dir=output_dir,
                    per_device_train_batch_size=self.hyperparams['batch_size'],
                    gradient_accumulation_steps=self.hyperparams['grad_accum'],
                    learning_rate=self.hyperparams['learning_rate'],
                    fp16=True,
                    max_steps=self.hyperparams['max_steps'],
                    warmup_ratio=self.hyperparams['warmup_ratio'],
                    save_steps=self.hyperparams['save_steps'],
                    logging_steps=self.hyperparams['logging_steps'],
                    lr_scheduler_type=self.hyperparams['lr_scheduler'],
                    report_to="none",
                ),
            )
        else:
            # 后续轮次：只加载adapter，不再加peft_config
            last_dir = os.path.join(self.output_root, f"round_{round_idx-1}")
            model = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    quantization_config=self.bnb_config,
                    trust_remote_code=True,
                    use_cache=False,
                    device_map={"": 0}
                ),
                last_dir,
                device_map={"": 0}
            )
            # 关键：确保LoRA参数可训练
            model.train()
            for name, param in model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True

            self.trainer = SFTTrainer(
                model=model,
                train_dataset=load_from_disk(os.path.join(self.data_dir, "train_dataset")),
                args=TrainingArguments(
                    output_dir=output_dir,
                    per_device_train_batch_size=self.hyperparams['batch_size'],
                    gradient_accumulation_steps=self.hyperparams['grad_accum'],
                    learning_rate=self.hyperparams['learning_rate'],
                    fp16=True,
                    max_steps=self.hyperparams['max_steps'],
                    warmup_ratio=self.hyperparams['warmup_ratio'],
                    save_steps=self.hyperparams['save_steps'],
                    logging_steps=self.hyperparams['logging_steps'],
                    lr_scheduler_type=self.hyperparams['lr_scheduler'],
                    report_to="none",
                ),
            )
        self.model = model

    def train_one_round(self, round_idx):
        self._create_trainer(round_idx)
        
        # 训练
        self.trainer.train()
        # 记录训练开始时的学习率
        initial_lr = self.trainer.optimizer.param_groups[0]['lr']
        print(f"Training round {round_idx} - Initial LR: {initial_lr}")
        
        # 获取训练过程中的学习率变化
        lr_history = []
        for log in self.trainer.state.log_history:
            if 'learning_rate' in log:
                lr_history.append(log['learning_rate'])
        
        # 使用最后的学习率（如果调度器改变了学习率）
        if lr_history:
            final_lr = lr_history[-1]
            print(f"Training round {round_idx} - Final LR: {final_lr}")
            # 更新hyperparams中的学习率
            self.hyperparams['learning_rate'] = final_lr
        else:
            # 如果没有记录学习率变化，使用当前优化器中的学习率
            current_lr = self.trainer.optimizer.param_groups[0]['lr']
            print(f"Training round {round_idx} - Current LR: {current_lr}")
            self.hyperparams['learning_rate'] = current_lr
    

        # last_lr = self.trainer.optimizer.param_groups[0]['lr']
        # # 覆盖 self.hyperparams，让下一轮的 TrainingArguments 用这一把 lr
        # self.hyperparams['learning_rate'] = last_lr
        # print(last_lr)
        # Save model
        self.trainer.save_model(self.trainer.args.output_dir)
        self.tokenizer.save_pretrained(self.trainer.args.output_dir)
        return self.trainer

    def evaluate(self, trainer):
        # Extract last training loss entry
        loss_entries = [e['loss'] for e in trainer.state.log_history if 'loss' in e]
        print(loss_entries)
        train_loss = loss_entries if loss_entries else [0.0]
        # Load validation data
        if isinstance(self.val_dataset, str) and self.val_dataset.endswith('.json'):
            with open(self.val_dataset, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
        else:
            val_data = load_from_disk(self.val_dataset) if isinstance(self.val_dataset, str) else self.val_dataset
        # Select samples
        if hasattr(val_data, 'select'):
            samples = val_data.select(range(min(3, len(val_data))))
        else:
            samples = (val_data[:3] if isinstance(val_data, list) else val_data['data'][:3])
        # Generate and collect
        data = []
        for ex in samples:
            inp = ex['prompt']
            toks = self.tokenizer(inp, return_tensors="pt", padding=True).to(self.model.device)
            out = self.model.generate(**toks)
            pred = self.tokenizer.decode(out[0], skip_special_tokens=True)
            data.append({"question": inp, "solution": ex.get('solution', ''), "answer": pred})
        processed = process_data(data)
        acc, *_ = llm_as_judge(processed, self.llm_judge_token, model=self.llm_judge_model, url=self.llm_judge_url)
        fmt_ok = all(check_format(item['answer']) for item in data)
        return {"accuracy": acc, "format_ok": fmt_ok, "train_loss": train_loss}

    def plan_hyperparams(self, metrics_dict):
        prompt = f"""
Current training metrics:
- epoch: {metrics_dict['epoch']}
- train_loss_history: {self.train_loss_history}
- judge_score_history: {self.judge_score_history}
- current_lr: {self.hyperparams['learning_rate']}
- max_epochs: {self.hyperparams['max_steps']}

Please decide on the next action.
"""
        payload = {
            "model": self.llm_judge_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "n": 1,
            "stream": False
        }
        headers = {"Authorization": self.llm_judge_token, "Content-Type": "application/json"}
        try:
            resp = requests.post(self.llm_judge_url, json=payload, headers=headers)
            print(resp.json())
            content = resp.json()['choices'][0]['message']['content']
            print(content)
            return json.loads(content)
        except Exception:
            return {"stop": False, "new_lr": None, "extend_epochs": None, "reason": "error"}

    def run(self):
        best_acc, no_improve = 0.0, 0
        best_dir = None
        for rnd in range(1, self.max_rounds+1):
            self.current_epoch = rnd
            print(f"=== Starting training round {rnd} ===")
            trainer = self.train_one_round(rnd)
            metrics = self.evaluate(trainer)
            acc = metrics['accuracy']; loss = metrics['train_loss']
            print(f"Metrics: {metrics}")
            if acc > best_acc:
                best_acc, no_improve = acc, 0
                best_dir = trainer.args.output_dir
            else:
                no_improve += 1
            if no_improve >= 3: break
            self.train_loss_history += loss
            self.judge_score_history.append(acc)
            action = self.plan_hyperparams({
                'epoch': rnd,
                'train_loss_history': self.train_loss_history,
                'judge_score_history': self.judge_score_history
            })
            if action['stop']: break
            if action['new_lr'] is not None: self.hyperparams['learning_rate'] = action['new_lr']
            if action['extend_epochs'] is not None: self.hyperparams['max_steps'] += action['extend_epochs']
        if best_dir is None:
            best_dir = self.trainer.args.output_dir
        print("Training completed. Best at", best_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TrainingAgent with configurable parameters.")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--warmup_ratio", type=float, default=0.15)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_accuracy", type=float, default=0.90)
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--llm_judge_token", type=str, required=True, help="API token for LLM judge")
    parser.add_argument("--llm_judge_model", type=str, required=True, help="Model name for LLM judge")
    parser.add_argument("--llm_judge_url", type=str, required=True, help="API URL for LLM judge")
    args = parser.parse_args()
    initial_hparams = {
        'batch_size': args.batch_size,
        'grad_accum': args.grad_accum,
        'learning_rate': args.learning_rate,
        'max_steps': args.max_steps,
        'warmup_ratio': args.warmup_ratio,
        'save_steps': args.save_steps,
        'logging_steps': args.logging_steps,
        'lr_scheduler': args.lr_scheduler,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
    }
    agent = TrainingAgent(
        base_model_path=args.base_model_path,
        data_dir=args.data_dir,
        val_dataset=args.val_dataset,
        output_root=args.output_root,
        initial_hyperparams=initial_hparams,
        llm_judge_token=args.llm_judge_token,
        llm_judge_model=args.llm_judge_model,
        llm_judge_url=args.llm_judge_url,
        target_accuracy=args.target_accuracy,
        max_rounds=args.max_rounds
    )
    agent.run()

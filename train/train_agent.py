import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_from_disk
import re
from data.internTA2_evaluation_utils import llm_as_judge, process_data

def check_format(answer: str) -> bool:
    """
    Returns True if the model's answer contains the literal sequence:
        **</think>**</think>**
    """
    pattern = r".*</think>.*</think>.*"
    return bool(re.search(pattern, answer, flags=re.DOTALL))

class TrainingAgent:
    """
    TrainingAgent implements an automated Evaluator-Optimizer loop:
    1. Fine-tunes a student model via knowledge distillation.
    2. Uses an LLM as a critic to evaluate model outputs.
    3. Adjusts hyperparameters and iterates until target performance is met.
    """
    def __init__(
        self,
        base_model_path,
        data_dir,
        val_dataset,
        output_root,
        initial_hyperparams,
        llm_judge_token, llm_judge_model, llm_judge_url,
        target_accuracy=0.90,
        max_rounds=5
    ):
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            use_cache=False,
            device_map='auto'
        )
        # Configure LoRA for parameter-efficient fine-tuning
        self.peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            bias="none",
            r=initial_hyperparams['lora_r'],
            lora_alpha=initial_hyperparams['lora_alpha'],
            lora_dropout=initial_hyperparams['lora_dropout'],
            target_modules=[
                "q_proj", "k_proj", "v_proj",
                "up_proj", "down_proj", "gate_proj"
            ],
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
        self.llm_judge_url   = llm_judge_url

    def train_one_round(self, round_idx):
        """
        Run a single training round with current hyperparameters.
        """
        output_dir = os.path.join(self.output_root, f"round_{round_idx}")
        training_args = TrainingArguments(
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
        )
        # Load training dataset
        train_ds = load_from_disk(os.path.join(self.data_dir, "train_dataset"))
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_ds,
            peft_config=self.peft_config,
            tokenizer=self.tokenizer,
            args=training_args,
        )
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return trainer

    def evaluate(self, trainer):
        """
        1. Generates answers on a subset of the validation set.
        2. Calls llm_as_judge(...) to get (acc, eq, diff, amb) and uses acc as 'accuracy'.
        3. Uses check_format(...) to set 'format_ok'.
        Returns: {'accuracy': float, 'format_ok': bool}
        """
        # 1) collect question/solution/answer triples
        samples = self.val_dataset.select(range(min(50, len(self.val_dataset))))
        data = []
        for ex in samples:
            inp = ex['input']
            # generate with your trainer’s model
            toks = self.tokenizer(inp, return_tensors="pt", padding=True).to(trainer.model.device)
            out = trainer.model.generate(**toks)
            pred = self.tokenizer.decode(out[0], skip_special_tokens=True)
            data.append({
                "question": inp,
                "solution": ex['output'],
                "answer": pred
            })

        # 2) strip off any extra thinking tags, then judge
        processed = process_data(data)
        # you'll need to have set these on self:
        #   self.llm_judge_token, self.llm_judge_model, self.llm_judge_url
        acc, eq_cnt, diff_cnt, amb_cnt = llm_as_judge(
            processed,
            self.llm_judge_token,
            model=self.llm_judge_model,
            url=self.llm_judge_url
        )

        # 3) ensure every answer matches the format check
        fmt_ok = all(check_format(item['answer']) for item in data)

        return {
            "accuracy": acc,
            "format_ok": fmt_ok
        }

    def adjust_hyperparams(self, metrics):
        """
        Adjust hyperparameters based on evaluation metrics.
        """
        if metrics['accuracy'] < self.target_accuracy:
            # Reduce learning rate and increase training steps
            self.hyperparams['learning_rate'] *= 0.8
            self.hyperparams['max_steps'] = int(self.hyperparams['max_steps'] * 1.2)
        # Additional adjustment strategies can be added here

        def run(self):
            """
            Main loop: train, evaluate, adjust hyperparameters until performance targets are met or max rounds reached.
            Tracks and retains the checkpoint with the highest accuracy.
            """
            best_acc = 0.0
            best_round = None
            best_dir = None

            for round_idx in range(1, self.max_rounds + 1):
                print(f"\n=== Starting training round {round_idx} ===")
                trainer = self.train_one_round(round_idx)
                print("Evaluating model performance...")
                metrics = self.evaluate(trainer)
                acc = metrics.get('accuracy', 0.0)
                print(f"Evaluation metrics: {metrics}")

                # If this round is our new best, save its path
                output_dir = os.path.join(self.output_root, f"round_{round_idx}")
                if acc > best_acc:
                    best_acc = acc
                    best_round = round_idx
                    best_dir = output_dir
                    print(f"🎉 New best model at round {round_idx} (acc={best_acc:.4f}).")

                # If it's good enough, we can stop early
                if acc >= self.target_accuracy and metrics.get('format_ok', False):
                    print("Target performance reached. Stopping training loop.")
                    break

                # Otherwise adjust and continue
                print("Performance not sufficient. Updating hyperparameters and continuing.")
                self.adjust_hyperparams(metrics)

            print("\n=== Training completed ===")
            if best_round is not None:
                print(f"Best model was from round {best_round} with accuracy {best_acc:.4f}.")
                print(f"Checkpoint directory: {best_dir}")
                # If you want to reload it into self.model:
                # self.model = AutoModelForCausalLM.from_pretrained(best_dir, ...)
            else:
                print("No valid checkpoints were produced.")



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
    val_dataset = load_from_disk(args.val_dataset)

    agent = TrainingAgent(
        base_model_path=args.base_model_path,
        data_dir=args.data_dir,
        val_dataset=val_dataset,
        output_root=args.output_root,
        initial_hyperparams=initial_hparams,
        target_accuracy=args.target_accuracy,
        max_rounds=args.max_rounds
    )
    agent.run()

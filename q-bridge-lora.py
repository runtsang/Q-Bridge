import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import wandb
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    Mxfp4Config,
)

from peft import LoraConfig, TaskType, get_peft_model


COMMON_LORA_TARGET_MODULES: tuple[str, ...] = (
    "gate_proj",
    "down_proj",
    "up_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune gpt-oss-20b on the CML-2-QML dataset")
    parser.add_argument("--output_dir", type=str, default="./q-bridge-checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face access token")
    parser.add_argument("--wandb", type=str, required=True, help="Weights & Biases API key")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=32768, help="Maximum tokenized sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for the learning rate scheduler")
    parser.add_argument("--train_model", type=str, default="Qwen/Qwen3-8B", help="trained_model")
    parser.add_argument("--hub_model_id", type=str, default="runjiazeng/Q-Bridge", help="Hub repository to push the fine-tuned model")
    parser.add_argument("--run_name", type=str, default="q-bridge-gpt-oss-20b", help="Weights & Biases run name")
    parser.add_argument("--lora_rank", type=int, default=64, help="Rank of the LoRA adaptation matrices")
    parser.add_argument("--lora_alpha", type=float, default=128, help="Alpha scaling factor for LoRA layers")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help=(
            "Comma separated list of module names to apply LoRA to. Defaults to common transformer"
            " projection layers (gate_proj,down_proj,up_proj)."
        ),
    )
    return parser.parse_args()


def resolve_column(dataset, name: str) -> str:
    lower_lookup: Dict[str, str] = {col.lower(): col for col in dataset.column_names}
    if name.lower() not in lower_lookup:
        raise KeyError(f"Expected column '{name}' in dataset columns {dataset.column_names}")
    return lower_lookup[name.lower()]


def build_prompt(cml_text: str) -> str:
    return (
        "You are an expert quantum machine learning researcher. "
        "Translate the provided classical machine learning (CML) description into its quantum machine learning (QML) counterpart.\n\n"
        f"CML Description:\n{cml_text}\n\n"
        "QML Solution:"
    )


def build_deepspeed_config(args: argparse.Namespace) -> Dict[str, object]:
    train_micro_batch_size = max(1, args.per_device_train_batch_size)
    gradient_accumulation_steps = max(1, args.gradient_accumulation_steps)
    train_batch_size = train_micro_batch_size * gradient_accumulation_steps

    return {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": train_micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "stage3_prefetch_bucket_size": 50 * 1024 * 1024,
            "stage3_param_persistence_threshold": 1_000_000,
            "reduce_bucket_size": 500 * 1024 * 1024,
        },
        "bf16": {"enabled": True},
        "steps_per_print": 2000,
        "wall_clock_breakdown": False,
        "zero_force_ds_cpu_optimizer": False,
        "zero_allow_untested_optimizer": True,
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_PROJECT"] = "Q-Bridge"
    os.environ["WANDB_ENTITY"] = "runjia"
    os.environ["WANDB_NAME"] = args.run_name

    wandb.login(key=args.wandb)
    login(token=args.token)

    raw_dataset = load_dataset("runjiazeng/CML-2-QML", split="train")
    cml_col = resolve_column(raw_dataset, "cml")
    qml_col = resolve_column(raw_dataset, "qml")

    average_length_column = None
    for average_length_name in ("average length", "average_length", "averageLength"):
        try:
            average_length_column = resolve_column(raw_dataset, average_length_name)
            break
        except KeyError:
            continue
    if average_length_column is None:
        raise KeyError(
            "Expected an 'average length' column (average length, average_length, averageLength) in the dataset"
        )

    def include_example(example):
        value = example.get(average_length_column)
        try:
            return float(value) <= (args.max_length / 2)
        except (TypeError, ValueError):
            return True

    filtered_dataset = raw_dataset.filter(include_example)
    print(f"Total training samples after filtering: {filtered_dataset.num_rows}")

    def prepare_dataset(example):
        cml_text = example[cml_col]
        qml_text = example[qml_col]
        prompt = build_prompt(cml_text)
        return {"prompt": prompt, "response": qml_text}

    prompt_dataset = filtered_dataset.map(
        prepare_dataset, remove_columns=filtered_dataset.column_names
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.train_model,
        use_fast=False,
        trust_remote_code=True,
        use_auth_token=args.token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_function(example):
        prompt = example["prompt"]
        response = example["response"]
        full_text = prompt + "\n" + response + tokenizer.eos_token
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        labels = tokenized["input_ids"].copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        tokenized["labels"] = labels
        return tokenized

    tokenized_dataset = prompt_dataset.map(tokenize_function, remove_columns=prompt_dataset.column_names)

    @dataclass
    class QBridgeDataCollator:
        tokenizer: PreTrainedTokenizerBase
        label_pad_token_id: int = -100

        def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
            labels = [feature["labels"] for feature in features]
            # Remove labels before padding so the tokenizer does not try to convert
            # the ragged label sequences into tensors prematurely.
            encoded_features = [
                {k: v for k, v in feature.items() if k != "labels"}
                for feature in features
            ]

            batch = self.tokenizer.pad(
                encoded_features,
                padding=True,
                return_tensors="pt",
            )

            max_length = batch["input_ids"].size(1)
            label_tensor = torch.full(
                (len(labels), max_length),
                self.label_pad_token_id,
                dtype=torch.long,
            )
            for i, label in enumerate(labels):
                label_length = len(label)
                if label_length > max_length:
                    raise ValueError("Label sequence length exceeds padded input length")
                label_tensor[i, :label_length] = torch.tensor(label, dtype=torch.long)

            batch["labels"] = label_tensor
            return batch

    data_collator = QBridgeDataCollator(tokenizer=tokenizer)

    def get_lora_targets(model, requested: Optional[List[str]] = None) -> List[str]:
        available_modules = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                suffix = name.split(".")[-1]
                if suffix == "lm_head":
                    continue
                available_modules.add(suffix)

        candidate_modules: List[str]
        if requested:
            candidate_modules = [module for module in requested if module in available_modules]
            if candidate_modules:
                return candidate_modules
            raise ValueError(
                "None of the requested LoRA target modules are present in the model."
            )

        candidate_modules = [
            module for module in COMMON_LORA_TARGET_MODULES if module in available_modules
        ]
        if candidate_modules:
            return candidate_modules

        return sorted(available_modules)

    model = AutoModelForCausalLM.from_pretrained(
        args.train_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=args.token,
        quantization_config=Mxfp4Config(dequantize=True),
    )
    model.gradient_checkpointing_enable()
    if getattr(model.config, "use_cache", True):
        model.config.use_cache = False

    target_modules: Optional[List[str]]
    requested_modules: Optional[List[str]] = None
    if args.lora_target_modules:
        requested_modules = [module.strip() for module in args.lora_target_modules.split(",") if module.strip()]

    target_modules = get_lora_targets(model, requested_modules)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    is_distributed_training = False
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        is_distributed_training = local_rank != -1 or world_size > 1
    except ValueError:
        # Fall back to non-distributed training if the environment variables
        # are unexpectedly formatted.
        is_distributed_training = False

    deepspeed_config = build_deepspeed_config(args) if is_distributed_training else None

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=min(8, os.cpu_count() or 1),
        gradient_checkpointing=True,
        report_to=["wandb"],
        optim="adamw_torch",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False if is_distributed_training else None,
        deepspeed=deepspeed_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    trainer.push_to_hub(
        repo_id=args.hub_model_id,
        use_auth_token=args.token,
        commit_message="Add Q-Bridge gpt-oss-20b fine-tuned weights",
    )
    tokenizer.push_to_hub(args.hub_model_id, use_auth_token=args.token)


if __name__ == "__main__":
    main()

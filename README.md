# Q-Bridge Repository Guide

## Overview
Q-Bridge couples a large corpus of paired classical machine learning (CML) and quantum machine learning (QML) programs with tooling for fine-tuning language models that translate between the two paradigms. The repository provides:

- Thousands of curated CML/QML code pairs generated during the ML-2-QML project.
- Training scripts (`q-bridge.py`, `q-bridge-lora.py`) that fine-tune large language models on those pairs using either full-model or LoRA adaptation.
- Utilities for turning the curated corpus into a Hugging Face dataset and for reproducing the classical and quantum baselines included in the corpus (`seed_codebase`).

## Repository Layout
| Path | Description |
| ---- | ----------- |
| `q-bridge.py` | Full-parameter fine-tuning script that filters the ML-2-QML dataset, tokenizes paired prompts/responses, and launches a Transformers `Trainer` run with optional DeepSpeed support. 【F:q-bridge.py†L1-L203】 |
| `q-bridge-lora.py` | LoRA-based fine-tuning variant that discovers suitable projection layers, wraps the backbone with PEFT adapters, and otherwise mirrors `q-bridge.py`. 【F:q-bridge-lora.py†L1-L206】 |
| `ML-2-QML/build_hf_dataset.py` | Converts the ML/QML collection logs into a Hugging Face `DatasetDict`, splitting into train/test partitions and optionally pushing to the Hub. 【F:ML-2-QML/build_hf_dataset.py†L1-L152】 |
| `ML-2-QML/log.json` | Master log describing every ML/QML pair (average token lengths, scaling paradigm, references, and file system locations). |
| `ML-2-QML/ML/` | Classical counterparts for each numbered record referenced in `log.json`. Every `N.py` file reproduces the classical source for entry `N` (thousands of scripts, one per ID). |
| `ML-2-QML/QML/` | Quantum rewrites that correspond one-to-one with the classical scripts in `ML/`, stored under the same numeric filenames. |
| `ML-2-QML/Ansatz/` | Library of scaled quantum circuit ansatze. |
| `ML-2-QML/Feature_Map/` | Library of scaled quantum feature-map. |
| `ML-2-QML/ML/Feature_Map_log.json`, `ML-2-QML/ansatz_log.json` | Auxiliary logs that record provenance for feature-map and ansatz. |
| `seed_codebase/` | Self-contained classical and quantum baselines that mirror the dataset (classification and regression workflows, prompt templates, and generation helpers). |
| `seed_codebase/ML-Classification/` | Classical PyTorch pipelines for tabular classification, including dataset preparation, model definition, and validation utilities. |
| `seed_codebase/ML-Regression/` | Classical regression baseline with analogous prep/validate scripts and stored evaluation artefacts. |
| `seed_codebase/QML-Classification/` (Provided by [Priyabrata Senapati](https://github.com/psenap)) | Quantum classification workflows (ansatz/feature-map variations, training, and validation scripts). |
| `seed_codebase/QML-Regression/` (Provided by [Priyabrata Senapati](https://github.com/psenap)) | Quantum regression counterparts with VQE-style models and notebooks for TorchConnector demonstrations. |
| `seed_codebase/ML-Github/`, `seed_codebase/QML-Github/` | Curated reference implementations drawn from GitHub that inspired the ML/QML conversions (autoencoders, quantum classifiers, etc.). |
| `seed_codebase/generate.py`, `ansatz_generate.py`, `feature_map_generate.py` | Prompt-based helpers for producing new ML/QML/problem instances, feature maps, or ansatze with large language models. |
| `q-bridge.py`, `q-bridge-lora.py` dependencies | The scripts rely on Hugging Face Transformers, `datasets`, `wandb`, `peft`, and PyTorch; ensure these packages are installed before running. |

> **Note:** The ML/QML corpora contain thousands of numbered `.py` files. They are mechanically generated per dataset entry; the description above applies to every numbered file in those folders.

## Key Python Entry Points & Example Commands
The following commands illustrate how to execute each primary script. Replace placeholder values (enclosed in angle brackets) with environment-specific paths, tokens, or configuration values.

### Full Fine-Tuning (`q-bridge.py`)
```bash
python q-bridge.py \
  --token "<hf_token>" \
  --wandb "<wandb_api_key>" \
  --output_dir "<path_to_checkpoint_dir>" \
  --train_model "<backbone_model_id>" \
  --hub_model_id "<huggingface_repo>" \
  --run_name "<wandb_run_name>" \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 32768 \
  --warmup_ratio 0.03
```

### LoRA Fine-Tuning (`q-bridge-lora.py`)
```bash
python q-bridge-lora.py \
  --token "<hf_token>" \
  --wandb "<wandb_api_key>" \
  --output_dir "<path_to_checkpoint_dir>" \
  --train_model "<backbone_model_id>" \
  --hub_model_id "<huggingface_repo>" \
  --run_name "<wandb_run_name>" \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 32768 \
  --warmup_ratio 0.03 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_target_modules "gate_proj,down_proj,up_proj"
```

### Dataset Packaging (`ML-2-QML/build_hf_dataset.py`)
```bash
python ML-2-QML/build_hf_dataset.py \
  --log-path ML-2-QML/log.json \
  --repo-root "<path_to_repo_root>" \
  --repo-id "<hf_dataset_repo>" \
  --token "<hf_token>" \
  --branch "<optional_branch_name>"
```

### Classical Baseline Pipelines
```bash
# Prepare tabular classification datasets
python seed_codebase/ML-Classification/prep_classification_datasets.py

# Train and evaluate the MLP classifier
python seed_codebase/ML-Classification/validate_cls.py \
  --data seed_codebase/ML-Classification/datasets_cls/iris_cls.npz

# Prepare regression datasets
python seed_codebase/ML-Regression/prep_datasets.py

# Evaluate the classical regression baseline
python seed_codebase/ML-Regression/validate.py \
  --data seed_codebase/ML-Regression/datasets/iris_regression.npz
```

### Quantum Baseline Pipelines
```bash
# Prepare quantum classification datasets
python seed_codebase/QML-Classification/prep_classification_datasets.py

# Validate quantum classification models
python seed_codebase/QML-Classification/validate_cls.py \
  --data seed_codebase/QML-Classification/datasets_cls/iris_cls.npz \
  --ansatz seed_codebase/QML-Classification/ansatz/real_amplitudes.py \
  --feature-map seed_codebase/QML-Classification/feature_map/zz_feature_map.py

# Prepare quantum regression datasets
python seed_codebase/QML-Regression/prep_datasets.py

# Validate quantum regression models
python seed_codebase/QML-Regression/validate.py \
  --data seed_codebase/QML-Regression/datasets/iris_regression.npz \
  --ansatz seed_codebase/QML-Regression/ansatz_variations.py \
  --feature-map seed_codebase/QML-Regression/feature_map_variations.py
```

### Prompt-Based Generation Utilities
```bash
# Generate paired ML/QML problems with GPT-style prompts
python seed_codebase/generate.py \
  --prompt-file seed_codebase/gpt_oss_20b_scaling_prompt_multi.txt \
  --output-dir "<path_to_generated_pairs>"

# Generate new ansatz candidates
python seed_codebase/ansatz_generate.py \
  --prompt-file seed_codebase/gpt_oss_20b_ansatz_prompt.txt \
  --output-dir "<path_to_ansatz_candidates>"

# Generate feature maps
python seed_codebase/feature_map_generate.py \
  --prompt-file seed_codebase/gpt_oss_20b_feature_map_prompt.txt \
  --output-dir "<path_to_feature_maps>"
```

## Workflow Summary
1. Use the seed baselines to understand the classical and quantum tasks included in the ML-2-QML corpus.
2. Package or update the Hugging Face dataset with `ML-2-QML/build_hf_dataset.py` when new code pairs are added.
3. Fine-tune a language model with either full-parameter (`q-bridge.py`) or LoRA (`q-bridge-lora.py`) training, pushing results back to the Hub for deployment.
4. Iterate on ansatz or feature-map designs using the prompt generators, and contribute new ML/QML pairs by extending the numbered scripts and updating `log.json`.


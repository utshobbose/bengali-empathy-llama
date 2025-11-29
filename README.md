# Bengali Empathetic LLaMA 3.1 Fine-Tuning

Fine-tuning LLaMA 3.1-8B-Instruct on Bengali empathetic conversations using LoRA and Unsloth.

## Overview

This project fine-tunes Meta's LLaMA 3.1-8B-Instruct model on Bengali empathetic counseling conversations. It uses parameter-efficient fine-tuning (LoRA) with 4-bit quantization to train on Kaggle's free GPU.

## Features

- 4-bit quantized training with Unsloth
- Automatic checkpoint resumption for long training sessions
- Full sequence length maintained (1024 tokens)
- Comprehensive evaluation (BLEU, ROUGE-L, Perplexity)
- Structured logging for experiments and responses

## Installation

```bash
pip install -qU "unsloth[colab-new]" accelerate peft trl transformers==4.57.1 datasets sentencepiece evaluate sacrebleu rouge-score
```

**Important:** Restart runtime after installation.

## Quick Start

```python
from bengali_empathy_finetuner import BengaliEmpathyFineTuner

# Configure and train
config = {
    "num_train_epochs": 2.0,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "save_steps": 200,
    "eval_steps": 200,
}

ft = BengaliEmpathyFineTuner(config)
ft.load_raw_dataset()
ft.preprocess_text()
ft.build_tokenizer_and_model()
ft.prepare_instruction_dataset()
ft.apply_lora()
ft.train_model()
metrics = ft.evaluate_model()
ft.save_lora_adapters()
```

## Dataset

Bengali Empathetic Conversations Corpus with 38,105 examples split into:
- Train: 30,484 (80%)
- Validation: 3,810 (10%)
- Test: 3,811 (10%)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | LLaMA 3.1-8B-Instruct |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| Max Sequence Length | 1024 |
| Learning Rate | 2e-4 |
| Batch Size | 2 × 4 = 8 (effective) |
| Epochs | 2 |

## Checkpointing

The system automatically saves checkpoints every 200 steps and resumes training if interrupted. Perfect for Kaggle's 12-hour GPU limit.

## Evaluation Metrics

- **Perplexity:** Language modeling quality
- **BLEU:** N-gram overlap
- **ROUGE-L:** Longest common subsequence

## Logging

Two JSONL files track all experiments:
- `llama_empathy_experiments_experiments.jsonl` - Training metrics
- `llama_empathy_experiments_responses.jsonl` - Generated responses

## Project Structure

```
├── bengali_empathy_finetuner.py  # Main training script
├── llama31_bengali_empathy/      # Model checkpoints
├── logs/                          # Experiment logs
└── README.md
```

## Requirements

- Python 3.11+
- Kaggle GPU T4 x2 or Nvidia GPUs
- Kaggle notebook or Google colab
- HuggingFace token

## License

MIT License

# =====================================================================
# BENGALI EMPATHY FINE-TUNER - WINDOWS OPTIMIZED
# bengali_empathy_finetuner.py
# =====================================================================

import os
import re
import json
import time
import glob
import unicodedata
import random
from typing import Any, Dict, List, Optional
import gc

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    import evaluate
    HAVE_EVALUATE = True
except Exception:
    evaluate = None
    HAVE_EVALUATE = False

HF_TOKEN = os.environ.get("HF_TOKEN", None)
if not HF_TOKEN:
    raise ValueError("Please set HF_TOKEN environment variable or paste it in the code")
login(token=HF_TOKEN)


class BengaliEmpathyFineTuner:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg: Dict[str, Any] = {
            "data_path": r"E:\bengali-empathy-llama\data\bengali_empathetic_conversations.csv",
            "output_dir": r"E:\bengali-empathy-llama\outputs\llama31_bengali_empathy",
            "log_base_path": r"E:\bengali-empathy-llama\outputs\llama_empathy_experiments.db",

            "base_model_id_hf": "meta-llama/Llama-3.1-8B-Instruct",

            "use_unsloth": False,

            # ⚡ SPEED OPTIMIZATIONS ⚡
            "max_seq_length": 512,  # Further reduced for memory (was 768)
            "learning_rate": 3e-4,
            
            "num_train_epochs": 1.0,

            # Optimized batch size
            "per_device_train_batch_size": 2,  # Keep at 2 for stability
            "gradient_accumulation_steps": 16, # Increased to 16 (effective batch = 32)
            
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            
            # Reduced logging overhead
            "logging_steps": 50,
            "save_steps": 500,
            "eval_steps": 500,
            "save_total_limit": 2,

            "eval_max_new_tokens": 48,

            # Optimized LoRA config
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),

            # Memory optimizations
            "use_gradient_checkpointing": True,
            "max_grad_norm": 0.5,
            
            # Data sampling for speed (RECOMMENDED)
            "max_train_samples": 15000,  # Limit dataset to avoid memory issues

            "system_prompt": (
                "আপনি একজন সহানুভূতিশীল বাংলা কাউন্সেলর। "
                "আপনি খুব ধীরে, নম্রভাবে এবং সম্মানজনক ভঙ্গিতে উত্তর দেবেন। "
                "ব্যক্তির অনুভূতিকে স্বীকার করবেন, আশ্বাস দেবেন এবং প্রয়োজন হলে "
                "পেশাদার সাহায্য নেওয়ার পরামর্শ দেবেন, কিন্তু কোন চিকিৎসা বা আইনি পরামর্শ দেবেন না।"
            ),

            "seed": 42,
        }

        if config is not None:
            self.cfg.update(config)

        random.seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        torch.manual_seed(self.cfg["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg["seed"])

        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.dataset_dict: Optional[DatasetDict] = None

        self.tokenizer = None
        self.model = None
        self.trainer: Optional[Trainer] = None
        self.train_loss: Optional[float] = None

        log_dir = os.path.dirname(self.cfg["log_base_path"]) or "."
        base_name = os.path.splitext(os.path.basename(self.cfg["log_base_path"]))[0] or "experiments"
        os.makedirs(log_dir, exist_ok=True)
        self.experiments_file = os.path.join(log_dir, f"{base_name}_experiments.jsonl")
        self.responses_file = os.path.join(log_dir, f"{base_name}_responses.jsonl")
        self._next_experiment_id = self._init_next_experiment_id()

    # ----------------------------
    # Logging
    # ----------------------------
    def _init_next_experiment_id(self) -> int:
        if not os.path.exists(self.experiments_file):
            return 1
        last_id = 0
        with open(self.experiments_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        last_id = max(last_id, int(obj["id"]))
                except json.JSONDecodeError:
                    continue
        return last_id + 1

    def _write_jsonl(self, path: str, obj: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _log_experiment(self, train_loss: Optional[float], val_loss: Optional[float], metrics: Dict[str, Any]) -> int:
        exp_id = self._next_experiment_id
        self._next_experiment_id += 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        record = {
            "id": exp_id,
            "model_name": self.cfg["base_model_id_hf"],
            "strategy": "hf_qlora_optimized",
            "lora_config": {
                "r": self.cfg["lora_r"],
                "alpha": self.cfg["lora_alpha"],
                "dropout": self.cfg["lora_dropout"],
                "target_modules": list(self.cfg["lora_target_modules"]),
            },
            "train_loss": float(train_loss) if train_loss is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "metrics": metrics,
            "timestamp": timestamp,
        }
        self._write_jsonl(self.experiments_file, record)
        return exp_id

    def _log_responses(self, experiment_id: int, inputs: List[str], outputs: List[str]) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        for inp, out in zip(inputs, outputs):
            rec = {
                "experiment_id": experiment_id,
                "input_text": inp,
                "response_text": out,
                "timestamp": timestamp,
            }
            self._write_jsonl(self.responses_file, rec)

    # ----------------------------
    # Data
    # ----------------------------
    @staticmethod
    def _normalize_text(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def load_raw_dataset(self) -> pd.DataFrame:
        path = self.cfg["data_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")

        df_raw = pd.read_csv(path)

        # ---- auto-fix column names ----
        rename_map = {}
        cols = set(df_raw.columns)

        if "Topic" not in cols and "Topics" in cols:
            rename_map["Topics"] = "Topic"
        if "Question" not in cols and "Questions" in cols:
            rename_map["Questions"] = "Question"

        if rename_map:
            df_raw = df_raw.rename(columns=rename_map)

        expected_cols = {"Topic", "Question-Title", "Question", "Answers"}
        if not expected_cols.issubset(df_raw.columns):
            raise ValueError(
                f"Missing columns {expected_cols - set(df_raw.columns)}. Got: {set(df_raw.columns)}"
            )
        # -------------------------------

        records = []
        for i, row in df_raw.iterrows():
            topic = str(row["Topic"]) if not pd.isna(row["Topic"]) else ""
            title = str(row["Question-Title"]) if not pd.isna(row["Question-Title"]) else ""
            question = str(row["Question"]) if not pd.isna(row["Question"]) else ""
            answer = str(row["Answers"]) if not pd.isna(row["Answers"]) else ""

            user_parts = []
            if topic:
                user_parts.append(f"[{topic}]")
            if title:
                user_parts.append(title)
            if question:
                user_parts.append(question)
            user_text = " ".join(user_parts).strip()

            dialogue_id = i
            records.append({"dialogue_id": dialogue_id, "turn_id": 0, "role": "user", "text": user_text})
            records.append({"dialogue_id": dialogue_id, "turn_id": 1, "role": "assistant", "text": answer})

        self.raw_df = pd.DataFrame(records)
        print(f"✓ Loaded {len(self.raw_df)} records from dataset")
        return self.raw_df

    def preprocess_text(self) -> pd.DataFrame:
        if self.raw_df is None:
            raise RuntimeError("Call load_raw_dataset() first.")
        df = self.raw_df.copy()
        df["text"] = df["text"].astype(str).apply(self._normalize_text)
        df = df[df["text"].str.len() > 5].reset_index(drop=True)
        self.cleaned_df = df
        print(f"✓ Preprocessed {len(df)} valid records")
        
        # Clear memory
        del self.raw_df
        gc.collect()
        return df

    def _group_dialogues(self, df: pd.DataFrame):
        df = df.sort_values(["dialogue_id", "turn_id"])
        dialogues = []
        for did, group in df.groupby("dialogue_id"):
            turns = []
            for _, row in group.iterrows():
                turns.append({"role": str(row["role"]).lower(), "content": row["text"]})
            dialogues.append({"dialogue_id": did, "turns": turns})
        return dialogues

    def prepare_instruction_dataset(self) -> DatasetDict:
        if self.cleaned_df is None:
            raise RuntimeError("Call preprocess_text() first.")
        if self.tokenizer is None:
            raise RuntimeError("Call build_tokenizer_and_model() first.")

        dialogues = self._group_dialogues(self.cleaned_df)
        system_prompt = self.cfg["system_prompt"]

        examples = []
        print("Building training examples...")
        for dlg in tqdm(dialogues, desc="Processing dialogues"):
            turns = dlg["turns"]
            if len(turns) < 2:
                continue
            user_text = next((t["content"] for t in turns if t["role"] == "user"), None)
            assistant_text = next((t["content"] for t in turns if t["role"] == "assistant"), None)
            if not user_text or not assistant_text:
                continue

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})

            chat_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            examples.append({
                "text": chat_str,
                "user_text": user_text,
                "assistant_text": assistant_text,
            })

        print(f"✓ Built {len(examples)} training examples")

        # Sample dataset if configured
        if self.cfg["max_train_samples"] and len(examples) > self.cfg["max_train_samples"]:
            print(f"⚡ Sampling {self.cfg['max_train_samples']} examples for faster training & memory efficiency")
            random.shuffle(examples)
            examples = examples[:self.cfg["max_train_samples"]]

        dataset = Dataset.from_list(examples).shuffle(seed=self.cfg["seed"])
        train_test = dataset.train_test_split(test_size=0.15, seed=self.cfg["seed"])
        val_test = train_test["test"].train_test_split(test_size=0.33, seed=self.cfg["seed"])

        self.dataset_dict = DatasetDict({
            "train": train_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        })
        
        # Clear memory
        del examples, dialogues
        gc.collect()
        
        print(self.dataset_dict)
        return self.dataset_dict

    # ----------------------------
    # Model / Tokenizer
    # ----------------------------
    def build_tokenizer_and_model(self):
        print("[HF] Loading 4-bit model:", self.cfg["base_model_id_hf"])

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["base_model_id_hf"],
            token=HF_TOKEN,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg["base_model_id_hf"],
            token=HF_TOKEN,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        print("✓ Model and tokenizer loaded")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        return self.tokenizer, self.model

    def apply_lora(self):
        if self.model is None:
            raise RuntimeError("Call build_tokenizer_and_model() first.")

        print("[PEFT] Preparing for k-bit training + applying LoRA...")
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.cfg["use_gradient_checkpointing"]
        )

        if self.cfg["use_gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")

        lora_cfg = LoraConfig(
            r=self.cfg["lora_r"],
            lora_alpha=self.cfg["lora_alpha"],
            lora_dropout=self.cfg["lora_dropout"],
            target_modules=list(self.cfg["lora_target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()
        print("✓ LoRA applied")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        return self.model

    # ----------------------------
    # Checkpoints
    # ----------------------------
    def find_latest_checkpoint(self) -> Optional[str]:
        out = self.cfg["output_dir"]
        if not os.path.exists(out):
            return None
        checkpoints = glob.glob(os.path.join(out, "checkpoint-*"))
        if not checkpoints:
            return None

        def step_num(p: str) -> int:
            try:
                return int(os.path.basename(p).split("-")[-1])
            except Exception:
                return -1

        latest = sorted(checkpoints, key=step_num)[-1]
        print(f"✓ Found checkpoint: {latest}")
        return latest

    # ----------------------------
    # Training
    # ----------------------------
    def _tokenize_dataset(self) -> DatasetDict:
        assert self.dataset_dict is not None
        assert self.tokenizer is not None

        max_len = self.cfg["max_seq_length"]

        def tok(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_len,
                padding=False,
            )

        print("Tokenizing datasets (single process to avoid Windows pickle issues)...")
        # Use single process to avoid Windows multiprocessing/pickle issues
        return DatasetDict({
            "train": self.dataset_dict["train"].map(
                tok, batched=True, remove_columns=self.dataset_dict["train"].column_names,
                desc="Tokenizing train"
            ),
            "validation": self.dataset_dict["validation"].map(
                tok, batched=True, remove_columns=self.dataset_dict["validation"].column_names,
                desc="Tokenizing validation"
            ),
            "test": self.dataset_dict["test"].map(
                tok, batched=True, remove_columns=self.dataset_dict["test"].column_names,
                desc="Tokenizing test"
            ),
        })

    def train_model(self):
        if self.dataset_dict is None:
            raise RuntimeError("Call prepare_instruction_dataset() first.")
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call build_tokenizer_and_model() and apply_lora() first.")

        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        fp16 = torch.cuda.is_available() and not bf16

        args = TrainingArguments(
            output_dir=self.cfg["output_dir"],
            num_train_epochs=self.cfg["num_train_epochs"],

            per_device_train_batch_size=self.cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=self.cfg["gradient_accumulation_steps"],

            learning_rate=self.cfg["learning_rate"],
            warmup_ratio=self.cfg["warmup_ratio"],
            weight_decay=self.cfg["weight_decay"],
            max_grad_norm=self.cfg["max_grad_norm"],

            logging_steps=self.cfg["logging_steps"],
            eval_strategy="steps",
            eval_steps=self.cfg["eval_steps"],
            save_strategy="steps",
            save_steps=self.cfg["save_steps"],
            save_total_limit=self.cfg["save_total_limit"],

            bf16=bf16,
            fp16=fp16,

            report_to="none",
            optim="adamw_8bit",

            # Windows-specific optimizations
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            gradient_checkpointing=self.cfg["use_gradient_checkpointing"],
            
            # Reduce overhead
            load_best_model_at_end=False,
            metric_for_best_model=None,
            
            # Memory optimizations
            per_device_eval_batch_size=1,  # Lower eval batch size
            eval_accumulation_steps=4,      # Accumulate eval predictions
        )

        tokenized = self._tokenize_dataset()
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=collator,
            tokenizer=self.tokenizer,
        )

        # Clear memory before training
        del tokenized
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        latest = self.find_latest_checkpoint()
        if latest:
            print(f"🔄 Resuming training from: {latest}")
        else:
            print("🆕 Starting fresh training...")

        train_result = self.trainer.train(resume_from_checkpoint=latest)
        self.train_loss = getattr(train_result, "training_loss", None)
        if self.train_loss is not None:
            print(f"✓ Final training loss: {self.train_loss:.4f}")
        return self.trainer, self.train_loss

    # ----------------------------
    # Evaluation
    # ----------------------------
    def evaluate_model(self) -> Dict[str, Any]:
        if self.trainer is None:
            raise RuntimeError("Train the model first.")
        
        print("Running evaluation...")
        eval_results = self.trainer.evaluate()
        eval_loss = eval_results.get("eval_loss", None)
        perplexity = float(torch.exp(torch.tensor(eval_loss)).item()) if eval_loss is not None else None

        metrics = {
            "eval_loss": float(eval_loss) if eval_loss is not None else None,
            "perplexity": perplexity,
        }

        # Skip generation metrics if evaluate not available
        if not HAVE_EVALUATE or self.dataset_dict is None:
            exp_id = self._log_experiment(self.train_loss, eval_loss, metrics)
            print(f"✓ Experiment {exp_id} logged")
            return metrics

        bleu_metric = evaluate.load("sacrebleu")
        rouge_metric = evaluate.load("rouge")

        device = next(self.model.parameters()).device

        eval_ds = self.dataset_dict["validation"]
        max_samples = min(15, len(eval_ds))  # Reduced for memory
        subset = eval_ds.select(range(max_samples))

        preds, refs = [], []
        inputs_logged, outputs_logged = [], []

        print(f"Generating {max_samples} samples for BLEU/ROUGE...")
        for ex in tqdm(subset, desc="Generating"):
            user_text = ex["user_text"]
            ref = ex["assistant_text"]

            messages = [
                {"role": "system", "content": self.cfg["system_prompt"]},
                {"role": "user", "content": user_text},
            ]
            chat_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            enc = self.tokenizer(chat_str, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                gen_ids = self.model.generate(
                    **enc,
                    max_new_tokens=self.cfg["eval_max_new_tokens"],
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                )

            gen_only = gen_ids[0][enc["input_ids"].shape[1]:]
            pred_text = self.tokenizer.decode(gen_only, skip_special_tokens=True).strip()

            preds.append(pred_text)
            refs.append(ref)
            inputs_logged.append(user_text)
            outputs_logged.append(pred_text)

        bleu_score = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])["score"]
        rouge_scores = rouge_metric.compute(predictions=preds, references=refs)
        rouge_l = float(rouge_scores.get("rougeL", 0.0))

        metrics.update({"bleu": float(bleu_score), "rougeL": rouge_l})

        exp_id = self._log_experiment(self.train_loss, eval_loss, metrics)
        self._log_responses(exp_id, inputs_logged, outputs_logged)
        print(f"✓ Experiment {exp_id} logged with BLEU/ROUGE metrics")
        return metrics

    def save_lora_adapters(self):
        os.makedirs(self.cfg["output_dir"], exist_ok=True)
        self.model.save_pretrained(self.cfg["output_dir"])
        self.tokenizer.save_pretrained(self.cfg["output_dir"])
        print(f"✓ Saved adapters + tokenizer to: {self.cfg['output_dir']}")


if __name__ == "__main__":
    print("=" * 70)
    print("BENGALI EMPATHY FINE-TUNER - WINDOWS OPTIMIZED")
    print("Target: 5-6 hours training time")
    print("=" * 70)

    config = {
        "num_train_epochs": 1.0,
        "use_unsloth": False,
        
        # Dataset limited to 15k samples for speed + memory
        "max_train_samples": 15000,
        
        # Adjust these if you have more/less GPU memory:
        # "per_device_train_batch_size": 2,  # Increase if you have >12GB VRAM
        # "max_seq_length": 512,              # Can go up to 768 if memory allows
    }

    ft = BengaliEmpathyFineTuner(config)

    try:
        print("\n[1/7] Loading raw dataset...")
        ft.load_raw_dataset()

        print("\n[2/7] Preprocessing text...")
        ft.preprocess_text()

        print("\n[3/7] Loading model & tokenizer...")
        ft.build_tokenizer_and_model()

        print("\n[4/7] Preparing instruction dataset...")
        ft.prepare_instruction_dataset()

        print("\n[5/7] Applying LoRA...")
        ft.apply_lora()

        print("\n[6/7] Training model...")
        ft.train_model()

        print("\n[7/7] Evaluating model...")
        metrics = ft.evaluate_model()
        print("\nFinal Metrics:", metrics)

        print("\nSaving adapters...")
        ft.save_lora_adapters()

        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("Your progress has been saved. Run again to resume from last checkpoint.")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
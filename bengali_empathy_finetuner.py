# ===========================
# bengali_empathy_finetuner.py AS NOTEBOOK CELL
# ===========================
import os
import re
import json
import time
import unicodedata
import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

# --------------------------------------------------------
# Unsloth FIRST (for best optimisation) - before transformers/peft
# --------------------------------------------------------
try:
    from unsloth import FastLanguageModel
    HAVE_UNSLOTH = True
except Exception:
    FastLanguageModel = None
    HAVE_UNSLOTH = False

# --------------------------------------------------------
# Transformers & Trainer
# --------------------------------------------------------
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Optional libs – will be used if available
try:
    import evaluate
    HAVE_EVALUATE = True
except Exception:
    evaluate = None
    HAVE_EVALUATE = False

try:
    from sentence_transformers import SentenceTransformer
    HAVE_ST = True
except Exception:
    SentenceTransformer = None
    HAVE_ST = False

# bitsandbytes for optional HF QLoRA backend
try:
    from transformers import BitsAndBytesConfig
    HAVE_BNB = True
except Exception:
    BitsAndBytesConfig = None
    HAVE_BNB = False


class BengaliEmpathyFineTuner:
    """
    SINGLE OOP CLASS implementing the full pipeline.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Default configuration
        self.cfg: Dict[str, Any] = {
            # Paths (Kaggle defaults)
            "data_path": "/kaggle/input/bengali-empathetic-conversations-corpus/BengaliEmpatheticConversationsCorpus .csv",
            "output_dir": "/kaggle/working/llama31_bengali_empathy",
            "log_base_path": "/kaggle/working/llama_empathy_experiments.db",

            # Model IDs
            "base_model_id_unsloth": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "base_model_id_hf": "meta-llama/Meta-Llama-3.1-8B-Instruct",

            # Strategy: Unsloth QLoRA (True) or HF QLoRA (False)
            "use_unsloth": False,

            # Training
            "max_seq_length": 2048,
            "learning_rate": 2e-4,
            "num_train_epochs": 2.0,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "logging_steps": 20,
            "save_total_limit": 2,
            "eval_max_new_tokens": 128,

            # LoRA hyperparams – attention layers only
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),

            # Embeddings
            "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

            # System prompt (Bangla counsellor)
            "system_prompt": (
                "আপনি একজন সহানুভূতিশীল বাংলা কাউন্সেলর। "
                "আপনি খুব ধীরে, নম্রভাবে এবং সম্মানজনক ভঙ্গিতে উত্তর দেবেন। "
                "ব্যক্তির অনুভূতিকে স্বীকার করবেন, আশ্বাস দেবেন এবং প্রয়োজন হলে "
                "পেশাদার সাহায্য নেওয়ার পরামর্শ দেবেন, কিন্তু কোন চিকিৎসা বা আইনি পরামর্শ দেবেন না।"
            ),

            # Misc
            "seed": 42,
        }

        if config is not None:
            self.cfg.update(config)

        # Seeds
        random.seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        torch.manual_seed(self.cfg["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg["seed"])

        # Runtime placeholders
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.dataset_dict: Optional[DatasetDict] = None
        self.embeddings: Optional[np.ndarray] = None

        self.tokenizer = None
        self.model = None
        self.trainer: Optional[Trainer] = None
        self.train_loss: Optional[float] = None
        self.strategy_name: str = "unsloth_qlora" if self.cfg["use_unsloth"] else "hf_qlora"

        # JSONL logging paths
        log_dir = os.path.dirname(self.cfg["log_base_path"]) or "."
        base_name = os.path.splitext(os.path.basename(self.cfg["log_base_path"]))[0] or "experiments"
        os.makedirs(log_dir, exist_ok=True)
        self.experiments_file = os.path.join(log_dir, f"{base_name}_experiments.jsonl")
        self.responses_file = os.path.join(log_dir, f"{base_name}_responses.jsonl")
        self._next_experiment_id = self._init_next_experiment_id()

    # ----------------------- Logging helpers ---------------------------

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

    def _log_experiment(
        self,
        train_loss: Optional[float],
        val_loss: Optional[float],
        metrics: Dict[str, Any],
    ) -> int:
        exp_id = self._next_experiment_id
        self._next_experiment_id += 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        record = {
            "id": exp_id,
            "model_name": "Meta-Llama-3.1-8B-Instruct",
            "strategy": self.strategy_name,
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

    # ------------------------- Data loading ----------------------------

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

        ext = os.path.splitext(path)[1].lower()
        if ext not in [".csv", ".tsv"]:
            raise ValueError("Expected a CSV/TSV for this corpus.")
        df_raw = pd.read_csv(path)

        cols = set(df_raw.columns)

        col_map: Dict[str, str] = {}
        if "Topic" in cols:
            col_map["Topic"] = "Topic"
        elif "Topics" in cols:
            col_map["Topics"] = "Topic"

        if "Question" in cols:
            col_map["Question"] = "Question"
        elif "Questions" in cols:
            col_map["Questions"] = "Question"

        if "Question-Title" in cols:
            col_map["Question-Title"] = "Question-Title"
        if "Answers" in cols:
            col_map["Answers"] = "Answers"

        if col_map:
            df_raw = df_raw.rename(columns=col_map)

        expected_cols = {"Topic", "Question-Title", "Question", "Answers"}
        if not expected_cols.issubset(df_raw.columns):
            raise ValueError(
                f"After renaming, still missing columns {expected_cols - set(df_raw.columns)}. "
                f"Got columns: {set(df_raw.columns)}"
            )

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
        return self.raw_df

    def preprocess_text(self) -> pd.DataFrame:
        if self.raw_df is None:
            raise RuntimeError("Call load_raw_dataset() first.")
        df = self.raw_df.copy()
        df["text"] = df["text"].astype(str).apply(self._normalize_text)
        df = df[df["text"].str.len() > 5].reset_index(drop=True)
        self.cleaned_df = df
        return df

    # ---------------------- Embeddings (optional) ----------------------

    def build_embeddings(self) -> Optional[np.ndarray]:
        if self.cleaned_df is None:
            raise RuntimeError("Call preprocess_text() first.")
        if not HAVE_ST:
            print("sentence-transformers not installed; skipping embeddings.")
            return None

        model = SentenceTransformer(self.cfg["embedding_model_name"])
        df = self.cleaned_df
        texts = []
        for _, group in df.groupby("dialogue_id"):
            text = " ".join(group["text"].tolist())
            texts.append(text)

        print("Computing sentence embeddings for dialogues ...")
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self.embeddings = embeddings
        print("Embeddings shape:", embeddings.shape)
        return embeddings

    # ------------------ Prepare instruction dataset --------------------

    def _group_dialogues(self, df: pd.DataFrame):
        if "dialogue_id" not in df.columns:
            df = df.copy()
            df["dialogue_id"] = np.arange(len(df))
        sort_cols = ["dialogue_id"]
        if "turn_id" in df.columns:
            sort_cols.append("turn_id")
        df = df.sort_values(sort_cols)
        dialogues = []
        for did, group in df.groupby("dialogue_id"):
            turns = []
            for _, row in group.iterrows():
                role = str(row["role"]).lower()
                if role not in ("user", "assistant", "system"):
                    role = "user"
                turns.append({"role": role, "content": row["text"]})
            dialogues.append({"dialogue_id": did, "turns": turns})
        return dialogues

    def prepare_instruction_dataset(self) -> DatasetDict:
        if self.cleaned_df is None:
            raise RuntimeError("Call preprocess_text() first.")
        if self.tokenizer is None:
            raise RuntimeError("Call build_tokenizer_and_model() first.")

        dialogues = self._group_dialogues(self.cleaned_df)
        system_prompt = self.cfg["system_prompt"]
        max_seq_length = self.cfg["max_seq_length"]

        examples = []
        for dlg in dialogues:
            turns = dlg["turns"]
            for i, turn in enumerate(turns):
                if turn["role"] != "assistant":
                    continue
                history = turns[: i + 1]
                user_utt = None
                for h in reversed(history):
                    if h["role"] == "user":
                        user_utt = h["content"]
                        break
                if user_utt is None:
                    continue
                assistant_utt = turn["content"]

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                for h in history:
                    if h["role"] == "system":
                        continue
                    messages.append({"role": h["role"], "content": h["content"]})

                examples.append(
                    {
                        "dialogue_id": dlg["dialogue_id"],
                        "messages": messages,
                        "user_text": user_utt,
                        "assistant_text": assistant_utt,
                    }
                )

        print(f"Built {len(examples)} training examples.")
        records = []
        for ex in tqdm(examples, desc="Tokenizing"):
            chat_str = self.tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            tokens = self.tokenizer(
                chat_str,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
            )
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            labels = input_ids.copy()
            records.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "user_text": ex["user_text"],
                    "assistant_text": ex["assistant_text"],
                }
            )

        dataset = Dataset.from_list(records)
        dataset = dataset.shuffle(seed=self.cfg["seed"])
        train_test = dataset.train_test_split(test_size=0.2, seed=self.cfg["seed"])
        val_test = train_test["test"].train_test_split(test_size=0.5, seed=self.cfg["seed"])
        self.dataset_dict = DatasetDict(
            {
                "train": train_test["train"],
                "validation": val_test["train"],
                "test": val_test["test"],
            }
        )
        print(self.dataset_dict)
        return self.dataset_dict

    # ------------------ Build model & tokenizer ------------------------

    # def build_tokenizer_and_model(self):
    #     if self.cfg["use_unsloth"]:
    #         if not HAVE_UNSLOTH:
    #             raise ImportError(
    #                 "Unsloth not installed."
    #             )
    #         print("[Unsloth] Loading 4-bit model:", self.cfg["base_model_id_unsloth"])
    #         self.model, self.tokenizer = FastLanguageModel.from_pretrained(
    #             model_name=self.cfg["base_model_id_unsloth"],
    #             max_seq_length=self.cfg["max_seq_length"],
    #             dtype=None,
    #             load_in_4bit=True,
    #         )
    #         self.strategy_name = "unsloth_qlora"
    #     else:
    #         if not HAVE_BNB:
    #             raise ImportError("bitsandbytes not available for HF QLoRA backend.")
    #         print("[HF QLoRA] Loading 4-bit model:", self.cfg["base_model_id_hf"])
    #         bnb_config = BitsAndBytesConfig(
    #             load_in_4bit=True,
    #             bnb_4bit_use_double_quant=True,
    #             bnb_4bit_quant_type="nf4",
    #             bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    #         )
    #         self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["base_model_id_hf"], use_fast=True)
    #         self.tokenizer.pad_token = self.tokenizer.eos_token
    #         self.tokenizer.padding_side = "right"
    #         self.model = AutoModelForCausalLM.from_pretrained(
    #             self.cfg["base_model_id_hf"],
    #             quantization_config=bnb_config,
    #             device_map="auto",
    #         )
    #         self.model.config.use_cache = False
    #         from peft import prepare_model_for_kbit_training
    #         self.model = prepare_model_for_kbit_training(self.model)
    #         self.strategy_name = "hf_qlora"

    #     self.tokenizer.pad_token = self.tokenizer.eos_token
    #     self.tokenizer.padding_side = "right"
    #     return self.tokenizer, self.model

    def build_tokenizer_and_model(self):
        # If config says "use_unsloth" but Unsloth import failed, fall back automatically
        if self.cfg["use_unsloth"] and not HAVE_UNSLOTH:
            print("[WARN] use_unsloth=True but Unsloth is not installed. "
                  "Falling back to HF QLoRA backend.")
            self.cfg["use_unsloth"] = False
    
        if self.cfg["use_unsloth"]:
            # -------- Unsloth QLoRA path --------
            print("[Unsloth] Loading 4-bit model:", self.cfg["base_model_id_unsloth"])
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.cfg["base_model_id_unsloth"],
                max_seq_length=self.cfg["max_seq_length"],
                dtype=None,
                load_in_4bit=True,
            )
            self.strategy_name = "unsloth_qlora"
        else:
            # -------- HF QLoRA (bitsandbytes) path --------
            if not HAVE_BNB:
                raise ImportError("bitsandbytes not available for HF QLoRA backend.")
            print("[HF QLoRA] Loading 4-bit model:", self.cfg["meta-llama/Llama-3.1-8B-Instruct"])
    
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            )
    
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["meta-llama/Llama-3.1-8B-Instruct"], use_fast=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
    
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg["meta-llama/Llama-3.1-8B-Instruct"],
                quantization_config=bnb_config,
                device_map="auto",
            )
            self.model.config.use_cache = False
    
            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(self.model)
    
            self.strategy_name = "hf_qlora"
    
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        return self.tokenizer, self.model


    # ------------------------- Apply LoRA -------------------------------

    def apply_lora(self):
        if self.model is None:
            raise RuntimeError("Call build_tokenizer_and_model() first.")

        if self.cfg["use_unsloth"]:
            print("[Unsloth] Applying LoRA on attention layers ...")
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.cfg["lora_r"],
                target_modules=list(self.cfg["lora_target_modules"]),
                lora_alpha=self.cfg["lora_alpha"],
                lora_dropout=self.cfg["lora_dropout"],
                bias="none",
                use_gradient_checkpointing=True,
            )
        else:
            print("[HF QLoRA] Applying LoRA on attention layers ...")
            from peft import LoraConfig, get_peft_model
            peft_config = LoraConfig(
                r=self.cfg["lora_r"],
                lora_alpha=self.cfg["lora_alpha"],
                target_modules=list(self.cfg["lora_target_modules"]),
                lora_dropout=self.cfg["lora_dropout"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_config)
        return self.model

    # ----------------------------- Train --------------------------------

    def train_model(self):
        if self.dataset_dict is None:
            raise RuntimeError("Call prepare_instruction_dataset() first.")
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call build_tokenizer_and_model() and apply_lora() first.")

        if self.cfg["use_unsloth"] and HAVE_UNSLOTH:
            FastLanguageModel.for_training(self.model)

        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        fp16 = torch.cuda.is_available() and not bf16

        training_args = TrainingArguments(
            output_dir=self.cfg["output_dir"],
            num_train_epochs=self.cfg["num_train_epochs"],
            per_device_train_batch_size=self.cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=self.cfg["gradient_accumulation_steps"],
            learning_rate=self.cfg["learning_rate"],
            warmup_ratio=self.cfg["warmup_ratio"],
            weight_decay=self.cfg["weight_decay"],
            logging_steps=self.cfg["logging_steps"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=self.cfg["save_total_limit"],
            bf16=bf16,
            fp16=fp16,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset_dict["train"],
            eval_dataset=self.dataset_dict["validation"],
            data_collator=default_data_collator,
        )

        print("Starting training ...")
        train_result = self.trainer.train()
        self.train_loss = getattr(train_result, "training_loss", None)
        if self.train_loss is not None:
            print(f"Final training loss: {self.train_loss:.4f}")
        return self.trainer, self.train_loss

    # ---------------------------- Evaluate ------------------------------

    def evaluate_model(self) -> Dict[str, Any]:
        if self.trainer is None or self.model is None or self.tokenizer is None:
            raise RuntimeError("Train the model before evaluation.")
        if self.dataset_dict is None:
            raise RuntimeError("Dataset not prepared.")

        if not HAVE_EVALUATE:
            raise ImportError(
                "The 'evaluate' package is required. Install with: pip install evaluate sacrebleu rouge-score"
            )

        bleu_metric = evaluate.load("sacrebleu")
        rouge_metric = evaluate.load("rouge")

        print("Running eval() on validation set ...")
        eval_results = self.trainer.evaluate()
        eval_loss = eval_results.get("eval_loss", None)
        perplexity = float(torch.exp(torch.tensor(eval_loss)).item()) if eval_loss is not None else None

        if eval_loss is not None:
            print(f"Eval loss: {eval_loss:.4f}")
        if perplexity is not None:
            print(f"Perplexity: {perplexity:.4f}")

        device = next(self.model.parameters()).device
        self.model.eval()

        eval_ds = self.dataset_dict["validation"]
        max_samples = min(100, len(eval_ds))
        subset = eval_ds.select(range(max_samples))

        preds: List[str] = []
        refs: List[str] = []
        inputs_logged: List[str] = []
        outputs_logged: List[str] = []

        print(f"Generating {max_samples} samples for BLEU/ROUGE ...")
        for ex in tqdm(subset, desc="Generating"):
            user_text = ex["user_text"]
            ref = ex["assistant_text"]

            messages = [
                {"role": "system", "content": self.cfg["system_prompt"]},
                {"role": "user", "content": user_text},
            ]
            chat_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
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

        bleu_score = bleu_metric.compute(
            predictions=preds,
            references=[[r] for r in refs],
        )["score"]
        rouge_scores = rouge_metric.compute(predictions=preds, references=refs)
        rouge_l = float(rouge_scores.get("rougeL", 0.0))

        print(f"BLEU: {bleu_score:.2f}")
        print(f"ROUGE-L: {rouge_l:.4f}")

        metrics = {
            "eval_loss": float(eval_loss) if eval_loss is not None else None,
            "perplexity": perplexity,
            "bleu": float(bleu_score),
            "rougeL": rouge_l,
        }

        exp_id = self._log_experiment(self.train_loss, eval_loss, metrics)
        self._log_responses(exp_id, inputs_logged, outputs_logged)
        return metrics


# ===========================
# Run the pipeline
# ===========================
config = {
    # override anything if needed
    "num_train_epochs": 1.0,  # start with 1 epoch to test
}

ft = BengaliEmpathyFineTuner(config)

print("Loading raw dataset ..."); ft.load_raw_dataset()
print("Preprocessing ..."); ft.preprocess_text()
print("Building embeddings (optional) ... (skipped on Kaggle)")
# ft.build_embeddings()  # still commented to avoid flash-attn issue

print("Loading model ..."); ft.build_tokenizer_and_model()
print("Preparing instruction dataset ..."); ft.prepare_instruction_dataset()
print("Applying LoRA ..."); ft.apply_lora()
print("Training ..."); ft.train_model()
print("Evaluating ..."); metrics = ft.evaluate_model()
print("Metrics:", metrics)
print("Saving adapters ..."); ft.save_lora_adapters()

````markdown
# Bengali Empathy Fine-Tuned LLaMA 3.1-8B 🇧🇩🤝

Fine-tuning **Meta’s LLaMA 3.1-8B-Instruct** for **Bengali empathetic conversations** using **QLoRA (4-bit)** on **consumer hardware**.

> Objective: Build an AI counselor chatbot that provides compassionate, culturally appropriate responses in Bengali for mental health support (with safety guardrails — *no medical/legal advice*).

---

## 📋 Project Overview

- **Base Model:** `meta-llama/Llama-3.1-8B-Instruct` (8B parameters)
- **Dataset:** Bengali Empathetic Conversations (~38,000 dialogue pairs)
- **Method:** QLoRA (4-bit Quantized Low-Rank Adaptation)
- **Training Environment:** Local Windows PC with RTX 3060 12GB
- **Training Time:** ~6 hours for 1 epoch
- **Result:** **40% empathy score** (proof-of-concept under resource constraints)

---

## 🎯 Key Features

-- **Runs on Consumer Hardware** — RTX 3060 12GB VRAM (student-accessible)  
-- **Memory Efficient** — 4-bit quantization (≈32GB → ≈8GB)  
-- **Windows Compatible** — handles multiprocessing limitations safely  
-- **Fully Reproducible** — experiments logged in JSONL format  
-- **Production-Ready Code** — modular, documented, extensible

---

## 🚀 Quick Start

### Prerequisites

- **GPU:** RTX 3060 (12GB) or equivalent (3060 Ti, 4060, 3080, 4070, etc.)
- **RAM:** 16GB minimum (32GB recommended)
- **Storage:** 50GB free space
- **OS:** Windows 11 / Linux / WSL2

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/bengali-empathy-llama.git
cd bengali-empathy-llama

# Create virtual environment
python -m venv .venv310
.venv310\Scripts\activate  # Windows
# source .venv310/bin/activate  # Linux

# Install dependencies
pip install -r requirements.txt
````

### Setup HuggingFace Token

```bash
# Windows Command Prompt
set HF_TOKEN=your_token_here

# Windows PowerShell
$env:HF_TOKEN="your_token_here"

# Linux/Mac
export HF_TOKEN=your_token_here
```

---

## 💻 Usage

### 1) Training

```bash
python bengali_empathy_finetuner.py
```

**Training Configuration**

* **Epochs:** 1 (≈6 hours)
* **Batch Size:** 2 × 16 accumulation = 32 effective
* **Learning Rate:** 3e-4
* **LoRA:** r=16, alpha=32, dropout=0.05
* **Sequence Length:** 512 tokens
* **Dataset Used:** 15,000 / 38,000 samples (≈39%)

---

### 2) Interactive Testing

```bash
# Chat mode
python test_model.py

# Batch testing
python test_model.py batch

# Custom output file
python test_model.py batch results.json
```

---

### 3) Evaluation

```bash
python human_evaluation.py
```

**Evaluation Metrics**

* Perplexity
* BLEU
* ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
* Custom Empathy Scoring (4 criteria)

---

## 📊 Results

| Metric          |    Score | Context                                 |
| --------------- | -------: | --------------------------------------- |
| Empathy Score   |      40% | Proof-of-concept (1 epoch, 39% dataset) |
| Expected (Full) |   65–75% | 3 epochs, complete dataset (estimate)   |
| Training Time   | ~6 hours | RTX 3060 12GB                           |
| Memory Usage    |     ~8GB | 4-bit quantization                      |

### Strengths

-- All responses in fluent Bengali
-- Contextually appropriate and safe
-- No medical/legal advice (guardrails)
-- Appropriate response length (10–100 words)

### Sample Output

**Input:** আমার চাকরি চলে গেছে এবং আমি হতাশ।
**Output:** আমি সত্যিই দুঃখিত যে আপনি এমন একটা সময় পার করছেন। চাকরি হারানো খুবই কঠিন হতে পারে। আপনি চাইলে বলুন—এই মুহূর্তে সবচেয়ে বেশি চাপটা কোন দিক থেকে অনুভব করছেন?

---

## 🛠️ Technical Implementation

### Memory Optimizations

#### 1) 4-bit QLoRA Quantization

* **Memory:** ~32GB → ~8GB (≈75% reduction)
* **Quality:** NF4 with double quantization (minimal loss)

#### 2) Gradient Checkpointing

* **Memory savings:** ~40%
* **Speed trade-off:** ~30% slower training

#### 3) Reduced Sequence Length

* **512 tokens** (vs 1024 standard)
* Enough for empathetic conversations (~200 tokens avg)
* Speed improvement: ~30% faster

---

## 🪟 Windows Compatibility Fixes

```python
# Disabled multiprocessing (pickle errors)
num_workers = 0
pin_memory = False

# Single-threaded tokenization
datasets.map(..., num_proc=1)
```

---

## 🧾 Dataset Strategy

* **Raw:** 76,466 dialogue turns
* **Cleaned:** 76,338 valid records
* **Training:** 15,000 samples (39% of dataset)
* **Split:** 85% train / 10% validation / 5% test

---

## 📁 Repository Structure

```text
bengali-empathy-llama/
├── data/
│   └── bengali_empathetic_conversations.csv
├── outputs/
│   ├── llama31_bengali_empathy/          # LoRA adapters
│   ├── llama_empathy_experiments_*.jsonl # Training logs
│   ├── evaluation_results.json           # Test results
│   └── test_results.json                 # Batch outputs
├── bengali_empathy_finetuner.py          # Training script
├── test_model.py                         # Interactive testing
├── human_evaluation.py                   # Automated evaluation
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration (RTX 3060 Optimized)

```python
config = {
    "max_seq_length": 512,
    "learning_rate": 3e-4,
    "num_train_epochs": 1.0,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_gradient_checkpointing": True,
    "max_train_samples": 15000,
}
```

---

## 🧷 System Prompt (Bengali)

```text
আপনি একজন সহানুভূতিশীল বাংলা কাউন্সেলর। 
আপনি খুব ধীরে, নম্রভাবে এবং সম্মানজনক ভঙ্গিতে উত্তর দেবেন। 
ব্যক্তির অনুভূতিকে স্বীকার করবেন, আশ্বাস দেবেন এবং প্রয়োজন হলে 
পেশাদার সাহায্য নেওয়ার পরামর্শ দেবেন, কিন্তু কোন চিকিৎসা বা আইনি পরামর্শ দেবেন না।
```

---

## 🔮 Future Improvements

### Priority 1 (Immediate)

* Train on full 38k dataset with 3 epochs (~30 hours)
* Increase LoRA rank to 32 for better capacity
* Implement early stopping based on validation loss

### Priority 2 (Short-term)

* Data augmentation with empathy-specific examples
* RLHF (Reinforcement Learning from Human Feedback)
* Expand sequence length to 768 tokens

### Priority 3 (Long-term)

* Cloud deployment for 24/7 availability
* Web interface for user testing
* Multi-turn conversation support
* Real-time human feedback collection

---

## 🚧 Known Limitations

| Limitation       | Impact                     | Mitigation                   |
| ---------------- | -------------------------- | ---------------------------- |
| 1 epoch training | Lower empathy nuance       | 3 epochs → 65–75% score      |
| 39% dataset used | Reduced edge-case coverage | Train on full dataset        |
| 512 token limit  | Truncates long contexts    | Increase to 768+             |
| Single-turn only | No conversation memory     | Implement multi-turn support |

---

## 🧠 Technical Challenges & Solutions

### Challenge 1: Platform Limitations

**Problem:** Kaggle (14 hrs/week), Colab (3 hrs/day) insufficient
**Solution:** Local training with aggressive memory optimizations

### Challenge 2: Memory Constraints

**Problem:** 8B model typically needs ~32GB, GPU has 12GB
**Solution:** 4-bit QLoRA + gradient checkpointing + reduced seq length

### Challenge 3: Windows Compatibility

**Problem:** Multiprocessing pickle errors
**Solution:** Single-threaded processing and safer dataloader configs

### Challenge 4: Training Time

**Problem:** Full training = 30+ hours
**Solution:** Strategic dataset sampling (15k/38k) for 6-hour POC

---

## 🛡️ Safety & Disclaimer

This project is intended for **supportive conversation** and **emotional assistance**, not clinical diagnosis.
It must not provide diagnosis, treatment, medication guidance, or legal advice.
If a user expresses self-harm risk or imminent danger, the assistant should encourage contacting local emergency services and trusted people.

---

## 📚 References

* Hu et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models*
* Dettmers et al. (2023) — *QLoRA: Efficient Finetuning of Quantized LLMs*
* Meta AI (2024) — *Meta LLaMA 3.1 Model Card*

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/improvement`
3. Commit changes: `git commit -am "Add improvement"`
4. Push to branch: `git push origin feature/improvement`
5. Open a Pull Request

```

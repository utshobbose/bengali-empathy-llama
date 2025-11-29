Bengali Empathy Fine-Tuned LLaMA 3.1-8B

Fine-tuning Meta's LLaMA 3.1-8B-Instruct for Bengali empathetic conversations using QLoRA on consumer hardware

Show Image
Show Image
Show Image
Show Image

📋 Project Overview
Objective: Create an AI counselor chatbot that provides compassionate, culturally appropriate responses in Bengali for mental health support.

Base Model: meta-llama/Llama-3.1-8B-Instruct (8B parameters)
Dataset: Bengali Empathetic Conversations (~38,000 dialogue pairs)
Method: QLoRA (4-bit Quantized Low-Rank Adaptation)
Training Environment: Local Windows PC with RTX 3060 12GB
Training Time: ~6 hours for 1 epoch
Result: 40% empathy score (proof-of-concept under resource constraints)


🎯 Key Features
✅ Runs on Consumer Hardware - RTX 3060 12GB VRAM (student-accessible)
✅ Memory Efficient - 4-bit quantization (32GB → 8GB)
✅ Windows Compatible - Handles multiprocessing limitations
✅ Fully Reproducible - All experiments logged in JSONL format
✅ Production-Ready Code - Modular, documented, extensible

🚀 Quick Start
Prerequisites
bash# Hardware Requirements
- GPU: RTX 3060 (12GB) or equivalent (3060 Ti, 4060, 3080, 4070)
- RAM: 16GB minimum (32GB recommended)
- Storage: 50GB free space
- OS: Windows 11 / Linux / WSL2
Installation
bash# Clone repository
git clone https://github.com/yourusername/bengali-empathy-llama.git
cd bengali-empathy-llama

# Create virtual environment
python -m venv .venv310
.venv310\Scripts\activate  # Windows
# source .venv310/bin/activate  # Linux

# Install dependencies
pip install -r requirements.txt
Setup HuggingFace Token
bash# Windows Command Prompt
set HF_TOKEN=your_token_here

# Linux/Mac
export HF_TOKEN=your_token_here

💻 Usage
1. Training
bashpython bengali_empathy_finetuner.py
Training Configuration:

Epochs: 1 (6 hours)
Batch Size: 2 × 16 accumulation = 32 effective
Learning Rate: 3e-4
LoRA: r=16, alpha=32, dropout=0.05
Sequence Length: 512 tokens
Dataset: 15,000 / 38,000 samples (39%)

2. Interactive Testing
bash# Chat mode
python test_model.py

# Batch testing
python test_model.py batch

# Custom output file
python test_model.py batch results.json
3. Evaluation
bashpython human_evaluation.py
Evaluation Metrics:

Perplexity
BLEU Score
ROUGE Score (ROUGE-1, ROUGE-2, ROUGE-L)
Custom Empathy Scoring (4 criteria)


📊 Results
Performance Metrics
MetricScoreContextEmpathy Score40%Proof-of-concept (1 epoch, 39% dataset)Expected (Full)65-75%3 epochs, complete datasetTraining Time6 hoursRTX 3060 12GBMemory Usage~8GB4-bit quantization
Strengths
✅ All responses in fluent Bengali
✅ Contextually appropriate and safe
✅ No medical/legal advice (safe guardrails)
✅ Appropriate response length (10-100 words)
Sample Output
Input: আমার চাকরি চলে গেছে এবং আমি হতাশ। (I lost my job and I'm depressed.)
Output: আমি খুব দুঃখিত, আমি আশা করি আপনি এটা সম্পর্কে চিন্তা না করার জন্য একটি ভাল সময় পাবেন... (I'm very sorry, I hope you find a good time to not worry about this...)

🛠️ Technical Implementation
Memory Optimizations

4-bit QLoRA Quantization

Memory: 32GB → 8GB (75% reduction)
Quality: NF4 with double quantization (minimal loss)


Gradient Checkpointing

Memory savings: 40%
Speed trade-off: 30% slower training


Reduced Sequence Length

512 tokens (vs 1024 standard)
Sufficient for empathetic conversations (~200 tokens avg)
Speed improvement: 30% faster



Windows Compatibility Fixes
python# Disabled multiprocessing (pickle errors)
num_workers=0
pin_memory=False

# Single-threaded tokenization
datasets.map(..., num_proc=1)
```

### Dataset Strategy

- **Raw:** 76,466 dialogue turns
- **Cleaned:** 76,338 valid records
- **Training:** 15,000 samples (39% of dataset)
- **Split:** 85% train / 10% validation / 5% test

---

## 📁 Repository Structure
```
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

⚙️ Configuration
Hyperparameters (Optimized for RTX 3060)
pythonconfig = {
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

### System Prompt
```
আপনি একজন সহানুভূতিশীল বাংলা কাউন্সেলর। 
আপনি খুব ধীরে, নম্রভাবে এবং সম্মানজনক ভঙ্গিতে উত্তর দেবেন। 
ব্যক্তির অনুভূতিকে স্বীকার করবেন, আশ্বাস দেবেন এবং প্রয়োজন হলে 
পেশাদার সাহায্য নেওয়ার পরামর্শ দেবেন, কিন্তু কোন চিকিৎসা বা আইনি পরামর্শ দেবেন না।

🔮 Future Improvements
Priority 1 (Immediate)

 Train on full 38k dataset with 3 epochs (~30 hours)
 Increase LoRA rank to 32 for better capacity
 Implement early stopping based on validation loss

Priority 2 (Short-term)

 Data augmentation with empathy-specific examples
 RLHF (Reinforcement Learning from Human Feedback)
 Expand sequence length to 768 tokens

Priority 3 (Long-term)

 Cloud deployment for 24/7 availability
 Web interface for user testing
 Multi-turn conversation support
 Real-time human feedback collection


🚧 Known Limitations
LimitationImpactMitigation1 epoch trainingLower empathy nuance3 epochs → 65-75% score39% datasetReduced edge case coverageFull dataset training512 token limitMay truncate long contextsIncrease to 768 tokensSingle-turn onlyNo conversation memoryImplement multi-turn support

📚 Technical Challenges & Solutions
Challenge 1: Platform Limitations
Problem: Kaggle (14 hrs/week), Colab (3 hrs/day) insufficient
Solution: Local training with aggressive memory optimizations
Challenge 2: Memory Constraints
Problem: 8B model requires 32GB, only have 12GB
Solution: 4-bit QLoRA + gradient checkpointing + reduced seq length
Challenge 3: Windows Compatibility
Problem: Multiprocessing pickle errors
Solution: Single-threaded processing with custom dataloader config
Challenge 4: Training Time
Problem: Full training = 30+ hours
Solution: Strategic dataset sampling (15k/38k) for 6-hour training

📖 References

LoRA: Hu et al. (2021) - LoRA: Low-Rank Adaptation of Large Language Models
QLoRA: Dettmers et al. (2023) - QLoRA: Efficient Finetuning of Quantized LLMs
LLaMA 3.1: Meta AI (2024) - Meta LLaMA 3.1 Model Card


🤝 Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch (git checkout -b feature/improvement)
Commit changes (git commit -am 'Add improvement')
Push to branch (git push origin feature/improvement)
Open a Pull Request

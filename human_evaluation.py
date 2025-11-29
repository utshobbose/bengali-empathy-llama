# =====================================================================
# HUMAN EVALUATION SCRIPT - Bengali Empathy Model
# Evaluate model responses with human scoring and automatic metrics
# =====================================================================

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime
import pandas as pd

try:
    import evaluate
    from rouge_score import rouge_scorer
    HAVE_METRICS = True
except ImportError:
    HAVE_METRICS = False
    print("⚠ Warning: evaluate or rouge_score not installed. Some metrics will be skipped.")

HF_TOKEN = os.environ.get("HF_TOKEN", None)


class BengaliEmpathyEvaluator:
    def __init__(self, base_model_id: str, adapter_path: str):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            token=HF_TOKEN,
            use_fast=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            token=HF_TOKEN,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        print("Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        
        self.system_prompt = (
            "আপনি একজন সহানুভূতিশীল বাংলা কাউন্সেলর। "
            "আপনি খুব ধীরে, নম্রভাবে এবং সম্মানজনক ভঙ্গিতে উত্তর দেবেন। "
            "ব্যক্তির অনুভূতিকে স্বীকার করবেন, আশ্বাস দেবেন এবং প্রয়োজন হলে "
            "পেশাদার সাহায্য নেওয়ার পরামর্শ দেবেন, কিন্তু কোন চিকিৎসা বা আইনি পরামর্শ দেবেন না।"
        )
        
        if HAVE_METRICS:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        
        print("✓ Model loaded successfully!")
    
    def generate_response(self, user_input: str, max_new_tokens: int = 150) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        chat_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(chat_str, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> dict:
        """Calculate ROUGE scores between generated and reference text"""
        if not HAVE_METRICS:
            return {}
        
        scores = self.rouge_scorer.score(reference, generated)
        return {
            "rouge1_f": scores['rouge1'].fmeasure,
            "rouge2_f": scores['rouge2'].fmeasure,
            "rougeL_f": scores['rougeL'].fmeasure,
        }
    
    def evaluate_empathy_criteria(self, response: str) -> dict:
        """
        Simple heuristic-based evaluation of empathy criteria
        This is a basic check - human evaluation is better
        """
        criteria = {
            "has_empathy_words": 0,
            "has_acknowledgment": 0,
            "has_support": 0,
            "appropriate_length": 0,
        }
        
        response_lower = response.lower()
        
        # Bengali empathy keywords
        empathy_words = ["বুঝি", "অনুভব", "কষ্ট", "দুঃখ", "সহানুভূতি", "চিন্তা"]
        acknowledgment_words = ["হ্যাঁ", "ঠিক", "সত্যি", "অবশ্যই"]
        support_words = ["সাহায্য", "পাশে", "সমর্থন", "পরামর্শ", "সহায়তা"]
        
        # Check empathy words
        if any(word in response_lower for word in empathy_words):
            criteria["has_empathy_words"] = 1
        
        # Check acknowledgment
        if any(word in response_lower for word in acknowledgment_words):
            criteria["has_acknowledgment"] = 1
        
        # Check support words
        if any(word in response_lower for word in support_words):
            criteria["has_support"] = 1
        
        # Check length (should be substantial but not too long)
        word_count = len(response.split())
        if 10 <= word_count <= 100:
            criteria["appropriate_length"] = 1
        
        criteria["total_score"] = sum(criteria.values())
        criteria["percentage"] = (criteria["total_score"] / 4.0) * 100
        
        return criteria
    
    def evaluate_test_set(self, test_cases: list, reference_responses: list = None) -> dict:
        """
        Evaluate model on a test set
        
        Args:
            test_cases: List of input questions
            reference_responses: Optional list of reference/ground truth responses
        
        Returns:
            Dictionary with evaluation results
        """
        results = []
        
        print("\n" + "="*70)
        print("EVALUATING TEST SET")
        print("="*70)
        
        for i, test_input in enumerate(test_cases):
            print(f"\nTest {i+1}/{len(test_cases)}")
            print(f"Input: {test_input[:60]}...")
            
            # Generate response
            generated = self.generate_response(test_input)
            print(f"Generated: {generated[:80]}...")
            
            # Evaluate
            result = {
                "test_id": i + 1,
                "input": test_input,
                "generated_response": generated,
            }
            
            # Calculate empathy criteria
            empathy_score = self.evaluate_empathy_criteria(generated)
            result.update(empathy_score)
            
            # Calculate ROUGE if reference is provided
            if reference_responses and i < len(reference_responses):
                reference = reference_responses[i]
                result["reference_response"] = reference
                rouge_scores = self.calculate_rouge_scores(generated, reference)
                result.update(rouge_scores)
            
            results.append(result)
        
        # Calculate aggregate statistics
        avg_empathy = sum(r["percentage"] for r in results) / len(results)
        
        summary = {
            "total_tests": len(results),
            "average_empathy_score": avg_empathy,
            "timestamp": datetime.now().isoformat(),
        }
        
        if reference_responses and HAVE_METRICS:
            avg_rouge1 = sum(r.get("rouge1_f", 0) for r in results) / len(results)
            avg_rouge2 = sum(r.get("rouge2_f", 0) for r in results) / len(results)
            avg_rougeL = sum(r.get("rougeL_f", 0) for r in results) / len(results)
            summary.update({
                "average_rouge1": avg_rouge1,
                "average_rouge2": avg_rouge2,
                "average_rougeL": avg_rougeL,
            })
        
        return {
            "summary": summary,
            "detailed_results": results,
        }
    
    def save_evaluation_results(self, results: dict, output_path: str):
        """Save evaluation results to JSON file"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    def print_summary(self, results: dict):
        """Print a summary of evaluation results"""
        summary = results["summary"]
        
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Average Empathy Score: {summary['average_empathy_score']:.2f}%")
        
        if "average_rouge1" in summary:
            print(f"Average ROUGE-1: {summary['average_rouge1']:.4f}")
            print(f"Average ROUGE-2: {summary['average_rouge2']:.4f}")
            print(f"Average ROUGE-L: {summary['average_rougeL']:.4f}")
        
        print("="*70)
        
        # Print detailed breakdown
        print("\nDETAILED BREAKDOWN:")
        for r in results["detailed_results"]:
            print(f"\nTest {r['test_id']}:")
            print(f"  Empathy Score: {r['percentage']:.1f}%")
            print(f"  - Has empathy words: {'✓' if r['has_empathy_words'] else '✗'}")
            print(f"  - Has acknowledgment: {'✓' if r['has_acknowledgment'] else '✗'}")
            print(f"  - Has support: {'✓' if r['has_support'] else '✗'}")
            print(f"  - Appropriate length: {'✓' if r['appropriate_length'] else '✗'}")


# =====================================================================
# MAIN EVALUATION
# =====================================================================

if __name__ == "__main__":
    BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    ADAPTER_PATH = r"D:\bengali-empathy-llama\outputs\llama31_bengali_empathy"
    OUTPUT_PATH = r"D:\bengali-empathy-llama\outputs\evaluation_results.json"
    
    # Initialize evaluator
    evaluator = BengaliEmpathyEvaluator(BASE_MODEL, ADAPTER_PATH)
    
    # Test cases
    test_questions = [
        "আমি খুব দুঃখিত এবং একাকী অনুভব করছি। কী করব?",
        "আমার চাকরি চলে গেছে এবং আমি হতাশ।",
        "পরীক্ষায় খারাপ ফলাফল হয়েছে, আমি কি ব্যর্থ?",
        "আমার পরিবারের সাথে ঝগড়া হয়েছে।",
        "আমি অনেক চাপে আছি এবং ঘুমাতে পারছি না।",
    ]
    
    # Optional: Provide reference responses for ROUGE calculation
    # If you don't have references, set this to None
    reference_responses = None  # or provide a list of expected responses
    
    # Run evaluation
    results = evaluator.evaluate_test_set(test_questions, reference_responses)
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    evaluator.save_evaluation_results(results, OUTPUT_PATH)
    
    print("\n✓ Evaluation complete!")

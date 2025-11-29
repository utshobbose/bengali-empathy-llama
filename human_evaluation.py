# =====================================================================
# HUMAN EVALUATION & SAMPLE GENERATION FOR BENGALI EMPATHY MODEL
# Generates test responses and enables human evaluation
# =====================================================================

import os
import json
import time
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import pandas as pd
from tqdm import tqdm

class BengaliEmpathyEvaluator:
    """
    Evaluator for Bengali Empathy Model
    - Generates responses for test prompts
    - Facilitates human evaluation
    - Produces evaluation reports
    """
    
    def __init__(self, 
                 base_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
                 adapter_path: str = r"E:\bengali-empathy-llama\outputs\llama31_bengali_empathy",
                 output_dir: str = r"E:\bengali-empathy-llama\evaluation",
                 use_4bit: bool = True):  # Use 4-bit for stability
        
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.output_dir = output_dir
        self.use_4bit = use_4bit
        os.makedirs(output_dir, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        
        self.system_prompt = (
            "আপনি একজন সহানুভূতিশীল বাংলা কাউন্সেলর। "
            "আপনি খুব ধীরে, নম্রভাবে এবং সম্মানজনক ভঙ্গিতে উত্তর দেবেন। "
            "ব্যক্তির অনুভূতিকে স্বীকার করবেন, আশ্বাস দেবেন এবং প্রয়োজন হলে "
            "পেশাদার সাহায্য নেওয়ার পরামর্শ দেবেন, কিন্তু কোন চিকিৎসা বা আইনি পরামর্শ দেবেন না।"
        )
        
    def load_model(self):
        """Load fine-tuned model with LoRA adapters"""
        print(f"Loading tokenizer from {self.adapter_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
        
        if self.use_4bit:
            # Use 4-bit loading (same as training) - most stable
            print(f"Loading base model {self.base_model_id} with 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )
            
            print(f"Loading LoRA adapters from {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        else:
            # Load in float16 (requires more memory)
            print(f"Loading base model {self.base_model_id} in float16...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            
            print(f"Loading LoRA adapters from {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            
            # Move to GPU after loading
            if torch.cuda.is_available():
                print("Moving model to GPU...")
                self.model = self.model.to('cuda')
        
        self.model.eval()
        print("✓ Model loaded successfully")
        
    def generate_response(self, 
                         user_input: str, 
                         max_new_tokens: int = 150,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> str:
        """Generate empathetic response for user input"""
        
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
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        gen_only = gen_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(gen_only, skip_special_tokens=True).strip()
        return response
    
    def get_test_prompts(self) -> List[Dict[str, str]]:
        """Get diverse test prompts covering various emotional scenarios"""
        return [
            {
                "category": "Depression",
                "prompt": "আমি খুব একা অনুভব করছি। কেউ আমার কথা বোঝে না।"
            },
            {
                "category": "Anxiety",
                "prompt": "পরীক্ষা নিয়ে আমি খুব চিন্তিত। মনে হয় আমি পারব না।"
            },
            {
                "category": "Relationship",
                "prompt": "আমার সাথে আমার সেরা বন্ধুর ঝগড়া হয়েছে। আমি কী করব?"
            },
            {
                "category": "Family",
                "prompt": "আমার বাবা-মা সবসময় আমাকে অন্যদের সাথে তুলনা করে। এটা আমাকে খুব কষ্ট দেয়।"
            },
            {
                "category": "Self-esteem",
                "prompt": "আমি মনে করি আমি যথেষ্ট ভালো নই। সবাই আমার থেকে ভালো।"
            },
            {
                "category": "Loss",
                "prompt": "আমার দাদু গত সপ্তাহে মারা গেছেন। আমি খুব কষ্ট পাচ্ছি।"
            },
            {
                "category": "Career",
                "prompt": "আমি চাকরি হারিয়েছি এবং ভবিষ্যৎ নিয়ে দুশ্চিন্তায় আছি।"
            },
            {
                "category": "Social",
                "prompt": "আমার নতুন কোন বন্ধু নেই। স্কুলে আমি সবসময় একা থাকি।"
            },
            {
                "category": "Stress",
                "prompt": "কাজের চাপ এত বেশি যে আমি ঘুমাতে পারছি না।"
            },
            {
                "category": "Health",
                "prompt": "আমার স্বাস্থ্য নিয়ে চিন্তা হচ্ছে কিন্তু ডাক্তারের কাছে যেতে ভয় পাচ্ছি।"
            }
        ]
    
    def generate_sample_responses(self, save: bool = True) -> List[Dict[str, Any]]:
        """Generate responses for all test prompts"""
        if self.model is None:
            self.load_model()
        
        test_prompts = self.get_test_prompts()
        results = []
        
        print(f"\nGenerating responses for {len(test_prompts)} test prompts...\n")
        
        for item in tqdm(test_prompts, desc="Generating"):
            response = self.generate_response(item["prompt"])
            
            result = {
                "category": item["category"],
                "user_input": item["prompt"],
                "model_response": response,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            results.append(result)
            
            # Print for review
            print(f"\n{'='*70}")
            print(f"Category: {item['category']}")
            print(f"User: {item['prompt']}")
            print(f"Assistant: {response}")
        
        if save:
            output_file = os.path.join(self.output_dir, "sample_responses.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n✓ Saved sample responses to: {output_file}")
        
        return results
    
    def create_human_evaluation_template(self, responses: List[Dict[str, Any]]):
        """Create Excel template for human evaluation"""
        
        eval_data = []
        for idx, resp in enumerate(responses, 1):
            eval_data.append({
                "ID": idx,
                "Category": resp["category"],
                "User Input": resp["user_input"],
                "Model Response": resp["model_response"],
                "Empathy Score (1-5)": "",
                "Appropriateness Score (1-5)": "",
                "Helpfulness Score (1-5)": "",
                "Language Quality (1-5)": "",
                "Overall Score (1-5)": "",
                "Comments": ""
            })
        
        df = pd.DataFrame(eval_data)
        output_file = os.path.join(self.output_dir, "human_evaluation_template.xlsx")
        df.to_excel(output_file, index=False)
        print(f"\n✓ Created human evaluation template: {output_file}")
        print("\nEvaluation Criteria:")
        print("  1. Empathy Score: How empathetic is the response?")
        print("  2. Appropriateness: Is the response appropriate for the situation?")
        print("  3. Helpfulness: Does the response provide helpful guidance?")
        print("  4. Language Quality: Is the Bengali language natural and correct?")
        print("  5. Overall Score: Overall quality of the response")
        print("\nScoring: 1 (Poor) to 5 (Excellent)")
        
        return output_file
    
    def analyze_human_evaluation(self, eval_file: str):
        """Analyze completed human evaluation results"""
        
        if not os.path.exists(eval_file):
            print(f"Evaluation file not found: {eval_file}")
            return
        
        df = pd.read_excel(eval_file)
        
        # Check if evaluation is complete
        score_columns = [
            "Empathy Score (1-5)",
            "Appropriateness Score (1-5)",
            "Helpfulness Score (1-5)",
            "Language Quality (1-5)",
            "Overall Score (1-5)"
        ]
        
        if df[score_columns].isna().any().any():
            print("⚠ Warning: Some scores are missing in the evaluation file")
        
        # Calculate statistics
        results = {
            "total_samples": len(df),
            "category_breakdown": df["Category"].value_counts().to_dict(),
            "average_scores": {}
        }
        
        for col in score_columns:
            metric_name = col.replace(" (1-5)", "")
            try:
                avg_score = df[col].astype(float).mean()
                std_score = df[col].astype(float).std()
                results["average_scores"][metric_name] = {
                    "mean": round(avg_score, 2),
                    "std": round(std_score, 2)
                }
            except:
                results["average_scores"][metric_name] = "N/A"
        
        # Category-wise analysis
        category_scores = {}
        for category in df["Category"].unique():
            category_df = df[df["Category"] == category]
            category_scores[category] = {
                "count": len(category_df),
                "avg_overall": round(category_df["Overall Score (1-5)"].astype(float).mean(), 2)
            }
        results["category_scores"] = category_scores
        
        # Save analysis
        analysis_file = os.path.join(self.output_dir, "human_evaluation_analysis.json")
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*70)
        print("HUMAN EVALUATION ANALYSIS")
        print("="*70)
        print(f"\nTotal Samples Evaluated: {results['total_samples']}")
        print("\nAverage Scores:")
        for metric, scores in results["average_scores"].items():
            if scores != "N/A":
                print(f"  {metric}: {scores['mean']} ± {scores['std']}")
            else:
                print(f"  {metric}: {scores}")
        
        print("\nCategory-wise Performance:")
        for category, data in category_scores.items():
            print(f"  {category}: {data['avg_overall']}/5 (n={data['count']})")
        
        print(f"\n✓ Detailed analysis saved to: {analysis_file}")
        
        return results
    
    def interactive_test(self):
        """Interactive testing mode"""
        if self.model is None:
            self.load_model()
        
        print("\n" + "="*70)
        print("INTERACTIVE TESTING MODE")
        print("="*70)
        print("Enter Bengali text to get empathetic responses.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            user_input = input("User: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = self.generate_response(user_input)
            print(f"Assistant: {response}\n")


def main():
    """Main execution"""
    print("="*70)
    print("BENGALI EMPATHY MODEL - EVALUATION SUITE")
    print("="*70)
    
    evaluator = BengaliEmpathyEvaluator()
    
    print("\nOptions:")
    print("1. Generate sample responses for test prompts")
    print("2. Create human evaluation template")
    print("3. Analyze human evaluation results")
    print("4. Interactive testing mode")
    print("5. Do all (1 + 2)")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        evaluator.load_model()
        responses = evaluator.generate_sample_responses()
        print("\n✓ Sample responses generated")
        
    elif choice == "2":
        # Load existing responses or generate new ones
        responses_file = os.path.join(evaluator.output_dir, "sample_responses.json")
        if os.path.exists(responses_file):
            with open(responses_file, "r", encoding="utf-8") as f:
                responses = json.load(f)
        else:
            evaluator.load_model()
            responses = evaluator.generate_sample_responses()
        
        evaluator.create_human_evaluation_template(responses)
        
    elif choice == "3":
        eval_file = os.path.join(evaluator.output_dir, "human_evaluation_template.xlsx")
        evaluator.analyze_human_evaluation(eval_file)
        
    elif choice == "4":
        evaluator.interactive_test()
        
    elif choice == "5":
        evaluator.load_model()
        responses = evaluator.generate_sample_responses()
        evaluator.create_human_evaluation_template(responses)
        print("\n✓ Ready for human evaluation!")
        
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
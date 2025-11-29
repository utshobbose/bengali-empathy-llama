# =====================================================================
# BENGALI EMPATHY MODEL TESTER - WITH CLI SUPPORT
# Test your fine-tuned model with real Bengali queries
# =====================================================================

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json

HF_TOKEN = os.environ.get("HF_TOKEN", None)

class BengaliEmpathyTester:
    def __init__(self, base_model_id: str, adapter_path: str):
        """
        Load the fine-tuned model for testing
        
        Args:
            base_model_id: Base Llama model (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            adapter_path: Path to your trained LoRA adapters
        """
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
        
        print("Loading base model with 4-bit quantization...")
        # Use the same quantization config as training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            token=HF_TOKEN,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        print("Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
        )
        self.model.eval()
        
        self.system_prompt = (
            "আপনি একজন সহানুভূতিশীল বাংলা কাউন্সেলর। "
            "আপনি খুব ধীরে, নম্রভাবে এবং সম্মানজনক ভঙ্গিতে উত্তর দেবেন। "
            "ব্যক্তির অনুভূতিকে স্বীকার করবেন, আশ্বাস দেবেন এবং প্রয়োজন হলে "
            "পেশাদার সাহায্য নেওয়ার পরামর্শ দেবেন, কিন্তু কোন চিকিৎসা বা আইনি পরামর্শ দেবেন না।"
        )
        
        print("✓ Model loaded successfully!")
        print(f"Device: {next(self.model.parameters()).device}")
    
    def generate_response(
        self, 
        user_input: str, 
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a response for a user input
        
        Args:
            user_input: The user's question/message in Bengali
            max_new_tokens: Maximum length of response
            temperature: Sampling temperature (lower = more focused)
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated response text
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        chat_str = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(chat_str, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Extract only the generated part (not the prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def interactive_test(self):
        """
        Interactive testing mode - chat with the model
        """
        print("\n" + "="*70)
        print("INTERACTIVE TESTING MODE")
        print("Type your Bengali question and press Enter")
        print("Type 'quit' or 'exit' to stop")
        print("="*70 + "\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\nModel: ", end="", flush=True)
            response = self.generate_response(user_input)
            print(response)
            print()
    
    def batch_test(self, test_cases: list) -> list:
        """
        Test multiple cases at once
        
        Args:
            test_cases: List of test questions
        
        Returns:
            List of results with inputs and outputs
        """
        results = []
        
        print("\n" + "="*70)
        print("BATCH TESTING")
        print("="*70 + "\n")
        
        for i, test_input in enumerate(test_cases, 1):
            print(f"Test {i}/{len(test_cases)}: {test_input[:50]}...")
            response = self.generate_response(test_input)
            
            result = {
                "input": test_input,
                "output": response,
            }
            results.append(result)
            
            print(f"Response: {response[:100]}...\n")
        
        return results
    
    def save_test_results(self, results: list, output_file: str):
        """
        Save test results to a JSON file
        
        Args:
            results: List of test results
            output_file: Path to save results
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ Results saved to: {output_file}")


# =====================================================================
# EXAMPLE USAGE WITH CLI SUPPORT
# =====================================================================

def print_usage():
    """Print usage instructions"""
    print("\nUsage:")
    print("  python test_model.py                    # Interactive mode")
    print("  python test_model.py interactive        # Interactive mode")
    print("  python test_model.py batch              # Batch testing mode")
    print("  python test_model.py batch <output.json> # Batch with custom output file")
    print()

if __name__ == "__main__":
    # Configure paths
    BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    ADAPTER_PATH = r"E:\bengali-empathy-llama\outputs\llama31_bengali_empathy"
    
    # Default test questions for batch mode
    DEFAULT_TEST_QUESTIONS = [
        "আমি খুব দুঃখিত এবং একাকী অনুভব করছি। কী করব?",
        "আমার চাকরি চলে গেছে এবং আমি হতাশ।",
        "পরীক্ষায় খারাপ ফলাফল হয়েছে, আমি কি ব্যর্থ?",
        "আমার পরিবারের সাথে ঝগড়া হয়েছে।",
        "আমি অনেক চাপে আছি এবং ঘুমাতে পারছি না।",
    ]
    
    # Parse command line arguments
    mode = "interactive"  # default
    output_file = r"E:\bengali-empathy-llama\outputs\test_results.json"
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode not in ["interactive", "batch", "help", "-h", "--help"]:
            print(f"Error: Unknown mode '{sys.argv[1]}'")
            print_usage()
            sys.exit(1)
        
        if mode in ["help", "-h", "--help"]:
            print_usage()
            sys.exit(0)
        
        # Custom output file path
        if len(sys.argv) > 2 and mode == "batch":
            output_file = sys.argv[2]
    
    # Initialize tester
    print("Initializing Bengali Empathy Tester...")
    tester = BengaliEmpathyTester(BASE_MODEL, ADAPTER_PATH)
    
    # Run selected mode
    if mode == "interactive":
        print("\n[Interactive Mode] Starting interactive test...")
        print("You can now chat with your model!")
        tester.interactive_test()
    
    elif mode == "batch":
        print("\n[Batch Mode] Running batch tests...")
        results = tester.batch_test(DEFAULT_TEST_QUESTIONS)
        
        # Save results
        tester.save_test_results(results, output_file)
        
        print("\n" + "="*70)
        print("BATCH TESTING COMPLETE")
        print(f"Results saved to: {output_file}")
        print("="*70)
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import json
from pathlib import Path
import os
import sys

# Try to import protobuf
try:
    import google.protobuf
except ImportError:
    print("Installing protobuf...")
    os.system(f"{sys.executable} -m pip install protobuf")
    import google.protobuf

# Configuration
MODEL_PATH = "./hindi_lora/hindi_lora_adapter"  # Path to your fine-tuned model
TEST_SAMPLES = [
    "Hello, how are you?",
    "What is your name?",
    "I love programming and artificial intelligence.",
    "The weather is nice today.",
    "Can you help me with this problem?",
    "Thank you very much for your assistance.",
    "Where is the nearest restaurant?",
    "I would like to book a table for two.",
    "What time does the train leave?",
    "Could you please speak more slowly?"
]

def load_model_and_tokenizer(model_path):
    """Load the fine-tuned model and tokenizer."""
    print("Loading model and tokenizer...")
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer with specific parameters for IndicTrans
    tokenizer = AutoTokenizer.from_pretrained(
        "ai4bharat/indictrans2-en-indic-dist-200M",  # Use base model for tokenizer
        trust_remote_code=True,
        use_fast=False  # Important for IndicTrans tokenizer
    )
    
    return model, tokenizer

def translate_text(model, tokenizer, text, max_length=128):
    """Translate English text to Hindi."""
    try:
        # Prepare input with language tags
        inputs = tokenizer(
            f"eng_Latn hin_Deva {text}",
            return_tensors="pt",
            max_length=384,
            truncation=True,
            padding=True
        ).to(model.device)
        
        # Generate translation
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        # Decode and clean up the output
        translation = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return translation
    except Exception as e:
        print(f"Error translating text: {e}")
        return None

def run_test_cases(model, tokenizer, test_cases):
    """Run translation on test cases and print results."""
    print("\n" + "="*80)
    print("RUNNING TRANSLATION TESTS")
    print("="*80)
    
    results = []
    for i, text in enumerate(tqdm(test_cases, desc="Translating")):
        translation = translate_text(model, tokenizer, text)
        results.append({
            "input": text,
            "translation": translation
        })
        
        # Print each result
        print(f"\nTest Case {i+1}:")
        print(f"  English: {text}")
        print(f"  Hindi:   {translation}")
    
    return results

def save_results(results, filename="translation_results.json"):
    """Save translation results to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {filename}")

def main():
    # Check if model exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first or update the MODEL_PATH.")
        return
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    # Run test cases
    results = run_test_cases(model, tokenizer, TEST_SAMPLES)
    
    # Save results
    save_results(results)
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()

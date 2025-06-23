from datasets import load_from_disk, Dataset
from pathlib import Path
from tqdm.auto import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
import os

def load_model_and_tokenizer():
    """Load the model and tokenizer with 8-bit quantization."""
    print("Loading model and tokenizer...")
    
    # Configure 8-bit quantization
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load model with 8-bit quantization
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "ai4bharat/indictrans2-en-indic-dist-200M",
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "ai4bharat/indictrans2-en-indic-dist-200M",
        trust_remote_code=True
    )
    
    # Debug: Print tokenizer details
    print("Special tokens map:", tokenizer.special_tokens_map)
    print("Vocab size:", tokenizer.vocab_size)
    print("Language-related attributes:", [attr for attr in dir(tokenizer) if "lang" in attr.lower()])
    for token in ["eng_Latn", "hin_Deva", "en_XX", "hi_IN", "__en__", "__hi__", "eng", "hin"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"Token '{token}' ID:", token_id, "(Valid)" if token_id != tokenizer.unk_token_id else "(Unknown)")
    
    return model, tokenizer

def generate_synthetic_data(input_path, output_path, num_samples=1000):
    """
    Generate synthetic Hindi translations for English sentences.
    
    Args:
        input_path: Path to the training data
        output_path: Path to save the synthetic data
        num_samples: Number of samples to generate
    """
    print(f"Loading dataset from {input_path}")
    try:
        train_data = load_from_disk(input_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Select a subset for generation
    if num_samples > 0:
        train_data = train_data.select(range(min(num_samples, len(train_data))))
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    def generate_translation(example):
        """Generate a single translation."""
        # Extract English text (remove the prompt)
        en_text = example["prompt"].replace("Translate from English to Hindi: ", "").strip()
        if not en_text:
            print(f"Warning: Empty input text for prompt: {example['prompt']}")
            return {
                "prompt": example["prompt"],
                "response": "",
                "source": "synthetic"
            }
        
        # Prepend source and target language tokens
        input_text = f"eng_Latn hin_Deva {en_text}"
        
        # Debug: Print input text
        print(f"Processing input: {input_text}")
        
        # Prepare input for tokenizer
        try:
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(model.device)
            
            # Debug: Print encoded inputs
            print(f"Encoded input IDs: {inputs['input_ids']}")
            print(f"Decoded input: {tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)}")
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=5,
                    early_stopping=True
                    # Removed forced_bos_token_id since target language is in input
                )
            
            # Decode and clean up
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error translating '{en_text}': {e}")
            translated = ""
        
        return {
            "prompt": example["prompt"],
            "response": translated,
            "source": "synthetic"
        }
    
    # Generate translations iteratively
    print(f"Generating {len(train_data)} synthetic translations...")
    synthetic_examples = []
    for example in tqdm(train_data, desc="Generating translations"):
        result = generate_translation(example)
        synthetic_examples.append(result)
    
    # Create synthetic dataset
    synthetic_data = Dataset.from_dict({
        "prompt": [ex["prompt"] for ex in synthetic_examples],
        "response": [ex["response"] for ex in synthetic_examples],
        "source": [ex["source"] for ex in synthetic_examples]
    })
    
    # Save the synthetic data
    print(f"Saving synthetic data to {output_path}")
    synthetic_data.save_to_disk(output_path)
    
    return synthetic_data

def combine_datasets(original_path, synthetic_path, output_path):
    """Combine original and synthetic datasets."""
    print("Combining datasets...")
    
    # Load datasets
    try:
        original_data = load_from_disk(original_path)
        synthetic_data = load_from_disk(synthetic_path)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None
    
    # Add source information
    original_data = original_data.map(lambda x: {"source": "original"})
    
    # Combine datasets
    combined_data = Dataset.from_dict({
        "prompt": original_data["prompt"] + synthetic_data["prompt"],
        "response": original_data["response"] + synthetic_data["response"],
        "source": original_data["source"] + synthetic_data["source"]
    })
    
    # Save combined dataset
    print(f"Saving combined dataset to {output_path}")
    combined_data.save_to_disk(output_path)
    
    return combined_data

if __name__ == "__main__":
    # Configuration
    TRAIN_PATH = "hindi_train_test/train"
    SYNTHETIC_PATH = "hindi_synthetic"
    COMBINED_PATH = "hindi_combined"
    NUM_SAMPLES = 1000  # Number of samples to generate
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(
        input_path=TRAIN_PATH,
        output_path=SYNTHETIC_PATH,
        num_samples=NUM_SAMPLES
    )
    
    # Combine with original data
    if synthetic_data is not None:
        combined_data = combine_datasets(
            original_path=TRAIN_PATH,
            synthetic_path=SYNTHETIC_PATH,
            output_path=COMBINED_PATH
        )
        
        if combined_data is not None:
            print("\nProcessing complete!")
            print(f"Original data size: {len(load_from_disk(TRAIN_PATH))}")
            print(f"Synthetic data size: {len(synthetic_data)}")
            print(f"Combined data size: {len(combined_data)}")
            print(f"\nSaved to:")
            print(f"- Synthetic data: {SYNTHETIC_PATH}")
            print(f"- Combined data: {COMBINED_PATH}")
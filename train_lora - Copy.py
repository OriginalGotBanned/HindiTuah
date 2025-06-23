import torch
import os
import transformers
import peft
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

print(f"transformers version: {transformers.__version__}")
print(f"peft version: {peft.__version__}")

def load_model_and_tokenizer(model_name="ai4bharat/indictrans2-en-indic-dist-200M"):
    logger.info("Loading model and tokenizer...")
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        logger.info("Model and tokenizer loaded successfully")
        logger.info(f"Special tokens map: {tokenizer.special_tokens_map}")
        logger.info(f"Vocab size: {tokenizer.vocab_size}")
        for token in ["eng_Latn", "hin_Deva"]:
            token_id = tokenizer.convert_tokens_to_ids(token)
            logger.info(f"Token '{token}' ID: {token_id} {'(Valid)' if token_id != tokenizer.unk_token_id else '(Unknown)'}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model/tokenizer: {e}")
        raise

def preprocess_dataset(dataset, tokenizer, max_length=384):
    truncation_count = 0
    error_count = 0
    valid_examples = 0
    
    def format_example(example):
        nonlocal truncation_count, error_count, valid_examples
        try:
            # Extract and clean the English text
            en_text = example["prompt"].replace("Translate from English to Hindi: ", "").strip()
            if not en_text:
                raise ValueError("Empty English text after cleaning")
            # Prepare input with language tags
            input_text = f"eng_Latn hin_Deva {en_text}"
            target_text = example["response"].strip()
            if not target_text:
                raise ValueError("Empty target text")
            # Tokenize input
            inputs = tokenizer(
                input_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                return_length=True
            )
            # Tokenize target
            targets = tokenizer(
                target_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                return_length=True
            )
            # Check for truncation
            input_length = inputs["length"][0].item()
            target_length = targets["length"][0].item()
            if input_length >= max_length or target_length >= max_length:
                truncation_count += 1
                logger.warning(f"Example truncated - Input length: {input_length}, Target length: {target_length}, Input: {input_text[:50]}...")
            # Validate tokenized output
            if not (inputs["input_ids"].shape[1] > 0 and targets["input_ids"].shape[1] > 0):
                raise ValueError("Invalid tokenization output")
            valid_examples += 1
            return {
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "labels": targets["input_ids"].squeeze()
            }
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing example: {e}")
            logger.error(f"Input text: {input_text[:100]}...")
            logger.error(f"Target text: {target_text[:100]}...")
            return None

    logger.info("Preprocessing dataset...")
    # Map the dataset
    processed_dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        num_proc=1  # Single process to avoid multiprocessing issues
    )
    # Log dataset info before filtering
    logger.info(f"Dataset after mapping: {len(processed_dataset)} examples")
    logger.info(f"Dataset columns: {processed_dataset.column_names}")
    
    # Filter out None values
    original_size = len(processed_dataset)
    processed_dataset = processed_dataset.filter(
        lambda x: all(k in x and x[k] is not None for k in ["input_ids", "attention_mask", "labels"]),
        num_proc=1
    )
    filtered_count = original_size - len(processed_dataset)
    
    logger.info("Preprocessing complete!")
    logger.info(f"- Original examples: {original_size}")
    logger.info(f"- Filtered examples: {filtered_count}")
    logger.info(f"- Valid examples: {valid_examples}")
    logger.info(f"- Truncated examples: {truncation_count}")
    logger.info(f"- Error examples: {error_count}")
    logger.info(f"- Remaining examples: {len(processed_dataset)}")
    
    if len(processed_dataset) == 0:
        raise ValueError("No valid examples remaining after preprocessing!")
    
    # Save a sample of the processed dataset for debugging
    processed_dataset.select(range(min(5, len(processed_dataset)))).save_to_disk("debug_processed_dataset")
    logger.info("Saved sample of processed dataset to 'debug_processed_dataset'")
    
    return processed_dataset

def main():
    MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
    DATASET_PATH = "hindi_combined"
    OUTPUT_DIR = "./hindi_lora"
    MAX_STEPS = 200
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 8
    MAX_LENGTH = 384
    torch.cuda.empty_cache()
    logger.info(f"Loading dataset from {DATASET_PATH}")
    try:
        dataset = load_from_disk(DATASET_PATH)
        logger.info(f"Dataset loaded: {len(dataset)} examples")
        logger.info(f"Dataset columns: {dataset.column_names}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    dataset = preprocess_dataset(dataset, tokenizer, MAX_LENGTH)
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=32,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=2e-4,
        max_steps=MAX_STEPS,
        fp16=True,
        logging_steps=50,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=True,
        report_to="none"
    )
    logger.info("Initializing Trainer...")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    logger.info("Starting training...")
    trainer.train()
    logger.info("Saving LoRA adapter...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "hindi_lora_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "hindi_lora_adapter"))
    logger.info(f"Training complete! Adapter saved to {OUTPUT_DIR}/hindi_lora_adapter")

if __name__ == "__main__":
    main()
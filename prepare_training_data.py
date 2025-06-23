from datasets import load_from_disk, DatasetDict
from pathlib import Path
import os

def format_for_finetuning(subset_path, output_dir="hindi_formatted"):
    """
    Format the Hindi subset data for fine-tuning.
    
    Args:
        subset_path (str): Path to the saved subset
        output_dir (str): Directory to save the formatted data
    """
    print("Loading subset data...")
    subset = load_from_disk(subset_path)
    
    def format_example(example):
        """Format a single example for translation fine-tuning."""
        return {
            "prompt": f"Translate from English to Hindi: {example['src'].strip()}",
            "response": example['tgt'].strip()
        }
    
    print("Formatting data for fine-tuning...")
    formatted_data = subset.map(format_example, remove_columns=subset.column_names)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save formatted data
    print(f"Saving formatted data to {output_path}...")
    formatted_data.save_to_disk(str(output_path))
    
    return formatted_data

def split_train_test(formatted_data, output_dir="hindi_train_test", test_size=0.2):
    """
    Split the formatted data into training and test sets.
    
    Args:
        formatted_data: The formatted dataset
        output_dir (str): Directory to save the split datasets
        test_size (float): Proportion of data to use for testing (0.0 to 1.0)
    """
    print("\nSplitting data into train and test sets...")
    train_test = formatted_data.train_test_split(test_size=test_size, seed=42)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save train and test sets
    train_path = output_path / "train"
    test_path = output_path / "test"
    
    print(f"Saving training set ({len(train_test['train'])} examples)...")
    train_test["train"].save_to_disk(str(train_path))
    
    print(f"Saving test set ({len(train_test['test'])} examples)...")
    train_test["test"].save_to_disk(str(test_path))
    
    print(f"\nDataset split complete!")
    print(f"- Training set: {len(train_test['train'])} examples")
    print(f"- Test set: {len(train_test['test'])} examples")
    
    return train_test

if __name__ == "__main__":
    # Configuration
    SUBSET_PATH = "hindi_data/hindi_subset_5000"  # Path to your saved subset
    FORMATTED_DIR = "hindi_formatted"
    OUTPUT_DIR = "hindi_train_test"
    
    # Step 1: Format the data for fine-tuning
    formatted_data = format_for_finetuning(
        subset_path=SUBSET_PATH,
        output_dir=FORMATTED_DIR
    )
    
    # Step 2: Split into train and test sets
    train_test = split_train_test(
        formatted_data=formatted_data,
        output_dir=OUTPUT_DIR,
        test_size=0.2
    )
    
    print("\nProcessing complete!")
    print(f"- Formatted data saved to: {FORMATTED_DIR}")
    print(f"- Train/Test split saved to: {OUTPUT_DIR}")

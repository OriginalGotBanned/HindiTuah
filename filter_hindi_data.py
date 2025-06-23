from datasets import load_dataset
import time
import os
from pathlib import Path

def filter_hindi_data(save_subset=False, subset_size=5000, output_dir="hindi_data"):
    """
    Load and process Hindi data from BPCC dataset.
    
    Args:
        save_subset (bool): Whether to save a subset of the data
        subset_size (int): Number of examples to include in the subset
        output_dir (str): Directory to save the subset
    """
    print("Loading dataset...")
    start_time = time.time()
    
    try:
        # Load the dataset (this will use the cached version if available)
        dataset = load_dataset("ai4bharat/BPCC", "bpcc-seed-latest")
        
        print(f"\nDataset loaded in {(time.time() - start_time)/60:.2f} minutes")
        
        # Print available language splits
        print("\nAvailable language splits:", list(dataset.keys()))
        
        # Use the Hindi split
        hindi_split = 'hin_Deva'
        if hindi_split not in dataset:
            print(f"\nError: {hindi_split} split not found in dataset")
            print("Available splits:", list(dataset.keys()))
            return None
            
        print(f"\nUsing Hindi split: {hindi_split}")
        
        # Get the Hindi data
        hindi_data = dataset[hindi_split]
        total_examples = len(hindi_data)
        print(f"Total Hindi examples available: {total_examples:,}")
        
        # Create a subset if requested
        if save_subset:
            subset_size = min(subset_size, total_examples)
            print(f"\nCreating a subset of {subset_size:,} examples...")
            
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save the subset
            subset_path = output_path / f"hindi_subset_{subset_size}"
            if not subset_path.exists():
                hindi_subset = hindi_data.select(range(subset_size))
                hindi_subset.save_to_disk(str(subset_path))
                print(f"Subset saved to: {subset_path}")
            else:
                print(f"Subset already exists at: {subset_path}")
            
            # Also save a smaller sample for quick inspection
            sample_size = min(100, subset_size)
            sample_path = output_path / f"hindi_sample_{sample_size}"
            if not sample_path.exists():
                sample_data = hindi_data.select(range(sample_size))
                sample_data.save_to_disk(str(sample_path))
                print(f"Sample ({sample_size} examples) saved to: {sample_path}")
        
        # Show sample structure and examples
        print("\n=== Sample Structure ===")
        print(hindi_data[0])
        
        # Print summary
        print(f"\n=== Summary ===")
        print(f"Total Hindi data points: {total_examples:,}")
        if save_subset:
            print(f"Created subset of size: {subset_size:,}")
        
        # Show some examples
        print("\n=== First 3 Examples ===")
        for i in range(min(3, len(hindi_data))):
            print(f"\n--- Example {i+1} ---")
            example = hindi_data[i]
            print(f"Source ({example['src_lang']}): {example['src'].strip()}")
            print(f"Target ({example['tgt_lang']}): {example['tgt'].strip()}")
        
        return hindi_data
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None

if __name__ == "__main__":
    # Set these parameters as needed
    SAVE_SUBSET = True
    SUBSET_SIZE = 5000
    OUTPUT_DIR = "hindi_data"
    
    hindi_data = filter_hindi_data(
        save_subset=SAVE_SUBSET,
        subset_size=SUBSET_SIZE,
        output_dir=OUTPUT_DIR
    )
    
    if hindi_data is not None:
        print("\nScript completed successfully!")

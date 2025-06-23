from datasets import load_from_disk

def inspect_prompts(dataset_path, num_examples=5):
    try:
        # Load the dataset
        print(f"Loading dataset from {dataset_path}...")
        dataset = load_from_disk(dataset_path)
        
        # Print dataset info
        print(f"\nDataset loaded successfully with {len(dataset)} examples")
        print(f"Available columns: {dataset.column_names}")
        
        # Print first few prompts
        print(f"\nFirst {num_examples} prompts:")
        print("-" * 80)
        for i in range(min(num_examples, len(dataset))):
            print(f"Example {i + 1}:")
            print(f"Prompt: {dataset[i].get('prompt', 'No prompt found')}")
            if 'response' in dataset[i]:
                print(f"Response: {dataset[i]['response']}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # You can change this path to inspect different datasets
    dataset_path = "hindi_train_test/train"
    inspect_prompts(dataset_path, num_examples=5)

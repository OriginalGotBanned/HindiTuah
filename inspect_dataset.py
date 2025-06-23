from datasets import load_dataset
import json

def inspect_dataset():
    print("Loading dataset to inspect its structure...")
    
    try:
        # Load the dataset
        dataset = load_dataset("ai4bharat/BPCC", "bpcc-seed-latest")
        
        # Print basic info
        print("\n=== Dataset Info ===")
        print(f"Type: {type(dataset)}")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Inspect each split
        for split_name in dataset.keys():
            print(f"\n=== Split: {split_name} ===")
            split = dataset[split_name]
            print(f"Number of examples: {len(split)}")
            
            # Show first example
            if len(split) > 0:
                print("\nFirst example structure:")
                example = split[0]
                print(json.dumps(example, indent=2, ensure_ascii=False))
                
                # Print all available fields
                print("\nAvailable fields:")
                for key in example.keys():
                    value = example[key]
                    print(f"- {key}: {type(value).__name__}")
                    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    inspect_dataset()

from datasets import load_dataset
import time

print("Starting to load BPCC dataset (this may take around 30 minutes)...")
start_time = time.time()

try:
    # Load the BPCC dataset with the latest seed configuration
    dataset = load_dataset("ai4bharat/BPCC", "bpcc-seed-latest")
    
    # Print dataset information
    print("\nDataset loaded successfully!")
    print(f"Time taken: {(time.time() - start_time)/60:.2f} minutes")
    print("\nDataset structure:")
    print(dataset)
    
    # Show a few examples
    print("\nFirst few examples:")
    for i in range(min(3, len(dataset['train']))):
        print(f"\nExample {i+1}:")
        print(dataset['train'][i])
        
except Exception as e:
    print(f"\nError loading dataset: {str(e)}")
    print("\nAvailable configurations:")
    print("- bpcc-seed-latest")
    print("- bpcc-seed-v1")
    print("- daily")
    print("- comparable")
    print("- ilci")
    print("- nllb-seed")
    print("- massive")
    print("- nllb-filtered")
    print("- samanantar-filtered")
    print("- samanantar++")
import time

print("Starting to load BPCC dataset (this may take around 30 minutes)...")
start_time = time.time()

# Load the human-annotated subset of BPCC dataset
dataset = load_dataset("ai4bharat/BPCC", "human")

# Print dataset information
print("\nDataset loaded successfully!")
print(f"Time taken: {(time.time() - start_time)/60:.2f} minutes")
print("\nDataset structure:")
print(dataset)

# Show a few examples
print("\nFirst few examples:")
for i in range(3):
    print(f"\nExample {i+1}:")
    print(dataset['train'][i])

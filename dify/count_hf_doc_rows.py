from datasets import load_dataset

def count_rows():
    dataset_name = "m-ric/huggingface_doc"
    print(f"Loading dataset {dataset_name} (streaming mode)...")
    
    try:
        # Load in streaming mode to avoid downloading the entire dataset just for counting
        ds = load_dataset(dataset_name, streaming=True)
        
        total_rows = 0
        for split in ds:
            print(f"Counting rows in split: '{split}'...")
            count = 0
            for _ in ds[split]:
                count += 1
            print(f"  Rows in '{split}': {count}")
            total_rows += count
            
        print(f"\nTotal rows in {dataset_name}: {total_rows}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    count_rows()

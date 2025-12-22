from datasets import load_dataset

def main():
    print("Loading dataset...")
    # Load the dataset
    ds = load_dataset("m-ric/huggingface_doc", split="train")
    print("Dataset loaded. Calculating max length...")
    
    # Calculate max length of "text" column
    # Iterate through the dataset to avoid loading everything into memory at once if possible,
    # though with standard load_dataset it's usually memory mapped.
    max_len = max((len(item['text']) for item in ds), default=0)
    
    print(f"Max length of 'text' column: {max_len}")

if __name__ == "__main__":
    main()

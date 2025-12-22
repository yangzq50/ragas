import argparse
from datasets import load_dataset
import sys

def calc_limit(dataset_name, limit_mb, column_name, split_name):
    limit_bytes = limit_mb * 1024 * 1024
    
    print(f"Loading dataset {dataset_name} (streaming mode)...")
    try:
        ds = load_dataset(dataset_name, streaming=True)
        
        if split_name not in ds:
            print(f"Split '{split_name}' not found in dataset.")
            return

        print(f"Processing split: '{split_name}' until {limit_mb}MB limit on column '{column_name}'...")
        
        total_bytes = 0
        total_rows = 0
        
        for row in ds[split_name]:
            text = row.get(column_name, '')
            if text is None:
                text = ""
            # encode to utf-8 to get byte size
            text_size = len(str(text).encode('utf-8'))
            
            if total_bytes + text_size > limit_bytes:
                print(f"Limit reached/exceeded at row: {total_rows + 1}")
                print(f"Total size would be: {total_bytes + text_size} bytes ({(total_bytes + text_size) / (1024*1024):.2f} MB)")
                print(f"Previous total size (within limit): {total_bytes} bytes ({total_bytes / (1024*1024):.2f} MB)")
                print(f"Count of rows fitting within limit: {total_rows}")
                return
            
            total_bytes += text_size
            total_rows += 1
            
        print(f"Finished dataset without exceeding limit. Total rows: {total_rows}, Total size: {total_bytes} bytes ({total_bytes / (1024*1024):.2f} MB)")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate row count exceeding a size limit for a Hugging Face dataset.")
    parser.add_argument("--dataset", type=str, default="m-ric/huggingface_doc", help="Dataset name (default: m-ric/huggingface_doc)")
    parser.add_argument("--limit", type=float, default=15, help="Size limit in MB (default: 15)")
    parser.add_argument("--column", type=str, default="text", help="Column name to check size of (default: text)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train)")

    args = parser.parse_args()
    
    calc_limit(args.dataset, args.limit, args.column, args.split)

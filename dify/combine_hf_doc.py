import argparse
from datasets import load_dataset
import sys

def combine_rows(dataset_name, limit_rows, column_name, split_name, delimiter, output_file):
    print(f"Loading dataset {dataset_name} (streaming mode)...")
    try:
        ds = load_dataset(dataset_name, streaming=True)

        if split_name not in ds:
            print(f"Split '{split_name}' not found in dataset.")
            return

        print(f"Processing split: '{split_name}'. Collecting first {limit_rows} rows from column '{column_name}'...")
        
        collected_texts = []
        rows_processed = 0
        
        for row in ds[split_name]:
            if limit_rows is not None and rows_processed >= limit_rows:
                break
                
            text = row.get(column_name, '')
            if text is None:
                text = ""
            collected_texts.append(str(text))
            rows_processed += 1
            
        combined_text = delimiter.join(collected_texts)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            print(f"Successfully wrote {rows_processed} combined rows to {output_file}")
        else:
             # Print a snippet if no output file
            print(f"Combined text length: {len(combined_text)} characters.")
            print(f"First 500 chars snippet:\n{combined_text[:500]}...")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine first N rows of a Hugging Face dataset column.")
    parser.add_argument("--dataset", type=str, default="m-ric/huggingface_doc", help="Dataset name (default: m-ric/huggingface_doc)")
    parser.add_argument("--rows", type=int, default=1900, help="Number of rows to combine (default: 1900)")
    parser.add_argument("--column", type=str, default="text", help="Column name to combine (default: text)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train)")
    # Interpret encoded newlines for the delimiter argument
    parser.add_argument("--delimiter", type=str, default="\n\n\n\n\n\n", help="Delimiter to join texts (default: \\n\\n\\n)")
    parser.add_argument("--output", type=str, default="combined_output.txt", help="Output file path (default: combined_output.txt)")

    args = parser.parse_args()
    
    # Process delimiter to handle escaped characters like \n
    delimiter = args.delimiter.encode('utf-8').decode('unicode_escape')
    
    combine_rows(args.dataset, args.rows, args.column, args.split, delimiter, args.output)

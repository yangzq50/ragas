import datasets
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
qa_count = 0
doc_count = 0

def get_question_doc_mapping():
    logger.info("Loading m-ric/huggingface_doc_qa_eval split 'train'...")
    try:
        qa_eval_ds = datasets.load_dataset("m-ric/huggingface_doc_qa_eval", split="train")
    except Exception as e:
        logger.error(f"Error loading qa_eval dataset: {e}")
        return {}

    logger.info("Loading m-ric/huggingface_doc split 'train'...")
    try:
        doc_ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")
    except Exception as e:
        logger.error(f"Error loading doc dataset: {e}")
        return{}

    global qa_count
    global doc_count
    qa_count = len(qa_eval_ds)
    logger.info(f"1. Total row count for m-ric/huggingface_doc_qa_eval (train): {qa_count}")
    
    doc_count = len(doc_ds)
    logger.info(f"2. Total row count for m-ric/huggingface_doc (train): {doc_count}")

    logger.info("Building lookup index for m-ric/huggingface_doc 'text' column...")
    doc_text_to_index = {}
    for i, row in enumerate(doc_ds):
        text = row.get("text")
        if text:
            if text not in doc_text_to_index:
                doc_text_to_index[text] = []
            doc_text_to_index[text].append(i)
    
    sorted_doc_texts = sorted(doc_text_to_index.keys())
    logger.info(f"Index built. Unique texts: {len(sorted_doc_texts)}")

    def get_indices_for_prefix(prefix):
        from bisect import bisect_left
        start_idx = bisect_left(sorted_doc_texts, prefix)
        all_indices = []
        for i in range(start_idx, len(sorted_doc_texts)):
            if sorted_doc_texts[i].startswith(prefix):
                all_indices.extend(doc_text_to_index[sorted_doc_texts[i]])
            else:
                break
        return all_indices

    logger.info("3. Checking existence of 'context' from qa_eval in doc dataset...")

    mapping = {}
    
    for i, row in enumerate(qa_eval_ds):
        context = row.get("context")
        if not context:
            raise ValueError(f"Row {i}: Context is empty/None. Row content: {row}")

        # Check for exact matches
        exact_indices = doc_text_to_index.get(context, [])
        
        # Check for all strings sharing this context as prefix
        all_indices_for_prefix = get_indices_for_prefix(context)
        
        # Filter out exact matches from prefix matches to distinguish them
        prefix_indices = [idx for idx in all_indices_for_prefix if idx not in exact_indices]
        
        if exact_indices or prefix_indices:
             match_info = []
             if exact_indices:
                 match_info.append(f"Exact match rows {sorted(list(set(exact_indices)))}")
             if prefix_indices:
                 match_info.append(f"Prefix shared by rows {sorted(list(set(prefix_indices)))}")
             
             logger.info(f"Row {i}: {' | '.join(match_info)}")
             mapping[i] = min(all_indices_for_prefix)
        else:
            raise ValueError(f"Row {i}: None (Not found in doc). Row content: {row}")

    return mapping

            #  # Try partial matching by halving the length of the prefix
            #  current_match_len = len(context) // 2
            #  matched_indices = []
             
            #  while current_match_len >= 20:
            #      prefix = context[:current_match_len]
            #      matched_indices = get_indices_for_prefix(prefix)
            #      if matched_indices:
            #          break
            #      current_match_len //= 2
             
            #  if matched_indices:
            #      found_count += 1
            #      logger.info(f"Row {i}: Partial match (first {current_match_len} chars) in doc rows {sorted(list(set(matched_indices)))}")
            #  else:
            #      not_found_count += 1
            #      logger.info(f"Row {i}: None (Not found in doc). Row content: {row}")


def main():
    mapping = get_question_doc_mapping()
    logger.info(f"Collected mapping for {len(mapping)} rows.")

    for k, v in mapping.items():
        logger.info(f"Question Row {k} -> Doc Row {v}")

    # also print reverse sorted Doc Row -> Question Row, Doc Row is the key, sorted by Doc Row
    reverse_mapping = {v: k for k, v in mapping.items()}
    # check reverse_mapping and mapping count
    logger.info(f"Reverse mapping count: {len(reverse_mapping)}")
    logger.info(f"Mapping count: {len(mapping)}")
    for k, v in sorted(reverse_mapping.items(), key=lambda item: item[0]):
        logger.info(f"Doc Row {k} -> Question Row {v}")

    logger.info("-" * 30)
    logger.info("Summary:")
    logger.info(f"Total QA Rows: {qa_count}")
    logger.info(f"Total Doc Rows: {doc_count}")
    logger.info(f"Total Matches Found: {len(mapping)}")

if __name__ == "__main__":
    main()

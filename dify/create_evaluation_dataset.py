import os
import argparse
import json
import logging
import requests
from datasets import load_dataset
from openai import OpenAI
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def retrieve_from_dify(query, dataset_id, api_key, base_url="https://api.dify.ai/v1"):
    """
    Performs Economical Inverted Index Retrieval in Dify.
    Ref: https://docs.dify.ai/api-reference/datasets/retrieve-chunks-from-a-knowledge-base-test-retrieval
    """
    url = f"{base_url}/datasets/{dataset_id}/retrieve"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "retrieval_model": {
            "score_threshold_enabled": False,
            "search_method": "semantic_search",
            "top_k": 10,
            "reranking_enable": True,
            "reranking_mode": "reranking_model",
            "reranking_model": {
                "reranking_provider_name": "langgenius/jina/jina",
                "reranking_model_name": "jina-reranker-v3"
            },
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        records = data.get("records", [])
        contexts = [record.get("segment", {}).get("content", "") for record in records]
        return contexts
    except Exception as e:
        logger.error(f"Dify retrieval error for query '{query}': {e}")
        raise

def generate_response(query, contexts, client, model="gpt-4o-mini"):
    """
    Generates a response based on the question and retrieved contexts using OpenAI.
    """
    if not contexts:
        return "No context found to answer the question."
    
    context_str = "\n\n".join(contexts)
    prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question. If the answer is not in the context, say you don't know."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI generation error for query '{query}': {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate Evaluation Dataset for Dify.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of rows to process.")
    parser.add_argument("--output", type=str, default="evaluation_dataset.jsonl", help="Output file path.")
    args = parser.parse_args()

    # Load environment variables
    api_key = os.getenv("DIFY_API_KEY")
    dataset_id = os.getenv("DIFY_DATASET_ID")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not all([api_key, dataset_id, openai_key]):
        logger.error("Missing environment variables: DIFY_API_KEY, DIFY_DATASET_ID, or OPENAI_API_KEY.")
        import sys
        sys.exit(1)

    openai_client = OpenAI(api_key=openai_key)

    logger.info("Loading m-ric/huggingface_doc_qa_eval dataset...")
    ds = load_dataset("m-ric/huggingface_doc_qa_eval", split="train")
    
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    samples = []
    logger.info(f"Processing {len(ds)} rows...")

    for row in tqdm(ds):
        question = row["question"]
        reference = row["answer"]
        
        # 1. Retrieve contexts from Dify
        retrieved_contexts = retrieve_from_dify(question, dataset_id, api_key)
        
        # 2. Generate response via OpenAI
        response = generate_response(question, retrieved_contexts, openai_client)
        
        # 3. Create SingleTurnSample
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=retrieved_contexts,
            response=response,
            reference=reference
        )
        samples.append(sample)

    # 4. Create EvaluationDataset and save
    eval_dataset = EvaluationDataset(samples=samples)
    eval_dataset.to_jsonl(args.output)
    logger.info(f"Evaluation dataset saved to {args.output}")

if __name__ == "__main__":
    main()

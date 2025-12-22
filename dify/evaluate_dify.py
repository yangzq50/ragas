import os
import json
import logging
import asyncio
import pandas as pd
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as tqdm_asyncio

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def evaluate_sample(sample: SingleTurnSample, metrics, llm, embeddings):
    results = {}
    for metric in metrics:
        kwargs = {}
        if metric.name == "faithfulness":
            kwargs = {
                "user_input": sample.user_input,
                "response": sample.response,
                "retrieved_contexts": sample.retrieved_contexts,
            }
        elif metric.name == "answer_relevancy":
            kwargs = {
                "user_input": sample.user_input,
                "response": sample.response,
            }
        elif metric.name == "context_precision":
            kwargs = {
                "user_input": sample.user_input,
                "reference": sample.reference,
                "retrieved_contexts": sample.retrieved_contexts,
            }
        elif metric.name == "context_recall":
            kwargs = {
                "user_input": sample.user_input,
                "reference": sample.reference,
                "retrieved_contexts": sample.retrieved_contexts,
            }
        else:
            raise ValueError(f"Unknown metric: {metric.name}")
            
        try:
            score = await metric.ascore(**kwargs)
            results[metric.name] = score.value
        except Exception as e:
            logger.error(f"Error evaluating metric {metric.name} for sample: {e}")
            results[metric.name] = None
    return results

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Dify RAG application using Ragas.")
    parser.add_argument("input_file", nargs="?", default="evaluation_dataset.jsonl", 
                        help="Path to the evaluation dataset JSONL file (default: evaluation_dataset.jsonl)")
    args = parser.parse_args()

    # Load environment variables or use hardcoded defaults from user context
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize OpenAI Client
    client = AsyncOpenAI(api_key=openai_key)
    
    # Initialize LLM and Embeddings for evaluation
    llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = OpenAIEmbeddings(client=client)

    input_file = args.input_file
    
    if not os.path.exists(input_file):
        logger.error(f"Evaluation dataset file not found: {input_file}")
        return

    logger.info(f"Loading evaluation dataset from {input_file}...")
    try:
        data = []
        with open(input_file, "r") as f:
            for i, line in enumerate(f):
                data.append(json.loads(line))
        
        eval_dataset = EvaluationDataset.from_list(data)
        logger.info(f"Loaded {len(eval_dataset)} samples for evaluation.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Initialize Metrics
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm)
    ]

    # Run Evaluation
    logger.info("Starting Ragas evaluation using new API...")
    
    tasks = [evaluate_sample(sample, metrics, llm, embeddings) for sample in eval_dataset]
    all_scores = await tqdm_asyncio.gather(*tasks, desc="Evaluating samples")

    # Combine data and scores for output
    results_data = []
    for i, sample in enumerate(eval_dataset):
        row = sample.to_dict()
        row.update(all_scores[i])
        results_data.append(row)

    df_results = pd.DataFrame(results_data)
    
    print("\nEvaluation Results (Mean Scores):")
    for metric in metrics:
        mean_score = df_results[metric.name].mean()
        print(f"{metric.name}: {mean_score:.4f}")
    
    # Save results to a file
    results_file = "evaluation_results.csv"
    df_results.to_csv(results_file, index=False)
    logger.info(f"Detailed results saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())

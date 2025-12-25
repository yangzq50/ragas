# RAGAS Evaluation Report: Dify RAG System

## 1. Introduction to RAGAS

**RAGAS** (Retrieval Augmented Generation Assessment) is a framework for evaluating the performance of Retrieval Augmented Generation (RAG) systems. RAG systems combine retrieval mechanisms with large language models to provide more accurate and contextually grounded responses.

RAGAS provides a set of metrics that evaluate different aspects of RAG performance:
- **Retrieval Quality**: How well does the system retrieve relevant information?
- **Generation Quality**: How well does the system generate accurate and relevant responses?

These metrics allow developers to identify weaknesses in their RAG pipelines and make data-driven improvements.

---

## 2. Evaluation Workflow

The evaluation was conducted in two steps:

### Step 1: Create Evaluation Dataset (`create_evaluation_dataset.py`)

This script generates the evaluation dataset by:

1. **Loading Test Data**: Uses the `m-ric/huggingface_doc_qa_eval` dataset containing question-answer pairs from HuggingFace documentation.

2. **Retrieving Context from Dify**: For each question, calls the Dify Knowledge Base API to retrieve relevant document chunks using configured retrieval settings:
   - **Search Method**: Semantic search
   - **Top-K**: Number of chunks to retrieve (3 or 10)
   - **Reranking**: Optional reranking using `jina-reranker-v3`

3. **Generating Responses**: Uses OpenAI `gpt-4o-mini` to generate answers based on the retrieved contexts.

4. **Saving Dataset**: Outputs a JSONL file containing:
   - `user_input`: The original question
   - `retrieved_contexts`: List of retrieved document chunks
   - `response`: Generated answer
   - `reference`: Ground truth answer

### Step 2: Evaluate with RAGAS (`evaluate_dify.py`)

This script evaluates the generated dataset using 4 RAGAS metrics:

1. **Loads** the evaluation dataset from JSONL format
2. **Initializes** metrics with OpenAI LLM and embeddings
3. **Evaluates** each sample asynchronously for all 4 metrics
4. **Outputs** results to CSV with all metric scores

---

## 3. RAGAS Metrics

### 3.1 Faithfulness

**Definition**: Measures how factually consistent the response is with the retrieved context. A faithful response only contains information that can be verified from the context.

**Calculation**:
$$\text{Faithfulness} = \frac{\text{Claims in response supported by context}}{\text{Total claims in response}}$$

**Range**: 0 to 1 (higher is better)

**Use Case**: Detects hallucinations where the model generates facts not present in the retrieved documents.

---

### 3.2 Answer Relevancy

**Definition**: Measures how well the response addresses the user's question. Penalizes incomplete or off-topic answers.

**Calculation**:
1. Generate synthetic questions from the response
2. Compute cosine similarity between embeddings of original question and synthetic questions
3. Average the similarities

$$\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \text{cosine\_sim}(E_{generated_i}, E_{original})$$

**Range**: 0 to 1 (higher is better)

**Use Case**: Ensures the response directly addresses what the user asked.

---

### 3.3 Context Precision

**Definition**: Evaluates whether relevant documents are ranked higher than irrelevant ones in the retrieved context.

**Calculation**:
$$\text{Context Precision@K} = \frac{\sum_{k=1}^{K} (\text{Precision@k} \times v_k)}{\text{Total relevant items in top K}}$$

Where $v_k \in \{0, 1\}$ indicates relevance at rank $k$.

**Range**: 0 to 1 (higher is better)

**Use Case**: Measures retrieval ranking quality - whether the most useful chunks appear first.

---

### 3.4 Context Recall

**Definition**: Measures how much of the ground truth answer can be attributed to the retrieved context.

**Calculation**:
$$\text{Context Recall} = \frac{\text{Claims in reference supported by context}}{\text{Total claims in reference}}$$

**Range**: 0 to 1 (higher is better)

**Use Case**: Ensures the retrieval system captures all necessary information to answer the question.

---

## 4. Evaluation Results Analysis

### 4.1 Configurations Tested

| Configuration | Top-K | Reranking |
|--------------|-------|-----------|
| top3_rerank | 3 | Enabled (jina-reranker-v3) |
| top3_no_rerank | 3 | Disabled |
| top10_rerank | 10 | Enabled (jina-reranker-v3) |
| top10_no_rerank | 10 | Disabled |

Each configuration was evaluated on 65 samples from the HuggingFace documentation QA dataset.

### 4.2 Missing Values Analysis

Some samples have missing metric values due to LLM evaluation failures (e.g., malformed responses, API errors):

| Configuration | Missing Values | Affected Metrics |
|--------------|----------------|------------------|
| top3_rerank | 1 (1.5%) | faithfulness, context_precision, context_recall |
| top3_no_rerank | 0 (0.0%) | - |
| top10_rerank | 0 (0.0%) | - |
| top10_no_rerank | 1 (1.5%) | answer_relevancy |

**Note**: Missing values are excluded from statistical calculations.

### 4.3 Results Summary

| Metric | top3_rerank | top3_no_rerank | top10_rerank | top10_no_rerank |
|--------|-------------|----------------|--------------|-----------------|
| **Faithfulness** | 0.7604 | 0.8016 | 0.8443 | **0.8747** |
| **Answer Relevancy** | 0.8251 | 0.8159 | **0.9180** | 0.9134 |
| **Context Precision** | **0.6862** | 0.6308 | 0.6092 | 0.3515 |
| **Context Recall** | 0.8047 | 0.8077 | 0.8795 | **0.9282** |

### 4.4 Key Findings

#### Faithfulness
- **Best**: top10_no_rerank (0.8747)
- **Trend**: More retrieved chunks (top10) leads to higher faithfulness
- **Insight**: Additional context helps the model generate more factually grounded responses

#### Answer Relevancy
- **Best**: top10_rerank (0.9180)
- **Trend**: top10 configurations significantly outperform top3
- **Insight**: More context enables more complete and relevant answers

#### Context Precision
- **Best**: top3_rerank (0.6862)
- **Trend**: Reranking dramatically improves precision, especially with more chunks
- **Insight**: Without reranking, top10 has very low precision (0.3515) because irrelevant chunks dilute the ranking

#### Context Recall
- **Best**: top10_no_rerank (0.9282)
- **Trend**: More chunks consistently improve recall
- **Insight**: Retrieving more documents ensures relevant information is not missed

### 4.5 Recommendations

1. **For high precision requirements**: Use **top3_rerank** - fewer chunks with reranking ensures relevant content is prioritized

2. **For high recall requirements**: Use **top10_no_rerank** - more chunks maximize coverage of relevant information

3. **For balanced performance**: Use **top10_rerank** - best overall balance with high answer relevancy and good recall while maintaining reasonable precision

4. **Always enable reranking with top10**: Without reranking, context precision drops significantly (0.61 → 0.35), indicating many irrelevant chunks in results

---

## 5. Conclusion

This evaluation demonstrates the trade-offs between different retrieval configurations:

- **Top-K setting**: More chunks improve recall and faithfulness but can hurt precision without reranking
- **Reranking**: Critical for maintaining context precision, especially with larger top-K values

The **top10_rerank** configuration provides the best overall performance for most use cases, balancing high answer relevancy (0.918) with strong context recall (0.880) and acceptable precision (0.609).

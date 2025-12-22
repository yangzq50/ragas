
## Document from Text
ref: https://docs.dify.ai/api-reference/documents/create-a-document-from-text

Creates a new document within an existing knowledge base directly from text content.

### Create a Document from Text
```python
import requests

url = "https://api.dify.ai/v1/datasets/{dataset_id}/document/create-by-text"

payload = {
    "name": "<string>",
    "text": "<string>",
    "indexing_technique": "high_quality",
    "doc_form": "text_model",
    "doc_language": "English",
    "process_rule": {
        "mode": "automatic",
        "rules": {
            "pre_processing_rules": [
                {
                    "id": "remove_extra_spaces",
                    "enabled": True
                }
            ],
            "segmentation": {
                "separator": "\n\n",
                "max_tokens": 1024,
                "chunk_overlap": 50
            }
        }
    },
    "retrieval_model": {
        "search_method": "hybrid_search",
        "reranking_enable": True,
        "reranking_mode": "reranking_model",
        "reranking_model": {
            "reranking_provider_name": "jina",
            "reranking_model_name": "jina-reranker-v3"
        },
        "top_k": 10,
        "score_threshold_enabled": False
    },
    "embedding_model": "jina-embeddings-v4",
    "embedding_model_provider": "jina"
}
headers = {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)
```
### 200
```json
{
  "document": {
    "id": "3c90c3cc-0d44-4b50-8888-8dd25736052a",
    "position": 123,
    "data_source_type": "<string>",
    "data_source_info": {},
    "dataset_process_rule_id": "3c90c3cc-0d44-4b50-8888-8dd25736052a",
    "name": "<string>",
    "created_from": "<string>",
    "created_by": "3c90c3cc-0d44-4b50-8888-8dd25736052a",
    "created_at": 123,
    "tokens": 123,
    "indexing_status": "<string>",
    "error": "<string>",
    "enabled": true,
    "disabled_at": 123,
    "disabled_by": "3c90c3cc-0d44-4b50-8888-8dd25736052a",
    "archived": true,
    "display_status": "<string>",
    "word_count": 123,
    "hit_count": 123,
    "doc_form": "<string>"
  },
  "batch": "<string>"
}
```

### Retrieve Chunks from a Knowledge Base / Test Retrieval
ref: https://docs.dify.ai/api-reference/datasets/retrieve-chunks-from-a-knowledge-base-test-retrieval

Performs a search query against a knowledge base to retrieve the most relevant chunks (segments). This endpoint can be used for both production retrieval and test retrieval.

### Retrieve Chunks from a Knowledge Base / Test Retrieval
```python
import requests

url = "https://api.dify.ai/v1/datasets/{dataset_id}/retrieve"

payload = {
    "query": "<string>",
    "retrieval_model": {
        "search_method": "hybrid_search",
        "reranking_enable": True,
        "reranking_mode": "reranking_model",
        "reranking_model": {
            "reranking_provider_name": "jina",
            "reranking_model_name": "jina-reranker-v3"
        },
        "top_k": 10,
        "score_threshold_enabled": False
    }
}
headers = {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)
```
200
```json
{
  "query": {
    "content": "<string>"
  },
  "records": [
    {
      "segment": {
        "id": "3c90c3cc-0d44-4b50-8888-8dd25736052a",
        "position": 123,
        "document_id": "3c90c3cc-0d44-4b50-8888-8dd25736052a",
        "content": "<string>",
        "answer": "<string>",
        "word_count": 123,
        "tokens": 123,
        "keywords": [
          "<string>"
        ],
        "index_node_id": "<string>",
        "index_node_hash": "<string>",
        "hit_count": 123,
        "enabled": true,
        "disabled_at": 123,
        "disabled_by": "3c90c3cc-0d44-4b50-8888-8dd25736052a",
        "status": "<string>",
        "created_by": "3c90c3cc-0d44-4b50-8888-8dd25736052a",
        "created_at": 123,
        "indexing_at": 123,
        "completed_at": 123,
        "error": "<string>",
        "stopped_at": 123,
        "document": {
          "id": "3c90c3cc-0d44-4b50-8888-8dd25736052a",
          "data_source_type": "<string>",
          "name": "<string>"
        }
      },
      "score": 123
    }
  ]
}
```

### Economical Inverted Index Retrieval
```python
import requests

url = "https://api.dify.ai/v1/datasets/{dataset_id}/retrieve"

payload = {
    "query": "<string>",
    "retrieval_model": {
        "search_method": "keyword_search",
        "reranking_enable": False,
        "top_k": 10,
        "score_threshold_enabled": False
    }
}
headers = {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)
```
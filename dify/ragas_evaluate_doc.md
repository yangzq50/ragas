# ragas evaluate usage

## deprecated usage
ref:
https://github.com/vibrantlabsai/ragas/issues/2525
collections metrics don't work with evaluate/aevaluate. `evaluate/aevaluate` will be deprecated and removed soon.
Please update evaluate_dify.py to use new evaluate api listed below:

## new usage

### Faithfulness

The **Faithfulness** metric measures how factually consistent a `response` is with the `retrieved context`. It ranges from 0 to 1, with higher scores indicating better consistency.

A response is considered **faithful** if all its claims can be supported by the retrieved context.

To calculate this:
1. Identify all the claims in the response.
2. Check each claim to see if it can be inferred from the retrieved context.
3. Compute the faithfulness score using the formula:

$$
\text{Faithfulness Score} = \frac{\text{Number of claims in the response supported by the retrieved context}}{\text{Total number of claims in the response}}
$$


#### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = Faithfulness(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ]
)
print(f"Faithfulness Score: {result.value}")
```

Output:

```
Faithfulness Score: 1.0
```

### Answer Relevancy

The **Answer Relevancy** metric measures how relevant a response is to the user input. It ranges from 0 to 1, with higher scores indicating better alignment with the user input.

An answer is considered relevant if it directly and appropriately addresses the original question. This metric focuses on how well the answer matches the intent of the question, without evaluating factual accuracy. It penalizes answers that are incomplete or include unnecessary details.

This metric is calculated using the `user_input` and the `response` as follows:  

1. Generate a set of artificial questions (default is 3) based on the response. These questions are designed to reflect the content of the response.  
2. Compute the cosine similarity between the embedding of the user input ($E_o$) and the embedding of each generated question ($E_{g_i}$).  
3. Take the average of these cosine similarity scores to get the **Answer Relevancy**:  

$$
\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \text{cosine similarity}(E_{g_i}, E_o)
$$  

$$
\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \frac{E_{g_i} \cdot E_o}{\|E_{g_i}\| \|E_o\|}
$$  

Where:  
- $E_{g_i}$: Embedding of the $i^{th}$ generated question.  
- $E_o$: Embedding of the user input.  
- $N$: Number of generated questions (default is 3, configurable via `strictness` parameter).  

**Note**: While the score usually falls between 0 and 1, it is not guaranteed due to cosine similarity's mathematical range of -1 to 1.

#### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerRelevancy

# Setup LLM and embeddings
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

# Create metric
scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)

# Evaluate
result = await scorer.ascore(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967"
)
print(f"Answer Relevancy Score: {result.value}")
```

Output:

```
Answer Relevancy Score: 0.9165088378587264
```

### Context Precision

Context Precision is a metric that evaluates the retriever's ability to rank relevant chunks higher than irrelevant ones for a given query in the retrieved context. Specifically, it assesses the degree to which relevant chunks in the retrieved context are placed at the top of the ranking.

It is calculated as the mean of the precision@k for each chunk in the context. Precision@k is the ratio of the number of relevant chunks at rank k to the total number of chunks at rank k.

$$
\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{\text{Total number of relevant items in the top } K \text{ results}}
$$

$$
\text{Precision@k} = {\text{true positives@k} \over  (\text{true positives@k} + \text{false positives@k})}
$$

Where $K$ is the total number of chunks in `retrieved_contexts` and $v_k \in \{0, 1\}$ is the relevance indicator at rank $k$.

#### Examples

#### Context Precision

The `ContextPrecision` metric evaluates whether retrieved contexts are useful for answering a question by comparing each context against a reference answer. Use this when you have a reference answer available.

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextPrecision(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=[
        "The Eiffel Tower is located in Paris.",
        "The Brandenburg Gate is located in Berlin."
    ]
)
print(f"Context Precision Score: {result.value}")
```

Output:
```
Context Precision Score: 0.9999999999
```

### Context Recall

Context Recall measures how many of the relevant documents (or pieces of information) were successfully retrieved. It focuses on not missing important results. Higher recall means fewer relevant documents were left out. In short, recall is about not missing anything important. 

Since it is about not missing anything, calculating context recall always requires a reference to compare against. The LLM-based Context Recall metric uses `reference` as a proxy to `reference_contexts`, which makes it easier to use as annotating reference contexts can be very time-consuming. To estimate context recall from the `reference`, the reference is broken down into claims, and each claim is analyzed to determine whether it can be attributed to the retrieved context or not. In an ideal scenario, all claims in the reference answer should be attributable to the retrieved context.

The formula for calculating context recall is as follows:

$$
\text{Context Recall} = \frac{\text{Number of claims in the reference supported by the retrieved context}}{\text{Total number of claims in the reference}}
$$

#### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextRecall(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    retrieved_contexts=["Paris is the capital of France."],
    reference="The Eiffel Tower is located in Paris."
)
print(f"Context Recall Score: {result.value}")
```

Output:

```
Context Recall Score: 1.0
```



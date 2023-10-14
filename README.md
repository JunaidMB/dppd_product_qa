# Dynamic Personalised Product Display - Product Question and Answer

This repository contains code which uses Retrieval Augmented Generation (RAG) to answer user queries about a product catalogue. The approach uses Cohere's embeddings and rerank model to perform dense retrieval followed by reranking results for relevancy. Once the results have been returned, they are passed as context, in a prompt template, to the ChatGPT API to return a response that answers the user query.

## Repository Structure

0. `semantic_search_tools`: A module which contains helper functions necessary for analysis.

1. `data`: Directory containing the product catalogue data as a JSON file and holds the vector index.

2. `product_qa_analysis`: Contains the scripts for performing dense retrieval and LLM generation. Also contains the `anthropomorphic_product_qa.py` which is product QA for a single product.

3. `prompt_templates`: Directory containing prompts to use with LLMs.

4. `scripts`: Contains product qa analysis as executable scripts.
   
5. `validation`: Directory containing validation analysis for the product QA analysis. 


## Areas for Improvement

1. Intergrate external memory: The script initialises a memory variable and makes an input variable in the prompt variable but I haven't implemented a working workflow to update memory and include it in the context. This will be the next milestone. This will also include caching to answer repeat queries.

2. Reduce latency: The response is not fast (~2 seconds). The workflow must be optimised to use appropriate data structures (Polars vs Pandas), make all lists into arrays and explore a faster search index.

3. Include screening for unsuitable queries: Some queries do not need to be entertained by the workflow or may be too simple to require a similarity search - thereby requiring a lookup. Need to add logic before the search workflow to deal with these queries without using the RAG workflow. Another example is when we ask global questions like "How many products are there?" or "what is the cheapest product?". This method is good at retrieving product level information but not good at answering global and cross catalogue levels questions. This is a limitation of the restrictive retrival and rerank parameter settings. Maybe we can construct a metadata structure to contain global data or we can explore tree of thought prompting.

4. Explore ways to reduce hallucination: There is an odd behaviour where the product name is altered to fit a user query (but product price and description is identical). The model fabricates products similar to catalogue products to appeal to the user. Need to address way to control this.

5. Extensive testing of queries and explore RAGAS: Include formalised testing of output. This looks interesting - [RAGAS](https://github.com/explodinggradients/ragas)
   
6. Look into sanitising user inputs and model outputs: Filter unsuitable, irrelevant or address repeated queries.

**Note**: I used pip-tools to create requirements.txt. Use `pip-sync` inside a virtual environment to load dependencies. A Cohere API key is required.

## References

1. [Large Language Models with Semantic Search by Cohere](https://www.deeplearning.ai/short-courses/large-language-models-semantic-search/)
2. [Cohere](https://cohere.com/)
3. [Lakera Guard](https://platform.lakera.ai/docs/quickstart)
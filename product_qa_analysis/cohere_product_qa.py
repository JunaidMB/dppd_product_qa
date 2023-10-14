import cohere
import json
import logging
import numpy as np
import os
import time
from annoy import AnnoyIndex
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import load_prompt
import pandas as pd
from pathlib import Path
from pprint import pprint
from semantic_search_tools.create_index import create_annoy_index
from semantic_search_tools.semantic_search import dense_retrieval, generate_llm_answer, reformat_dict

load_dotenv(override=True)

data_dir = Path("data/product_catalogue_data")
prompt_dir = Path("prompt_templates")

query = "Do you have any footwear for men?"
perform_rerank = True

# Initialise Cohere Client
cohere_apikey = os.environ.get("COHERE_APIKEY")
co = cohere.Client(cohere_apikey)

# Load Product Descriptions
with open(data_dir / "product_descriptions.json", "r") as openfile:
    product_descriptions = json.load(openfile)

product_description_list = [reformat_dict(prod) for prod in product_descriptions]

# Take an input query
#start_time = time.time()

# Embed user query
query_embed = co.embed(texts=[query], model="embed-english-v2.0").embeddings

# Create an index
index_name = data_dir/"product_embedding"
index_name_full = "".join([str(index_name), ".ann"])

logging.info("Creating an index")
create_annoy_index(data_to_index=product_description_list, index_name=index_name)
product_description_index = AnnoyIndex(np.array(query_embed[0]).shape[0], 'angular')
product_description_index.load(index_name_full)

# Perform Dense Retrieval to get most similar product description embeddings
retrieved_results = dense_retrieval(query=query, n=10, index=product_description_index)

retrieved_texts = []
for _, idx in enumerate(retrieved_results['text_id']):
    retrieved_texts.append( product_description_list[idx] )

retrieved_results['product_description_text'] = retrieved_texts
context = "\n".join(retrieved_results['product_description_text'])

if perform_rerank: 
    # Perform Rerank of answers
    reranked_results = co.rerank(query=query, documents=retrieved_results['product_description_text'], model='rerank-english-v2.0')

    relevance_scores = [i.relevance_score for i in reranked_results]
    retrieved_results["rerank_relevance_scores"] = relevance_scores

    # Filter ranked results to those above a rerank threshold
    results_df = pd.DataFrame(retrieved_results)

    rel_score_threshold = 0.9
    rel_score_thres = np.quantile(relevance_scores, q = [rel_score_threshold])

    filtered_results = results_df[results_df.rerank_relevance_scores > rel_score_thres[0]]
    
    # Build prompt to wrap our answers and give context
    context = "\n".join(filtered_results['product_description_text'].tolist())

# Use an LLM to answer question about a product
# Load LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

# Initialise external memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
memory.input_key="question"
memory.output_key="answer"

# Load Product QA Prompt Template
dense_retrieval_product_qa_prompt = load_prompt(prompt_dir / "dense_retrieval_product_qa_prompt.json")


query_response = generate_llm_answer(
    query=query,
    context=context,
    history=memory,
    llm=llm,
    prompt=dense_retrieval_product_qa_prompt
)

print(query_response)

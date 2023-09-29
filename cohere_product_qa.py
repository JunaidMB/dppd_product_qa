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
from langchain.prompts import PromptTemplate
import pandas as pd
from pprint import pprint
from semantic_search_tools.create_index import create_annoy_index
from semantic_search_tools.semantic_search import dense_retrieval, generate_llm_answer, reformat_dict

load_dotenv(override=True)

query = "Do you have any footwear for men?"
perform_rerank = True

# Initialise Cohere Client
cohere_apikey = os.environ.get("COHERE_APIKEY")
co = cohere.Client(cohere_apikey)

# Load Product Descriptions
with open("product_descriptions.json", "r") as openfile:
    product_descriptions = json.load(openfile)

product_description_list = [reformat_dict(prod) for prod in product_descriptions]

# Take an input query
#start_time = time.time()

# Embed user query
query_embed = co.embed(texts=[query], model="embed-english-v2.0").embeddings

# Create an index
index_name = "product_embedding"
index_name_full = "".join([index_name, ".ann"])

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

# Build prompt to wrap our answers and give context
template = """
    You are a helpful chatbot that answers questions accurately about a product. 
    A customer will ask questions about a product they are interested in, your role is to use the context supplied to give the customer the information they require about a product.
    
    When giving your answer, obey the rules below:
    Rule 1: Restrict your answer to information in the context. Do not mention anything in your answer unless it has reference in the context.
    Rule 2: If you don't know the answer, say that you don't know, don't try to make up an answer. 
    Rule 3: If the product does not exist, mention that you have no product that meets a customer's requirements.
    Rule 4: Keep the answer as concise as possible. 
    Rule 5: Answer in the style of a customer service chatbot

    We also provide a conversation history to see what a customer has asked before
    History: {history}

    Use the following pieces of context to answer the question at the end. 
    Context: {context}

    Answer the following in a step by step fashion:
    Question: {question}

    Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

query_response = generate_llm_answer(
    query=query,
    context=context,
    history=memory,
    llm=llm,
    prompt=QA_CHAIN_PROMPT
)

print(query_response)

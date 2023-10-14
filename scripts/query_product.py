import argparse
import cohere
import json
import logging
import numpy as np
import os
import pandas as pd
import time
from annoy import AnnoyIndex
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import load_prompt
from pprint import pprint
from pathlib import Path
from typing import Dict, List
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from semantic_search_tools.create_index import create_annoy_index
from semantic_search_tools.semantic_search import dense_retrieval, generate_llm_answer, reformat_dict

load_dotenv(override=True)

# Configure the logging settings to print logs to the console (standard output)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialise Cohere Client
cohere_apikey = os.environ.get("COHERE_APIKEY")
co = cohere.Client(cohere_apikey)

prompt_dir = Path("prompt_templates")
data_dir = Path("data/product_catalogue_data")

def main(query: str, perform_rerank: bool, product_descriptions: List[Dict], rel_score_threshold: float = 0.75) -> str:

    # Convert product descriptions into a string
    #start_time = time.time()

    logging.info(f"Converting Product Catalogue into text")
    product_description_list = [reformat_dict(prod) for prod in product_descriptions]

    # end_time = time.time()
    # section_time = end_time - start_time
    # logging.info(f"Converting Product Catalogue into text took {section_time:.2f} seconds to execute")


    # Take an input query
    #start_time = time.time()
    query = query.lower()

    # Embed user query
    # start_time = time.time()

    logging.info(f"Embedding User Query")
    query_embed = co.embed(texts=[query], model="embed-english-v2.0").embeddings

    # end_time = time.time()
    # section_time = end_time - start_time
    # logging.info(f"Embedding User Query took {section_time:.2f} seconds to execute")


    # Create an index
    # start_time = time.time()

    logging.info(f"Creating Product Index")
    index_name = data_dir / "product_embedding"
    index_name_full = "".join([str(index_name), ".ann"])


    create_annoy_index(data_to_index=product_description_list, index_name=index_name)
    product_description_index = AnnoyIndex(np.array(query_embed[0]).shape[0], 'angular')
    product_description_index.load(index_name_full)

    # end_time = time.time()
    # section_time = end_time - start_time
    # logging.info(f"Creating Product Index took {section_time:.2f} seconds to execute")

    # Perform Dense Retrieval to get most similar product description embeddings
    # start_time = time.time()

    logging.info(f"Performing Dense Retrieval and finding corresponding text")
    retrieved_results = dense_retrieval(query=query, n=10, index=product_description_index)

    retrieved_texts = []
    for _, idx in enumerate(retrieved_results['text_id']):
        retrieved_texts.append( product_description_list[idx] )

    retrieved_results['product_description_text'] = retrieved_texts
    context = "\n".join(retrieved_results['product_description_text'])

    # end_time = time.time()
    # section_time = end_time - start_time
    # logging.info(f"Performing Dense Retrieval and finding corresponding text took {section_time:.2f} seconds to execute")



    if perform_rerank: 
        # Perform Rerank of answers
        # start_time = time.time()

        logging.info(f"Performing Rerank to compute relevance between retrieved texts with user query")
        reranked_results = co.rerank(query=query, documents=retrieved_results['product_description_text'], model='rerank-english-v2.0')

        relevance_scores = [i.relevance_score for i in reranked_results]
        retrieved_results["rerank_relevance_scores"] = relevance_scores

        # Filter ranked results to those above a rerank threshold
        results_df = pd.DataFrame(retrieved_results)

        rel_score_thres = np.quantile(relevance_scores, q = rel_score_threshold)

        filtered_results = results_df[results_df.rerank_relevance_scores > rel_score_thres]
        
        # Build prompt to wrap our answers and give context
        context = "\n".join(filtered_results['product_description_text'].tolist())

        # end_time = time.time()
        # section_time = end_time - start_time
        # logging.info(f"Performing Rerank took {section_time:.2f} seconds to execute")

    logging.info(f"context: {context}")

    # Use an LLM to answer question about a product
    # start_time = time.time()

    logging.info(f"Using LLM to answer user query")
    # Load LLM
    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)

    # Initialise external memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    memory.input_key="question"
    memory.output_key="answer"

    # Build prompt to wrap our answers and give context
    # template = """
    #     You are a helpful chatbot that answers questions accurately about a product. 
    #     A customer will ask questions about a product they are interested in, your role is to use the context supplied to give the customer the information they require about a product.
    #     If you don't know the answer, say that you don't know, don't try to make up an answer. 
    #     If the product does not exist, mention that you have no product that meets a customer's requirements.
    #     Keep the answer as concise as possible. Restrict your answer to information in the context.
    #     Use the following pieces of context to answer the question at the end. 
    #     {context}
    #     We also provide a conversation history to see what a customer has asked before
    #     {history}
    #     Question: {question}
    #     Helpful Answer:"""
    
    # Load Product QA Prompt Template
    dense_retrieval_product_qa_prompt = load_prompt(prompt_dir / "dense_retrieval_product_qa_prompt.json")

    query_response = generate_llm_answer(
        query=query,
        context=context,
        history=memory,
        llm=llm,
        prompt=dense_retrieval_product_qa_prompt
    )

    # end_time = time.time()
    # section_time = end_time - start_time
    # logging.info(f"Using LLM to answer user query took {section_time:.2f} seconds to execute")
    print(f"Final Answer: {query_response}")
    return query_response, context
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='User Query Parser')
    parser.add_argument('--query', help='Enter the query for the product catalogue')
    parser.add_argument('--perform_rerank', action="store_true", help='Whether to perform rerank or not')
    args = parser.parse_args()


    # Load Product Descriptions
    with open(data_dir / "product_descriptions.json", "r") as openfile:
        product_descriptions = json.load(openfile)
    

    main(query=args.query,
        perform_rerank=args.perform_rerank,
        product_descriptions=product_descriptions)
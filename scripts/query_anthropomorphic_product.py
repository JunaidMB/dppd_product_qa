import argparse
import json
import logging
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import load_prompt
from typing import Dict, List
from pathlib import Path
import pandas as pd
import os
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
prompt_dir = Path("prompt_templates")

def main(query: str, product_name: str, product_descriptions: List[Dict]) -> str:

    # Convert product descriptions into a string
    #start_time = time.time()

    logging.info(f"Converting Product Catalogue into DataFrame")
    
    # Load Product Descriptions
    product_descriptions_df = pd.DataFrame(product_descriptions)


    # end_time = time.time()
    # section_time = end_time - start_time
    # logging.info(f"Converting Product Catalogue into text took {section_time:.2f} seconds to execute")


    # Take an input query
    #start_time = time.time()
    query = query.lower()

    # Embed user query
    # start_time = time.time()

    logging.info(f"Load Prompt Template")
    # Load Anthropromorphic Product QA Prompt Template
    anthropromorphic_product_qa_prompt = load_prompt(prompt_dir / "anthropromorphic_product_qa_prompt.json")


    # end_time = time.time()
    # section_time = end_time - start_time
    # logging.info(f"Embedding User Query took {section_time:.2f} seconds to execute")


    # Create an index
    # start_time = time.time()

    logging.info(f"Retrieve Product Context")
    context = reformat_dict( product_descriptions_df[product_descriptions_df.product_name == product_name].to_dict("records")[0] )

    
    # end_time = time.time()
    # section_time = end_time - start_time
    # logging.info(f"Creating Product Index took {section_time:.2f} seconds to execute")

    # Perform Dense Retrieval to get most similar product description embeddings
    # start_time = time.time()

    logging.info(f"Performing LLM step to obtain response.")
    
    # end_time = time.time()
    # section_time = end_time - start_time
    # logging.info(f"Performing Dense Retrieval and finding corresponding text took {section_time:.2f} seconds to execute")

    # Provide input data for the prompt
    input_data = {
        "context": context,
        "question": query,
        "product_name": product_name
        }

    # Use an LLM to answer question about a product
    # Load LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.0
        )

    # Create LLMChain
    chain = LLMChain(
        llm=llm,
        prompt=anthropromorphic_product_qa_prompt,
        )

    query_response = chain.run(input_data)
    print(query_response)
    

    # end_time = time.time()
    # section_time = end_time - start_time
    # logging.info(f"Using LLM to answer user query took {section_time:.2f} seconds to execute")
    print(f"Final Answer: {query_response}")
    return query_response, context
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='User Query Parser')
    parser.add_argument('--query', help='Enter the query for the product catalogue')
    parser.add_argument('--product_name', help='Name of Product to ask question for')
    args = parser.parse_args()

    
    data_dir = Path("data/product_catalogue_data")


    # Load Product Descriptions
    with open(data_dir / "product_descriptions.json", "r") as openfile:
        product_descriptions = json.load(openfile)

    main(
        query=args.query,
        product_name=args.product_name,
        product_descriptions=product_descriptions
        )
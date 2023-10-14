import cohere
from annoy import AnnoyIndex
import numpy as np
import os
from typing import Dict, List, Type
from dotenv import load_dotenv
import pandas as pd
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv(override=True)

# Initialise Cohere Client
cohere_apikey = os.environ.get("COHERE_APIKEY")
co = cohere.Client(cohere_apikey)
x = 5

# Reformat JSON so all information is in a single string of text for each product
def reformat_dict(x: Dict) -> str: 
    return f"The product name is {x['product_name']}. The product type is {x['product_type']}. The product price in pounds is {x['product_price_in_pounds']}. {x['product_description']}"


# Dense Retrieval
def dense_retrieval(query: str, n: int, index: Type[AnnoyIndex]) -> Dict:
    
    # Dense retrieval of user query
    query_embed = co.embed(texts=[query], model="embed-english-v2.0").embeddings

    # Retrieve the nearest neighbors
    similar_item_ids = index.get_nns_by_vector(query_embed[0], n, include_distances=True)

    return {"text_id": similar_item_ids[0], "similarity_score": similar_item_ids[1]}

# Generate LLM Response
def generate_llm_answer(
    query: str, 
    context: str, 
    history: Type[ConversationBufferMemory], 
    llm: Type[ChatOpenAI], 
    prompt: Type[PromptTemplate]
    ) -> str:


    # Provide input data for the prompt
    input_data = {"context": context, "question": query, "history": history}

    # Create LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the LLMChain to obtain response
    response = chain.run(input_data)
    return response
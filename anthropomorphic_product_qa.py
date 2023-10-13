import importlib
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
from langchain.chains import LLMChain
import pandas as pd
from pprint import pprint
from semantic_search_tools.semantic_search import reformat_dict

load_dotenv(override=True)

# Load Product Descriptions
with open("product_descriptions.json", "r") as openfile:
    product_descriptions = json.load(openfile)

product_descriptions_df = pd.DataFrame(product_descriptions)

# Create a Prompt Template
template = """
    You are a helpful chatbot that answers questions accurately about a product.
    Given the name of a product, you will assume the identity of the product and give your answers in the first person.
    
    When giving your answer, obey the rules below:
    Rule 1: Restrict your answer to information in the context. Do not mention anything in your answer unless it has reference in the context.
    Rule 2: If you don't know the answer, say that you don't know, don't try to make up an answer. 

    Use the following product name to inform your identity:
    Product Name: {product_name}

    Use the following pieces of context to answer the question at the end:
    Context: {context}

    Answer the following in a step by step fashion:
    Question: {question}

    Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Specify the question and product name
product_name = "Gourmet Coffee Blend"
question = "What flavour are you?"

context = reformat_dict( product_descriptions_df[product_descriptions_df.product_name == product_name].to_dict("records")[0] )

# Provide input data for the prompt
input_data = {"context": context, "question": question, "product_name": product_name}

# Use an LLM to answer question about a product
# Load LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

# Create LLMChain
chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)

# Run the LLMChain to obtain response
response = chain.run(input_data)
print(response)


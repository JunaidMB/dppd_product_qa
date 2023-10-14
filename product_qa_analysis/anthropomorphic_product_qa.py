import json
import numpy as np
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import load_prompt
from langchain.chains import LLMChain
import pandas as pd
from pprint import pprint
from pathlib import Path
from semantic_search_tools.semantic_search import reformat_dict

load_dotenv(override=True)

data_dir = Path("data/product_catalogue_data")
prompt_dir = Path("prompt_templates")

# Load Product Descriptions
with open(data_dir/"product_descriptions.json", "r") as openfile:
    product_descriptions = json.load(openfile)

product_descriptions_df = pd.DataFrame(product_descriptions)

# Load Anthropromorphic Product QA Prompt Template
anthropromorphic_product_qa_prompt = load_prompt(prompt_dir / "anthropromorphic_product_qa_prompt.json")

# Specify the question and product name
product_name = "Gourmet Coffee Blend"
question = "What flavour are you?"

context = reformat_dict( product_descriptions_df[product_descriptions_df.product_name == product_name].to_dict("records")[0] )

# Provide input data for the prompt
input_data = {
    "context": context,
    "question": question,
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

# Run the LLMChain to obtain response
response = chain.run(input_data)
print(response)


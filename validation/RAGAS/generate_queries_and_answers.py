import json
from functools import partial
from pprint import pprint
from pathlib import Path
from scripts.query_product import main

data_dir = Path("data/product_catalogue_data")
save_dir = Path("data/qa_validation_data")

# Load Product Descriptions
with open(data_dir / "product_descriptions.json", "r") as openfile:
    product_descriptions = json.load(openfile)

# Test with example queries
queries = [
    "I just purchased a Gaming console, what other products do you have to augment my gaming needs?",
    "How much does the Modern Glass Dining Table cost?",
    "I have a sweet tooth, have you got any chocolate? If so, how much does it cost?",
    "How much does the Sony Television cost?",
    "Do you know if the Artisanal Chocolate Truffles are sourced from fair trade chocolate?"
]

# Make a partial function to make generating responses easier
generate_semantic_query_response = partial(
    main,
    perform_rerank=False,
    product_descriptions=product_descriptions
    )

# Save responses and contexts
query_responses = []
contexts = []
for idx, query in enumerate(queries):
    query_response, context = generate_semantic_query_response(query=query)
    query_responses.append(query_response)
    contexts.append(context)

# Write responses to JSON
queries_and_responses = []

for idx, _ in enumerate(query_responses):
    queries_and_responses.append( {"query": queries[idx], "response": query_responses[idx], "context": contexts[idx]} )

pprint(queries_and_responses)

with open(save_dir / "queries_and_responses.json", "w") as fp:
    json.dump(queries_and_responses, fp)




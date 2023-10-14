import json
import pandas as pd
from datasets import Dataset
from pathlib import Path
from langchain.prompts import load_prompt
from scripts.query_product import main
from ragas import evaluate

data_dir = Path("data/qa_validation_data")
save_dir = Path("data/qa_validation_data/RAGAS")

# RAG Evaluation with RAGAS
# Read Data
with open(data_dir / "queries_and_responses.json", "r") as openfile:
    queries_and_responses = json.load(openfile)

# Convert Responses to HF Dataset
queries_and_responses_df = pd.DataFrame(queries_and_responses)

## Reformat columns to be consistent with RAGAS
queries_and_responses_df = queries_and_responses_df[["query", "context", "response"]]
queries_and_responses_df.columns = ["question", "contexts", "answer"]

queries_and_responses_df["contexts"] = queries_and_responses_df['contexts'].apply(lambda x: [x])

queries_and_responses_dataset = Dataset.from_pandas(queries_and_responses_df)
queries_and_responses_dataset.reset_format()

# Compute RAGAS metrics
results = evaluate(queries_and_responses_dataset)

# Save DataFrame
prompt_dir = Path("prompt_templates")
dense_retrieval_product_qa_prompt = load_prompt(prompt_dir / "dense_retrieval_product_qa_prompt.json")

prompt_template = dense_retrieval_product_qa_prompt.template
model_name = "gpt-4"

ragas_results_df = pd.DataFrame({
    'queries_and_responses': str(queries_and_responses),
    "prompt_template": prompt_template,
    "model": model_name,
    "ragas_score": results['ragas_score'],
    'answer_relevancy': results['answer_relevancy'],
    'context_relavency': results['context_relavency'],
    'faithfulness': results['faithfulness']
    }, index = [0])

ragas_results_df.to_csv(save_dir / "ragas_results.csv")
import os
import json
import logging
import asyncio
import pandas as pd

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from azure.ai.evaluation import evaluate as azure_evaluate
from azure.identity import DefaultAzureCredential
from datasets import Dataset

from src.chain import SupplyChainAssistant

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluations")

def load_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)

async def run_ragas_evaluation():
    logger.info("Starting Ragas Evaluation...")
    dataset_path = os.path.join("data", "golden_dataset.json")
    data = load_dataset(dataset_path)
    
    assistant = SupplyChainAssistant()
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for item in data:
        q = item["question"]
        questions.append(q)
        ground_truths.append([item["ground_truth"]]) # Ragas expects list of strings
        
        # Run chain
        logger.info(f"Generating answer for: {q}")
        result = assistant.execute_query(q)
        
        answers.append(result.get("response", "No response"))
        # Context is tricky here, we can pass the raw data snippet or the reasoning
        # For this example, we assume the 'data' field from chain is the retrieved context
        contexts.append([result.get("data", "")])
        
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    hf_dataset = Dataset.from_dict(dataset_dict)
    
    results = evaluate(
        hf_dataset,
        metrics=[faithfulness, answer_relevancy],
    )
    
    logger.info(f"Ragas Results: {results}")
    return results

# def run_azure_evaluation():
#     logger.info("Starting Azure AI Evaluation (Placeholder)...")
#     # azure_evaluate(...)
#     pass

if __name__ == "__main__":
    # Running sync wrapper for async Ragas
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_ragas_evaluation())

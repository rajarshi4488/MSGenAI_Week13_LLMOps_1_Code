# RAG Evaluations

This directory contains the scripts and configurations for evaluating the Supply Chain RAG pipeline.

## Overview
We use **Ragas** (Retrieval Augmented Generation Assessment) to measure the quality of our RAG system.

## Metrics Used
- **Faithfulness**: Measures how factually accurate the generated answer is given the retrieved context.
- **Answer Relevancy**: Measures how relevant the answer is to the user's question.

## Files
- **`eval_script.py`**: The main script that:
    1. Loads the `golden_dataset.json` from the `data/` directory.
    2. Runs the key queries through the `SupplyChainAssistant`.
    3. Computes the Ragas metrics.
    4. Logs the results.

## How to Run
Ensure your `.env` file is configured with OpenAI API keys.

```bash
python evaluations/eval_script.py
```

# Prompts Configuration

This directory manages the prompt templates using the **Prompty** format. Prompty allows for versioning, metadata management, and model configuration directly alongside the prompt text.

## Prompts
- **`intent_classification.prompty`**: Determines if a user query is valid and whether it requires Python or SQL analysis. Uses `gpt-5-mini`.
- **`code_generation.prompty`**: Generates Python (Pandas) code for data analysis. Uses `gpt-5-mini`.
- **`sql_generation.prompty`**: Generates DuckDB-compatible SQL queries. Uses `gpt-5-mini`.
- **`code_evaluation.prompty`**: The **LLM Judge**. Evaluates generated code for safety and correctness. Uses `gpt-4.2`.
- **`response_synthesis.prompty`**: Formulates the final natural language answer based on the execution result. Uses `gpt-5-mini`.

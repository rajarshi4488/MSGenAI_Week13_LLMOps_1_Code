# Source Code

This directory contains the core application logic.

## Components
- **`app.py`**: The Streamlit user interface entry point. Handles chat session state and user interaction.
- **`chain.py`**: The main orchestration logic.
    - Implements `SupplyChainAssistant`.
    - Manages the chain execution flow: Intent -> Code Gen -> Judge -> Execution -> Response.
    - Handles retry logic if the Judge rejects the code.
- **`ingestion.py`**: Handles loading the CSV data and basic preprocessing (column cleaning, type conversion).
- **`sql_tools.py`**: Provides the `SQLTools` class for executing queries on the dataframe using DuckDB.
- **`vector_store.py`**: Abstract interface for vector storage, supporting `ChromaVectorStore` and `AzureVectorStore`.
- **`config.py`**: Manages environment variables and sets up OpenTelemetry (logging/tracing).



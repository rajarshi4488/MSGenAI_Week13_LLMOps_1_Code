# Supply Chain LLMOps Case Study 🚛📦

An AI-driven Natural Language Interface for querying supply chain data, demonstrating advanced LLMOps practices including RAG, Code Generation (Python/SQL), LLM-as-a-judge evaluation, and Infra-as-Code.

## 🌟 Key Features
- **Natural Language Querying**: Ask questions about orders, risk, and financials in plain English.
- **Dual-Mode Generation**: Automatically selects between **Python** (Pandas) for complex analysis and **SQL** (DuckDB) for efficient retrieval.

## 💡 Sample Queries
Try these queries to explore the dataset:
- "What is the total sales?"
- "Which shipping mode has the highest average profit?"
- "List the top 5 product categories by sales."
- "Show a line chart showing the revenue/ sales over the time periods."
- "Show a bar chart showing the total sales by market."
- "Show a pie chart showing the total sales by product category."

**Safety First**: Implements an **LLM Judge** (`gpt-4.1`) to validate generated code before execution.

**Evaluation**: Integrated RAG pipeline evaluation using **Ragas** and Azure AI Evaluation.

**Vector Store**: Flexible integration with **ChromaDB** (local) and **Azure AI Search** (production).

## 📂 Project Structure
- `src/`: Source code including the **FastAPI** backend (`api.py`) and **Streamlit** client (`app.py`).
- `prompts/`: Managed prompts (`.prompty` files) for all LLM interactions.
- `data/`: Real-world datasets (`DataCoSupplyChainDataset.csv`), domain knowledge, and evaluation triples.
- `tests/`: Unit and integration tests for API and core logic.
- `evaluations/`: Ragas evaluation scripts.
- `infra/`: Azure Bicep templates for cloud deployment.

### Create Python Virtual Environment
Ensure you've created a python virtual environment 
```bash
python -m venv venv
```
### Activate Python virtual environment
For Mac/ Linux enter the following command:
```bash
source venv/bin/activate
```
For Windows systems, enter the following command:
```bash
venv/Scripts/activate
```
### Install Libraries
```bash
pip install -r requirements.txt
```

### Running the App
Ensure your `.env` file is configured with the necessary API keys.
```bash
streamlit run src/app.py
```
The app will be available at `http://localhost:8501`.

### Running Evaluations

To evaluate the RAG pipeline using Ragas:
```bash
python evaluations/eval_script.py
```

### Code Quality & Linting

This project uses modern Python code quality tools configured in `pyproject.toml`:

#### Code Formatting
**Black** - Automatic code formatting to PEP 8 compliance:
```bash
# Check formatting
black --check src tests

# Auto-format code
black src tests
```

#### Linting
**Ruff** - Fast, comprehensive Python linter (includes Flake8, isort, pyupgrade):
```bash
# Run linting
ruff check src tests

# Auto-fix issues
ruff check --fix src tests
```


**Note**: All tools are configured in `pyproject.toml` with:
- Line length: 100 characters
- Target Python version: 3.11
- Import sorting enabled (isort via Ruff)

## ☁️ Deployment

This project includes a Dockerfile that can be used for deplpoyment including Azure App Services.

### Resources Created
- **Azure App Service Plan** (Linux, B1 Tier)
- **Azure App Service** (Python 3.10)

## 🛠️ Models Used
- **GPT-5 Mini**: Intent classification, code generation, response synthesis.
- **GPT-4.1**: LLM Judge for code validation.
- **Text-Embedding-3-Small**: RAG embeddings.
"""
Supply Chain RAG Pipeline - Main Chain Logic.

This module implements the core RAG (Retrieval Augmented Generation) pipeline
that processes user queries about supply chain data. The pipeline includes:

1. Intent Classification - Determines if query needs Python or SQL
2. Code Generation - Generates Python (Pandas) or SQL (DuckDB) code
3. LLM Judge - Validates generated code for safety and correctness
4. Execution - Runs the validated code on supply chain data
5. Retrieval - Fetches relevant context from vector store
6. Response Synthesis - Combines results and context into natural language

Prompts are loaded via the Prompty templates for all LLM interactionss.
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, Any, Optional

from langchain_core.runnables import RunnablePassthrough
from langchain_prompty import create_chat_prompt
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_experimental.utilities.python import PythonREPL
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import Config, logger, tracer
from src.ingestion import DataIngestion
from src.sql_tools import SQLTools


def get_llm(temperature=1.0, overrides: Dict[str, Any] = None):
    """
    Initialize and return the main LLM instance.

    Creates an Azure OpenAI or OpenAI chat model for general tasks like
    intent classification, code generation, and response synthesis.

    Args:
        temperature (float): Sampling temperature (0.0-2.0). Higher = more random.
                           Default 1.0 as required by gpt-5-mini.
        overrides (Dict[str, Any], optional): Override config values from UI.
                                              Keys: AZURE_OPENAI_API_KEY,
                                              AZURE_OPENAI_ENDPOINT, etc.

    Returns:
        AzureChatOpenAI | ChatOpenAI: Configured LLM instance
    """
    overrides = overrides or {}
    api_key = overrides.get("AZURE_OPENAI_API_KEY") or Config.AZURE_OPENAI_API_KEY
    endpoint = overrides.get("AZURE_OPENAI_ENDPOINT") or Config.AZURE_OPENAI_ENDPOINT
    api_version = overrides.get("AZURE_OPENAI_API_VERSION") or Config.AZURE_OPENAI_API_VERSION
    deployment = overrides.get("AZURE_DEPLOYMENT") or os.getenv("AZURE_DEPLOYMENT", "gpt-5-mini")

    if api_key:
        return AzureChatOpenAI(
            azure_deployment=deployment,
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
            temperature=temperature,
        )
    else:
        return ChatOpenAI(temperature=temperature)


def get_judge_llm(temperature=0.0, overrides: Dict[str, Any] = None):
    """
    Initialize and return the LLM Judge instance.

    Creates a dedicated Azure OpenAI model (typically GPT-4.1) for evaluating
    generated code. Uses temperature=0.0 for deterministic, conservative evaluation.

    Args:
        temperature (float): Sampling temperature. Default 0.0 for consistency.
        overrides (Dict[str, Any], optional): Override config values from UI.

    Returns:
        AzureChatOpenAI | ChatOpenAI: Configured Judge LLM instance
    """
    overrides = overrides or {}
    api_key = overrides.get("AZURE_OPENAI_API_KEY") or Config.AZURE_OPENAI_API_KEY
    endpoint = overrides.get("AZURE_OPENAI_ENDPOINT") or Config.AZURE_OPENAI_ENDPOINT
    api_version = overrides.get("AZURE_OPENAI_API_VERSION") or Config.AZURE_OPENAI_API_VERSION
    deployment = overrides.get("AZURE_JUDGE_DEPLOYMENT") or Config.AZURE_JUDGE_DEPLOYMENT

    if api_key:
        return AzureChatOpenAI(
            azure_deployment=deployment,
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
            temperature=temperature,
        )
    else:
        return ChatOpenAI(temperature=temperature)


class SupplyChainAssistant:
    """
    Main orchestrator for Supply Chain data analysis queries.

    This class coordinates the entire RAG pipeline from query to response.
    It manages:
    - Data loading and schema extraction
    - LLM initialization (main + judge)
    - Code execution tools (Python REPL, SQL engine)
    - LangChain chains for each pipeline stage
    - Vector store retrieval for domain knowledge

    Attributes:
        ingestion (DataIngestion): Handles CSV loading and vector indexing
        df (pd.DataFrame): Loaded supply chain data
        schema_info (str): String representation of DataFrame schema for LLM context
        llm (AzureChatOpenAI): Main LLM for general tasks
        judge_llm (AzureChatOpenAI): Dedicated LLM for code validation
        python_repl (PythonREPL): Python execution environment
        sql_tools (SQLTools): DuckDB query execution wrapper
        *_chain: LangChain pipelines for intent, generation, evaluation, synthesis
    """

    def __init__(self, overrides: Dict[str, Any] = None):
        """
        Initialize the Supply Chain Assistant.

        Sets up data loading, LLM clients, execution environments, and
        all LangChain chains from Prompty templates.

        Args:
            overrides (Dict[str, Any], optional): Configuration overrides from UI sidebar.
                                                  Allows dynamic model/endpoint changes.
        """
        self.ingestion = DataIngestion(os.path.join("data", "DataCoSupplyChainDataset.csv"))
        self.df = self.ingestion.load_data()
        self.schema_info = self.ingestion.get_schema_info()
        self.overrides = overrides or {}

        # Tools
        self.python_repl = PythonREPL()
        self.python_repl.globals["df"] = self.df
        self.python_repl.globals["pd"] = pd

        self.sql_tools = SQLTools(self.df)

        self.llm = get_llm(overrides=self.overrides)
        self.judge_llm = get_judge_llm(overrides=self.overrides)
        self._setup_chains()

    def _setup_chains(self):
        # Helper for absolute paths
        def get_prompt_path(filename):
            return os.path.join(os.getcwd(), "prompts", filename)

        # 1. Intent & Routing
        self.intent_prompt = create_chat_prompt(get_prompt_path("intent_classification.prompty"))
        self.intent_chain = self.intent_prompt | self.llm | JsonOutputParser()

        # 2. Code Generation (Python)
        self.py_gen_prompt = create_chat_prompt(get_prompt_path("code_generation.prompty"))
        self.py_gen_chain = self.py_gen_prompt | self.llm | StrOutputParser()

        # 3. SQL Generation
        self.sql_gen_prompt = create_chat_prompt(get_prompt_path("sql_generation.prompty"))
        self.sql_gen_chain = self.sql_gen_prompt | self.llm | StrOutputParser()

        # 4. Code Evaluation (Judge)
        self.eval_prompt = create_chat_prompt(get_prompt_path("code_evaluation.prompty"))
        self.eval_chain = self.eval_prompt | self.judge_llm | JsonOutputParser()

        # 5. Response Synthesis
        self.response_prompt = create_chat_prompt(get_prompt_path("response_synthesis.prompty"))
        self.response_chain = self.response_prompt | self.llm | StrOutputParser()

    def execute_query(self, user_query: str) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a user query.

        This is the main entry point that orchestrates the entire pipeline:
        1. Intent Classification - Route to Python or SQL
        2. Code Generation - Generate appropriate code
        3. LLM Judge Evaluation - Validate code safety and correctness
        4. Execution - Run approved code (with retry on rejection)
        5. Vector Retrieval - Fetch domain knowledge context
        6. Response Synthesis - Generate natural language answer

        Args:
            user_query (str): Natural language question about supply chain data.
                            E.g., "What are total sales?" or "Show top products"

        Returns:
            Dict[str, Any]: Response dictionary with keys:
                - response (str): Natural language answer
                - code (str, optional): Generated code that was executed
                - data (str, optional): Raw execution results (truncated)
                - image (matplotlib.figure.Figure, optional): Plot if generated

        Example:
            >>> assistant = SupplyChainAssistant()
            >>> result = assistant.execute_query("Plot sales by category")
            >>> print(result["response"])
            >>> if "image" in result:
            ...     plt.show()
        """
        logger.info(f"Processing query: {user_query}")

        # Step 1: Intent
        try:
            intent_res = self.intent_chain.invoke({"user_query": user_query})
            if not intent_res.get("is_valid", False):
                return {"response": intent_res.get("message", "Invalid query.")}

            language = intent_res.get("language", "python").lower()
            logger.info(f"Selected Language: {language}")
        except Exception as e:
            logger.error(f"Intent Error: {e}")
            return {"response": "Error understanding query."}

        # Step 2: Generation Loop
        max_retries = 1
        code = ""
        for attempt in range(max_retries + 1):
            try:
                if language == "python":
                    code = self.py_gen_chain.invoke(
                        {"user_query": user_query, "schema_info": self.schema_info}
                    )
                    code = code.replace("```python", "").replace("```", "").strip()
                else:  # sql
                    code = self.sql_gen_chain.invoke(
                        {"user_query": user_query, "schema_info": self.schema_info}
                    )
                    code = code.replace("```sql", "").replace("```", "").strip()

                # Step 3: Judge Evaluation
                eval_res = self.eval_chain.invoke(
                    {"user_query": user_query, "generated_code": code, "language": language}
                )

                if eval_res.get("is_safe") and eval_res.get("is_correct"):
                    logger.info("Code approved by Judge.")
                    break  # Safe to proceed
                else:
                    logger.warning(f"Judge rejected code: {eval_res.get('feedback')}")
                    if attempt < max_retries:
                        # Retry with feedback (simplification: just re-generate, usually prompt would take feedback)
                        # Ideally we pass feedback back to generator, but for this simpler loop we just retry
                        logger.info("Retrying generation...")
                        continue
                    else:
                        return {
                            "response": f"I could not generate safe code. Reason: {eval_res.get('feedback')}"
                        }

            except Exception as e:
                logger.error(f"Gen/Eval Error: {e}")
                return {"response": "Error generating or evaluating code."}

        # Step 4: Execution
        result = None
        try:
            if language == "python":
                # Use exec to capture local variables
                # Prepare execution context
                local_vars = {}
                global_vars = {"pd": pd, "df": self.df, "plt": plt, "sns": sns}

                # Execute safely (assuming code is trusted/checked by Judge)
                try:
                    exec(code, global_vars, local_vars)

                    # Extract result
                    result_obj = local_vars.get("result", None)
                    result = (
                        str(result_obj) if result_obj is not None else "No result variable found."
                    )

                    # Extract figure
                    fig = local_vars.get("fig", None)
                    if fig:
                        logger.info("Plot generated.")
                        return {
                            "response": "Plot generated successfully.",
                            "code": code,
                            "data": result,
                            "image": fig,
                        }

                except Exception as exec_err:
                    raise exec_err
            else:
                # SQL
                df_res = self.sql_tools.execute_query(code)
                result = df_res.to_string()

            logger.info(f"Execution Result: {result}")
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return {"response": f"Error executing code: {str(e)}", "code": code}

        # Step 4.5: Retrieval (RAG)
        context_text = ""
        try:
            # Retrieve relevant docs from vector store
            docs = self.ingestion.vector_store.similarity_search(user_query, k=2)
            context_text = "\n".join([d.page_content for d in docs])
            logger.info(f"Retrieved Context: {context_text}")
        except Exception as e:
            logger.warning(f"Retrieval failed or skipped: {e}")

        # Step 5: Response Synthesis
        try:
            final_response = self.response_chain.invoke(
                {
                    "user_query": user_query,
                    "data_result": str(result)[:2000],  # Truncate massive results
                    "context": context_text,
                }
            )
            return {"response": final_response, "code": code, "data": str(result)[:500] + "..."}
        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            return {"response": "Error synthesizing answer.", "code": code}


if __name__ == "__main__":
    assistant = SupplyChainAssistant()
    print(assistant.execute_query("What are the total sales?"))

import duckdb
import pandas as pd
import logging

logger = logging.getLogger("supply_chain_llmops")


class SQLTools:
    def __init__(self, df: pd.DataFrame):
        self.con = duckdb.connect(database=":memory:")
        self.con.register("supply_chain", df)

    def execute_query(self, query: str):
        """Executes a SQL query on the registered dataframe."""
        try:
            logger.info(f"Executing SQL: {query}")
            # Ensure query operates on 'supply_chain' table
            if "supply_chain" not in query.lower():
                # Simple heuristic: try to replace table reference if needed, or fail.
                # Ideally prompt handles this, but safety net:
                pass

            result_df = self.con.execute(query).df()
            return result_df
        except Exception as e:
            logger.error(f"SQL Execution Error: {e}")
            raise e

    def get_schema(self):
        return self.con.execute("DESCRIBE supply_chain").df().to_string()

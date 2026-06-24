import pandas as pd
import os
import logging

logger = logging.getLogger("supply_chain_llmops")


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.vector_store import get_vector_store


class DataIngestion:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.vector_store = get_vector_store()
        self.doc_path = os.path.join(os.path.dirname(file_path), "domain_knowledge.txt")

    def load_data(self):
        """Loads data from the CSV file and indexes unstructured text."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            # DataCo dataset often contains characters requiring latin-1 encoding
            self.df = pd.read_csv(self.file_path, encoding='latin-1')
            logger.info(f"Loaded data from {self.file_path} with shape {self.df.shape}")
            self._preprocess()

            # Index domain knowledge (Basic check to avoid re-indexing every time in dev)
            self._index_documents()

            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _index_documents(self):
        if not os.path.exists(self.doc_path):
            logger.warning(f"Domain knowledge file not found at {self.doc_path}")
            return

        try:
            with open(self.doc_path, "r", encoding='utf-8') as f:
                text = f.read()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.create_documents([text])
            self.vector_store.add_documents(docs)
            logger.info("Indexed domain knowledge.")
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")

    def _preprocess(self):
        """Basic preprocessing steps."""
        # Convert column names to a cleaner format (snake_case)
        # Specifically handling 'order date (DateOrders)' which is common in this dataset
        self.df.columns = [
            c.strip().replace(" ", "_").replace("(", "").replace(")", "").lower() 
            for c in self.df.columns
        ]

        # Ensure date columns are datetime
        # After the above transformation, 'order date (DateOrders)' becomes 'order_date_dateorders'
        date_cols = ['order_date_dateorders', 'shipping_date_dateorders']
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])
        
        # Also create a generic 'order_date' for easier prompting if needed
        if 'order_date_dateorders' in self.df.columns:
            self.df['order_date'] = self.df['order_date_dateorders']

        logger.info("Data preprocessing complete.")

    def get_schema_info(self):
        """Returns string representation of schema for LLM context."""
        if self.df is None:
            return "Data not loaded."

        buffer = []
        buffer.append("Columns and Data Types:")
        for col, dtype in self.df.dtypes.items():
            buffer.append(f"- {col} ({dtype})")

        buffer.append("\nSample Data (first 3 rows):")
        buffer.append(self.df.head(3).to_string(index=False))

        return "\n".join(buffer)


if __name__ == "__main__":
    # Test ingestion
    ingestion = DataIngestion(os.path.join("data", "DataCoSupplyChainDataset.csv"))
    df = ingestion.load_data()
    print(ingestion.get_schema_info())

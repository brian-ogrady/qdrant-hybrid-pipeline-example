"""
Hybrid search pipeline example using Qdrant vector database with data from a CSV file.

This script demonstrates how to set up a hybrid search pipeline using a YAML configuration
and ingest data from a specified CSV file.
"""

import argparse
import math
import os
import time
import uuid
import pandas as pd

from hybrid_search import HybridPipeline
from hybrid_search.config_yaml_loader import create_hybrid_pipeline_from_yaml
from qdrant_client import QdrantClient
from tqdm import tqdm

def _setup_qdrant() -> QdrantClient:
    """
    Initialize and configure a Qdrant client instance.
    """
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    client = QdrantClient(host=host, port=port, timeout=180)
    return client

def _load_data_from_csv(file_path: str, text_column: str, batch_size: int):
    """
    Loads data from a CSV file and yields it in batches.

    Args:
        file_path (str): The path to the CSV file.
        text_column (str): The name of the column containing the text to be indexed.
        batch_size (int): The number of records per batch.

    Yields:
        tuple: Each yield contains:
            - list of document texts
            - list of document UUIDs
            - list of payload dictionaries
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} records from {file_path}")

        for i in tqdm(range(0, len(df), batch_size), desc="Batching CSV data"):
            batch_df = df.iloc[i:i+batch_size]
            
            documents = batch_df[text_column].tolist()
            # The rest of the columns will be the payload
            payloads = batch_df.to_dict(orient='records')
            document_ids = [uuid.uuid4() for _ in range(len(batch_df))]

            yield documents, document_ids, payloads

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return
    except KeyError:
        print(f"Error: The CSV must contain a column named '{text_column}'.")
        return

def _insert_batches(
    pipeline: HybridPipeline,
    file_path: str,
    text_column: str,
    batch_size: int,
):
    """
    Insert batches of data from a CSV into the Qdrant collection.
    """
    df = pd.read_csv(file_path)
    num_records = len(df)
    total_batches = math.ceil(num_records / batch_size)
    
    data_generator = _load_data_from_csv(file_path, text_column, batch_size)
    
    with tqdm(total=total_batches, desc="Inserting batches") as pbar:
        for documents, document_ids, payloads in data_generator:
            try:
                start_time = time.time()
                pipeline.insert_documents(
                    documents=documents,
                    document_ids=document_ids,
                    payloads=payloads,
                    batch_size=batch_size,
                )
                end_time = time.time()
                print(f"Insertion time for batch: {end_time - start_time:.2f} seconds")
                pbar.update(1)
            except Exception as e:
                print(f"Error inserting batch: {e}")

def main():
    """
    Main entry point for the hybrid search pipeline example.
    
    Parses command line arguments, sets up the Qdrant client and hybrid pipeline
    using a YAML config, ingests data from a CSV, and performs a sample query.
    """
    parser = argparse.ArgumentParser(description="Ingest CSV data into Qdrant using a YAML config.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file for the pipeline."
    )
    parser.add_argument(
        "--csv-path", 
        type=str, 
        required=True, 
        help="Path to the CSV file containing the data to ingest."
    )
    parser.add_argument(
        "--text-column", 
        type=str, 
        required=True, 
        help="The name of the column in the CSV that contains the text to be indexed."
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=128, 
        help="Batch size for inserting documents."
    )
    parser.add_argument(
        "--query", 
        type=str, 
        default="What is the capital of France?", 
        help="A sample query to run after ingestion."
    )
    args = parser.parse_args()

    qdrant = _setup_qdrant()

    # --- Corrected Section ---
    # 1. Create the config object using the dedicated factory function
    pipeline_config = create_hybrid_pipeline_from_yaml(args.config)

    # 2. Initialize the pipeline with the created config object
    pipeline = HybridPipeline(
        qdrant_client=qdrant,
        collection_name="data_from_csv",
        hybrid_pipeline_config=pipeline_config,
    )

    _insert_batches(
        pipeline, 
        args.csv_path, 
        args.text_column, 
        args.batch_size
    )

    print(f"\nRunning sample query: '{args.query}'")
    results = pipeline.search(query=args.query, top_k=5)
    
    print("\n--- Search Results ---")
    if not results:
        print("No results found.")
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Score: {result.score:.4f})")
        # Print the original text from the payload
        print(f"  Text: {result.payload.get(args.text_column, 'N/A')}")
        # Print other relevant payload data
        other_payload = {k: v for k, v in result.payload.items() if k != args.text_column and k != "document_id"}
        print(f"  Metadata: {other_payload}")


if __name__ == "__main__":
    main()
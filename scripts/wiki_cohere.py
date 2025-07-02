"""
Hybrid search pipeline example using Qdrant vector database with Wikipedia dataset.

This script demonstrates how to set up a hybrid search pipeline using Qdrant with multiple embedding models:
- Dense embeddings (BAAI/bge-small-en-v1.5)
- Sparse embeddings (Qdrant/bm25)
- Late-interaction embeddings (answerdotai/answerai-colbert-small-v1)

The script loads Wikipedia data and implements multi-tenant vector search with partition filtering.
"""

import argparse
import math
import os
import random
import time
import uuid

from datasets import load_dataset
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from hybrid_search import HybridPipeline, HybridPipelineConfig, SentenceTransformerEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Distance,
    HnswConfigDiff,
    VectorParams,
    SparseVectorParams,
    KeywordIndexParams,
    MultiVectorConfig,
    MultiVectorComparator,
)
from tqdm import tqdm


def _setup_qdrant() -> QdrantClient:
    """
    Initialize and configure a Qdrant client instance.
    
    Uses environment variables to determine host and port, with defaults if not specified:
    - QDRANT_HOST: Defaults to 'localhost'
    - QDRANT_PORT: Defaults to '6333'
    
    Returns:
        QdrantClient: Configured Qdrant client with a 180-second timeout
    """
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    client = QdrantClient(host=host, port=port, timeout=180)
    return client


def _configure_pipeline() -> HybridPipelineConfig:
    """
    Configure a hybrid search pipeline with multiple embedding models and parameters.
    
    Sets up three different embedding approaches:
    1. Dense embeddings using BAAI/bge-small-en-v1.5
    2. Sparse embeddings using Qdrant/bm25
    3. Late-interaction embeddings using answerdotai/answerai-colbert-small-v1
    
    Also configures multi-tenant support with tenant_id partitioning.
    
    Returns:
        HybridPipelineConfig: Complete configuration for hybrid search pipeline
    """
    text_model = SentenceTransformerEmbedding("intfloat/multilingual-e5-large", device="mps")
    sparse_model = SparseTextEmbedding("Qdrant/bm25")
    late_model = LateInteractionTextEmbedding("answerdotai/answerai-colbert-small-v1")
            
    dense_params = VectorParams(
        size=1024,
        distance=Distance.COSINE,
        on_disk=True,
        quantization_config=BinaryQuantization(
            binary=BinaryQuantizationConfig(
                always_ram=True
            )
        )
    )

    sparse_params = SparseVectorParams()

    late_params = VectorParams(
        size=96, 
        distance=Distance.COSINE,
        on_disk=True,
        multivector_config=MultiVectorConfig(
            comparator=MultiVectorComparator.MAX_SIM
        ),
        hnsw_config=HnswConfigDiff(
            m=0,
        )
    )
            
    partition_field = "tenant_id"
    partition_index = KeywordIndexParams(
        type="keyword",
        is_tenant=True,
        on_disk=True,
    )
            
    return HybridPipelineConfig(
        text_embedding_config=(text_model, dense_params),
        sparse_embedding_config=(sparse_model, sparse_params),
        multi_tenant=False,
        replication_factor=2,
        shard_number=3,
    )


def _query(pipeline: HybridPipeline, query: str, num_results: int = 10):
    """
    Perform a search query against the hybrid pipeline.
    
    Executes a search with the given query text, filtering by a specific tenant.
    
    Args:
        pipeline (HybridPipeline): The configured hybrid search pipeline
        query (str): The text query to search for
        num_results (int, optional): Maximum number of results to return. Defaults to 10.
        
    Returns:
        list: Search results from tenant_id_0 partition
    """
    results_a = pipeline.search(
            query=query, 
            top_k=num_results,
        )
    return results_a


def _prepare_batches(
    num_records: int,
    num_tenants: int,
    batch_size: int = 100,
):
    """
    Prepare batches of Wikipedia data with assigned tenant IDs for insertion.
    
    Loads data from Cohere's Wikipedia dataset, assigns randomly weighted tenant IDs,
    and yields batches of data ready for insertion into Qdrant.
    
    Args:
        num_records (int): Total number of records to process from Wikipedia
        num_tenants (int): Number of different tenant IDs to distribute data across
        batch_size (int, optional): Number of records per batch. Defaults to 100.
        
    Yields:
        tuple: Each yield contains:
            - list of document texts
            - list of document UUIDs
            - list of payload dictionaries containing tenant_id
    """
    dataset = load_dataset(
        "Cohere/wikipedia-2023-11-embed-multilingual-v3",
        "en",
        split="train",
        streaming=True,
    ).take(num_records)

    tenant_ids = [f"tenant_id_{i}" for i in range(num_tenants)]
    weights = [random.random() for _ in range(num_tenants)]
    probabilities = [w / sum(weights) for w in weights]

    batch = []
    record_count = 0

    for record in dataset:
        batch.append(record["text"])
        record_count += 1
        
        if len(batch) == batch_size or record_count == num_records:
            current_batch_size = len(batch)
            document_ids = [uuid.uuid4() for _ in range(current_batch_size)]
            tenants = random.choices(tenant_ids, weights=probabilities, k=current_batch_size)
            payloads = [{"tenant_id": tenant_id} for tenant_id in tenants]
            yield batch, document_ids, payloads
            
            batch = []

            if record_count >= num_records:
                break
            
    if batch:
        current_batch_size = len(batch)
        document_ids = [uuid.uuid4() for _ in range(current_batch_size)]
        tenants = random.choices(tenant_ids, weights=probabilities, k=current_batch_size)
        payloads = [{"tenant_id": tenant_id} for tenant_id in tenants]
        yield batch, document_ids, payloads


def _insert_batches(
    pipeline: HybridPipeline,
    num_records: int,
    num_tenants: int,
    batch_size: int = 100,
):
    """
    Insert batches of Wikipedia data into the Qdrant collection via the hybrid pipeline.
    
    Processes data in batches with a progress bar, timing each insertion operation.
    Handles exceptions that might occur during insertion.
    
    Args:
        pipeline (HybridPipeline): The configured hybrid search pipeline
        num_records (int): Total number of records to insert
        num_tenants (int): Number of different tenant IDs to distribute data across
        batch_size (int, optional): Number of records per batch. Defaults to 100.
    """
    total_batches = math.ceil(num_records / batch_size)
    
    with tqdm(total=total_batches, desc="Inserting batches") as pbar:
        for batch, document_ids, payloads in _prepare_batches(num_records, num_tenants, batch_size):
            try:
                start_time = time.time()
                pipeline.insert_documents(
                    documents=batch,
                    document_ids=document_ids,
                    payloads=payloads,
                    batch_size=batch_size,
                )
                end_time = time.time()
                print(f"Inserting time: {end_time - start_time} seconds")
                pbar.update(1)
            except Exception as e:
                print(f"Error inserting batch: {e}")


def main():
    """
    Main entry point for the Wikipedia hybrid search pipeline example.
    
    Parses command line arguments, sets up the Qdrant client and hybrid pipeline,
    loads and inserts Wikipedia data, and performs a sample query.
    
    Command line arguments:
        --num-records: Number of Wikipedia records to insert (default: 10000)
        --num-tenants: Number of tenant IDs to distribute data across (default: 10)
        --batch-size: Number of records per insertion batch (default: 2000)
    """
    parser = argparse.ArgumentParser(description="Insert Wikipedia dataset into Qdrant")
    parser.add_argument("--num-records", type=int, default=1000000, help="Number of records to insert")
    parser.add_argument("--num-tenants", type=int, default=10, help="Number of tenants to insert")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for inserting documents")
    args = parser.parse_args()

    qdrant = _setup_qdrant()
    pipeline_config = _configure_pipeline()
    pipeline = HybridPipeline(
        qdrant_client=qdrant,
        collection_name="wikipedia",
        hybrid_pipeline_config=pipeline_config,
    )
    _insert_batches(pipeline, args.num_records, args.num_tenants, args.batch_size)

    _query(pipeline, "What is the capital of France?")


if __name__ == "__main__":
    main()

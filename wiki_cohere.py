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
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    client = QdrantClient(host=host, port=port, timeout=180)
    return client


def _configure_pipeline() -> HybridPipelineConfig:
    text_model = SentenceTransformerEmbedding("BAAI/bge-small-en-v1.5", device="mps")
    sparse_model = SparseTextEmbedding("Qdrant/bm25")
    late_model = LateInteractionTextEmbedding("answerdotai/answerai-colbert-small-v1")
            
    dense_params = VectorParams(
        size=384,
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
        late_interaction_text_embedding_config=(late_model, late_params),
        partition_config=(partition_field, partition_index),
        multi_tenant=True,
        replication_factor=2,
        shard_number=3,
    )


def _prepare_batches(
    num_records: int,
    num_tenants: int,
    batch_size: int = 100,
):
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
    parser = argparse.ArgumentParser(description="Insert Wikipedia dataset into Qdrant")
    parser.add_argument("--num-records", type=int, default=10000, help="Number of records to insert")
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


if __name__ == "__main__":
    main()

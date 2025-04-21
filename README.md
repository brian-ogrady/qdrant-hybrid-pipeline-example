# qdrant-hybrid-pipeline-example
Example implementation of Hybrid Search Pipeline in Qdrant

This example will spin up a Qdrant instance with two nodes. The details on the collection create can be found in the wiki_cohere.py file.

### Hybrid Search Pipeline

We will leverage a hybrid vector search pipeline w/ the following configuration:

- Dense embeddings using `BAAI/bge-small-en-v1.5`
  - Binary Quantization
- Sparse embeddings using `Qdrant/bm25`
- ColBERT embeddings using `answerdotai/answerai-colbert-small-v1`
  - Config specifies to not create the HNSW index
  - Used only for reranking
- Multitenancy
  - We will specify multitenancy using a payload index
- Three shards
- Data is replicated across the nodes

All embeddings will be generated using `fastembed` and `sentence-transformers`. We will also be leveraging the Python package I created, `qdrant-hybrid-pipeline`, which is imported via `hybrid_search` (I need to change that name).


## Implementation Notes & Performance Considerations

* **Functionality:**
  * The pipeline successfully implements the required hybrid search (dense, sparse, late-interaction), leverages Qdrant's query API, handles multi-tenancy via partitioning, uses Binary Quantization for dense vectors, and is demonstrated with a 1M record dataset setup via Docker Compose with the specified cluster configuration (2 nodes, 3 shards, replication=2).
* **Performance:** During testing on Apple Silicon (MPS), a performance bottleneck was observed during the ingestion phase, primarily related to the late-interaction embeddings (ColBERT via `fastembed`). 
    * Initial tests with `fastembed` for dense embeddings also showed slower performance on MPS compared to `sentence-transformers`, so the latter was used for the dense model in the final example (`wiki_cohere.py`).
    * The `Qdrant/bm25` sparse model was chosen for its speed, adding minimal latency compared to alternatives like SPLADE which proved significantly slower.
    * The ColBERT model inference via `fastembed` remains the main contributor to the ingestion time in the provided example (`wiki_cohere.py`). Investigation into alternative ColBERT libraries (`pylate`, `ragatouille`, `colbert-ai`) revealed dependency or maintenance issues.
* **Focus:** The priority for this assignment was placed on demonstrating the correct architectural implementation and usage of Qdrant's features as specified, using readily available and functional library components. Further performance optimization for specific hardware (like MPS) would be a subsequent step in a production scenario.

## Potential Improvements
* **Async HybridPipeline:**
  * This would help with large insertions like the 1 million example included here.
* **Client Side Retry Logic:**
  * The current implementation doesn't retry if insertions fail.
* **Error Handling:**
  * Not much is done to handle errors from insertions, reads, etc, this is left the to Qdrant Client. Future versions would implement more comprehensive logic.

### Execution

`chmod +x run_pipeline.sh`
`./run_pipeline.sh`

You can edit the shell script to determine the number of records to insert. Default is 10,000. 

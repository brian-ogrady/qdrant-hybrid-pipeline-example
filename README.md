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

### Execution

`chmod +x run_pipeline.sh`
`./run_pipeline.sh`

You can edit the shell script to determine the number of records to insert. Default is 10,000. 

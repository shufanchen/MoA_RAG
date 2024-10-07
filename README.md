# MoA_RAG
MoA_Rag is a framework that combines a mixture of agents (MoA) with Retrieval-Augmented Generation (RAG).

## Folder Structure
- **dataset**: Contains raw data files, including JSONL-formatted QA pairs and index files for each dataset.
- **model**: Stores the embedding models used for retrieval.
- **results**: Stores the outputs generated by MoA_Rag as well as the corresponding evaluation metric results.
- **retrieval_results**: Stores retrieval results and evaluation metrics for each individual retriever.

## Python Files
- **MoA_Rag.py**: 
    - Supports setting the number of layers (2 or 3).
    - Allows adjusting the temperature for each layer.
    - Configures the number and type of models for each layer.
    - Sets different retrievers (embeddings) for models.
    - Customizes prompts for each layer.
- **make_index.py**: 
    - Builds the knowledge vector database for efficient retrieval.
- **evaluation.py**: 
    - Calculates 7 metrics specific to RAGAS.
- **evaluation_local.py**: 
    - Computes 4 traditional evaluation metrics.
- **eval_retrieve.py**: 
    - Evaluates retrieval performance using metrics like MRR and hit rate.


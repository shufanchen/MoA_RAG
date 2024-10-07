# !/usr/bin/python
# -*- coding: utf-8 -*-
import warnings

from llama_index.core import VectorStoreIndex,Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding


import json
import os
# 忽略警告
warnings.filterwarnings('ignore')


def build_vector_index(file_path, model_name="BAAI/bge-small-en-v1.5", chunk_size=512, chunk_overlap=100, persist_dir="./index_storage"):
    """
    Reads a JSONL file, extracts document contexts, and builds a vector index using a specified embedding model.

    Parameters:
        file_path (str): The path to the JSONL file containing documents.
        model_name (str): The HuggingFace embedding model name to use for encoding. Default is "BAAI/bge-small-en-v1.5".
        chunk_size (int): The size of each chunk to split the document into for indexing. Default is 512.
        chunk_overlap (int): The overlap size between chunks. Default is 100.
        persist_dir (str): The directory to store the built index. Default is "./index_storage".

    Returns:
        None
    """
    # Read the JSONL file line by line
    datalines = read_jsonl_file(file_path)

    # Parse each line into a Document object
    documents = []
    updated_data = []
    for line in datalines:
        row = json.loads(line)
        context = row['context']
        document = Document(text=context)
        document.metadata['doc_id'] = document.doc_id  # Store document ID as metadata
        documents.append(document)
        row['doc_id'] = document.doc_id
        updated_data.append(row)  # Save updated row with 'doc_id'

    # Initialize embedding model for vector indexing
    #embed_model = HuggingFaceEmbedding(model_name=model_name)
    try:
        embed_model = OptimumEmbedding(folder_name="./model/"+model_name)
    except:
        embed_model = HuggingFaceEmbedding(model_name=model_name)
    Settings.embed_model = embed_model  # Set global embedding model

    try:
        # Define how documents are split into chunks for indexing
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Create the vector index from documents
        index = VectorStoreIndex.from_documents(documents, transformations=[splitter], show_progress=True)

        # Persist the index to the specified directory
        index.storage_context.persist(persist_dir=persist_dir)
        print(f"Index built successfully! The index root directory is: {persist_dir}")
    except Exception as e:
        # Handle any exceptions during the indexing process
        print(f"Failed to build index: {e}")
    # Generate new file path with '_add_id' suffix
    embed_name = model_name.split('/')[-1]
    new_file_path = file_path.replace(".jsonl", f"_{embed_name}_add_id.jsonl")

    # Save the updated data with 'doc_id' to the new JSONL file
    with open(new_file_path, 'w', encoding='utf-8') as f:
        for sample in updated_data:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def read_jsonl_file(file_path):
    """
    Reads a JSONL file and returns its content as a list of lines.

    Parameters:
        file_path (str): The path to the JSONL file.

    Returns:
        list: A list of JSON strings, each representing a document.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


#先构建一个索引,定义待读取的语料包
#input_dir="./dataset/TAT_QA/TAT_ragas.jsonl"

input_dir = "./dataset/tech_QA/tech_ragas.jsonl"

#input_dir = './dataset/ragas-wiki/wiki_ragas.jsonl'
dataset_name = os.path.basename(input_dir).split("_")[0]
#设置索引的生成根目录
#persist_dir="./dataset/TAT_index"
#persist_dir="./dataset/tech_index"
#选择一个向量模型
embed_model = "BAAI/bge-large-en-v1.5"
embed_model = 'thenlper/gte-small'
#embed_model = 'yco/bilingual-embedding-base'
#embed_model = "jinaai/jina-embeddings-v2-small-en"
model_suffix = embed_model.split("/")[1]  # 取 / 之后的部分
persist_dir = f"./dataset/{dataset_name}_index_{model_suffix}"
OptimumEmbedding.create_and_save_optimum_model(
    embed_model, "./model/"+embed_model
)
build_vector_index(input_dir, model_name=embed_model, chunk_size=512, chunk_overlap=100, persist_dir=persist_dir)
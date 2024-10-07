# !/usr/bin/python
# -*- coding: utf-8 -*-
import json
from llama_index.core.evaluation.retrieval.metrics import MRR, HitRate
from retrieve import retrieve_chunks
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from tqdm import tqdm
import os
import sys

# Ensure stdout uses UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')


def compute_mrr_and_hit_rate(source_file_path: str, rag_result_file_path: str, k: int = 10):
    # 读取源文件
    source_data = {}
    with open(source_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            source_data[entry['question']] = entry['doc_id']  # 使用问题为键，文档ID为值

    # 读取RAG结果文件
    queries = []
    retrieved_results = []
    with open(rag_result_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            question = entry['question']
            chunks_ids = entry['chunks_id']  # 获取检索到的文档ID列表

            queries.append(question)
            retrieved_results.append(chunks_ids)

    # 初始化 MRR 和 Hit Rate 计算器
    mrr_calculator = MRR()
    hit_rate_calculator = HitRate()

    # 初始化 MRR 和 Hit Rate 分数
    total_mrr = 0.0
    total_hit_rate = 0.0

    # 遍历每个查询，计算 MRR 和 Hit Rate
    unsuccessful_query = []
    # Initialize progress bar
    for query, retrieved_ids in tqdm(zip(queries, retrieved_results), total=len(queries),
                                     desc="Calculating MRR and Hit Rate", unit="query"):
        expected_id = source_data.get(query)

        # 确保期望的文档 ID 存在
        if expected_id is None:
            continue

        # 计算 MRR
        mrr_result = mrr_calculator.compute(expected_ids=[expected_id], retrieved_ids=retrieved_ids)
        total_mrr += mrr_result.score

        # 计算 Hit Rate
        hit_rate_result = hit_rate_calculator.compute(expected_ids=[expected_id], retrieved_ids=retrieved_ids)
        if hit_rate_result.score < 1:
            print(query)
            unsuccessful_query.append(query)
        total_hit_rate += hit_rate_result.score

    # 计算平均 MRR 和 Hit Rate
    avg_mrr = total_mrr / len(queries)
    avg_hit_rate = total_hit_rate / len(queries)
    print(f"Average MRR: {avg_mrr}")
    print(f"Average Hit Rate: {avg_hit_rate}")
    save_path = rag_result_file_path.replace('.jsonl', f"_eval_result.txt")
    # Save the average MRR and Hit Rate to a text file
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"Average MRR: {avg_mrr}\n")
        f.write(f"Average Hit Rate: {avg_hit_rate}\n")
        f.write(f"Unsuccussful queries: \n")
        for i, data in enumerate(unsuccessful_query):
            cleaned_data = " ".join(line.strip() for line in data.splitlines() if line.strip())
            f.write(f"{i + 1}: {cleaned_data} \n")
    return avg_mrr, avg_hit_rate


def process_jsonl_and_retrieve(input_jsonl, persist_dir, embed_model_name, top_k=1):
    # Generate output file path based on input file name
    base_name = os.path.basename(input_jsonl)
    file_name, _ = os.path.splitext(base_name)
    output_dir = './retrieval_results/'
    os.makedirs(output_dir, exist_ok=True)
    output_jsonl = os.path.join(output_dir, f"{file_name}_top{top_k}_retrieval.jsonl")
    try:
        embed_model = OptimumEmbedding(folder_name="./model/"+embed_model_name,device="coda:0")
    except:
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name,device="coda:0")
    # Count total lines for progress bar
    with open(input_jsonl, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for _ in infile)

    with open(input_jsonl, 'r', encoding='utf-8') as infile, open(output_jsonl, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, total=total_lines, desc="Processing", unit="line"):
            sample = json.loads(line.strip())

            # Retrieve chunks based on question
            question = sample.get("question", "")
            chunks, chunks_id = retrieve_chunks(persist_dir=persist_dir, query_str=question, embed_model=embed_model,
                                                top_k=top_k)

            # Add retrieved context and chunk IDs to sample
            sample['context'] = chunks
            sample['chunks_id'] = chunks_id

            # Write modified sample to output file
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Retrieval results saved to: {output_jsonl}")
    return output_jsonl
# Usage Example:
if __name__ == "__main__":
# 使用示例
    #source_file_path = './dataset/tech_QA/tech_ragas_bge-base-en-v1.5_add_id.jsonl'
    #source_file_path = './dataset/ragas-wiki/wiki_ragas_bge-base-en-v1.5_add_id.jsonl'
    source_file_path = './dataset/ragas-wiki/wiki_ragasbge-small-en-v1.5_add_id.jsonl'
    #source_file_path ='./dataset/ragas-wiki/wiki_ragas_bge-large-en-v1.5_add_id.jsonl'
    source_file_path = './dataset/tech_QA/tech_ragas_add_id.jsonl'
    index_dir = './dataset/tech_index_bge-small-en-v1.5'
    #index_dir = './dataset/wiki_index_bge-small-en-v1.5'
    #index_dir = './dataset/wiki_index_bge-large-en-v1.5'

    #index_dir = './dataset/wiki_index_jina-embeddings-v2-small-en'
    embedding = "BAAI/bge-small-en-v1.5"
    #embedding = "jinaai/jina-embeddings-v2-small-en"
    save_path = process_jsonl_and_retrieve(source_file_path,index_dir,embedding,top_k=1)
    #save_path = "./retrieval_results/wiki_ragas_add_id_retrieval.jsonl"
    print(compute_mrr_and_hit_rate(source_file_path,save_path))


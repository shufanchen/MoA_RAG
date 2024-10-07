# !/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
from llama_index.core import Settings, StorageContext, load_index_from_storage
# 忽略警告
warnings.filterwarnings('ignore')

def remove_empty_lines(text):
    return "\n".join([line for line in text.splitlines() if line.strip()])
"""
方法描述：基于索引进行查询
【入参】persist_dir:待加载的索引持久化存储目录
【入参】query_str:待查询的字符串
【入参】embed_model:嵌入模型
【入参】top_k:返回的chunk数量
"""
def retrieve_chunks(persist_dir=None,query_str=None,embed_model=None,top_k=1):

    Settings.embed_model=embed_model

    #重新加载存储上下文
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    loaded_index = load_index_from_storage(storage_context)


    #创建查询引擎并发起查询
    base_retriever = loaded_index.as_retriever(similarity_top_k=top_k)



    retrievals = base_retriever.retrieve(query_str)
    chunks = ''
    chunks_id = []
    for i,chunk in enumerate(retrievals):
        chunks+=f'Passage{i+1}:{chunk.text}\n'
        chunks_id.append(chunk.metadata['doc_id'])
    chunks = remove_empty_lines(chunks)

    return chunks,chunks_id
# !/usr/bin/python
# -*- coding: utf-8 -*-
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
import asyncio
import together
from together import AsyncTogether, Together
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from retrieve import retrieve_chunks  # 直接调用已经写好的retrieve_chunks
import os
import json
from tqdm import tqdm
from evaluation import process_all_jsonl_files
#from eval_retrieve import compute_mrr_and_hit_rate
from aiohttp import ClientError
from datetime import datetime
import csv
ccc=1
c = 1

class MOA_RAG:
    def __init__(self, api_key, embed_model_names=["BAAI/bge-small-en-v1.5"],
                 aggregator_model="mistralai/Mixtral-8x22B-Instruct-v0.1",
                 top_k=1, layers=2, max_tokens=512, temperature=[0.7, 0.8],
                 index_dirs=["./dataset/tech_index_bge-small-en-v1.5"]):
        self.client = Together(api_key=api_key)
        self.async_client = AsyncTogether(api_key=api_key)
        self.persist_dirs = index_dirs
        #self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        #self.embed_model = OptimumEmbedding(folder_name="./model/" + embed_model_name)
        self.embed_model_names = embed_model_names
        self.embed_models = [OptimumEmbedding(folder_name="./model/" + model_name) for model_name in self.embed_model_names]

        # try:
        #     self.embed_model = OptimumEmbedding(folder_name="./model/" + embed_model_name)
        # except:
        #     self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        self.aggregator_model = aggregator_model
        self.top_k = top_k
        self.layers = layers
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reference_models = [
            # "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
            # "meta-llama/Llama-3.2-3B-Instruct-Turbo",
            # "Gryphe/MythoMax-L2-13b-Lite",
            "google/gemma-2-9b-it",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "mistralai/Mistral-7B-Instruct-v0.3",
            # "Qwen/Qwen2-72B-Instruct",
            # 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
            # 'deepseek-ai/deepseek-llm-67b-chat',
            # "mistralai/Mixtral-8x22B-Instruct-v0.1",
            # "databricks/dbrx-instruct",
        ]
        self.judge_model = "Gryphe/MythoMax-L2-13b-Lite"
        #self.judge_model = "Qwen/Qwen1.5-72B-Chat"
        self.timestamp_save = None
        self.baselines_base_path = None
        self.final_base_path = None
        self.first_system_prompt = """You are an expert with advanced comprehension skills. Please answer the query based on the provided contexts. Ensure your answer is accurate and directly addresses the query."""
        # self.judger_system_prompt = """
        # You are a professional judge tasked with evaluating a single response based on the query, context, and that specific model's answer.
        # First, determine whether the context is strongly related to the query. If it is, consider the context, query, and the response in your analysis; otherwise, base your judgment solely on the query and the response.
        # Your task is to analyze this single response's strengths, weaknesses, and relevance to the query, ensuring that it is both accurate and appropriate for the next layer of model processing.
        # Finally, make a clear decision on the response in your last sentence: state either "Decision: Accept" or "Decision: Reject". This decision should be based on the response's relevance, accuracy, and its value in contributing to a correct and complete answer to the query.
        # Ensure that the decision is on a new line and follows your analysis for easy identification.
        # """
        self.judger_system_prompt = """You are a professional answer reviewer. Evaluate whether the answer appropriately uses the context to respond to the question. Consider:
        1. **Relevance**: Does the answer address the question using the context?
        2. **Accuracy**: Is the information correct based on the context?
        3. **Completeness**: Does the answer cover all key points of the question?
        Provide your reasoning briefly, then make a clear decision: state either "Decision: Accept" or "Decision: Reject".
        """

        self.aggregator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
"""
        # self.aggregator_system_prompt = """
        # # You have been provided with a question, materials, and various model responses, all of which aim to address the given question. Your task is to synthesize a single, high-quality response by following these principles:
        # #
        # # 1. Critically evaluate each model's response to determine if it correctly and effectively addresses the question. Retain only the parts that are accurate and relevant, filtering out any incorrect, unsupported, or off-topic content.
        # # 2. Carefully assess the provided materials—a potentially lengthy passage—for segments that are truly relevant to the question. Integrate content from the materials only when it directly enhances the accuracy and completeness of the answer. If parts of the materials are irrelevant or unhelpful, they should be disregarded.
        # # 3. For accurate responses that emphasize different perspectives, extract and integrate the most insightful and relevant elements from each. Prioritize synthesizing content that not only correctly answers the question but also provides depth and completeness to the final response.
        # # 4. Craft a final response that is coherent, comprehensive, and directly addresses the question. This response should refine and elevate the initial inputs, ensuring a balanced, accurate, and thorough answer that covers all critical aspects of the query.
        # # """
    async def close(self):
        await self.async_client.session.close()

    def construct_user_prompt(self, materials, question, prev_response=None):
        """Construct the user prompt for the LLM model."""
        if prev_response:
            if len(prev_response) == 1:
                # 将每个prev_response标记为对应模型的名字和顺序
                return f"""
                    Query: {question}
                    Contexts: {materials}
                    Response to be judged:
                    {prev_response[0]}
                    """
            else:
                # 将每个prev_response标记为对应模型的名字和顺序
                responses_with_model_names = []
                for i, response in enumerate(prev_response):
                    if "Decision: Reject." in response:
                        clean_response = response.replace("Decision: Reject.", "").strip()
                        responses_with_model_names.append(
                            f"{i + 1}. [{self.reference_models[i]}]: {clean_response} \n"
                            f"Please note:The response from model '{self.reference_models[i]}' may contain an error, possibly due to referencing incorrect or irrelevant context. Please reflect on potential errors in this model's reply and aim to avoid similar mistakes in your own response."
                        )
                    else:
                        responses_with_model_names.append(f"{i + 1}. [{self.reference_models[i]}]: {response}")

                responses_with_model_names = "\n".join(responses_with_model_names)
                return f"""
                    Query: {question}
                    Contexts: {materials}
                    Responses from previous models:
                    {responses_with_model_names}
                    """
        else:
            return f"""
                Query: {question}
                Contexts: {materials}
                """

    def construct_user_prompt_old(self, materials, question):
        """Construct the user prompt for the LLM model."""
        return f"""
        Question:{question}
        Materials:{materials}
        You are an expert with advanced comprehension skills. Carefully analyze the provided materials and the query, and assess the relevance of each part of the material to the query. If the material provides relevant information, incorporate it into your response. If not, draw upon your own knowledge to address the question. Ensure your answer is accurate and directly addresses the query.
        """

    async def run_llm(self, model, question, context, proposer_response=None, temperature_cur=0.7):
        """Run a single LLM call with a model while accounting for previous responses and rate limits."""
        for sleep_time in [1, 2, 4, 8,16]:
            try:
                if proposer_response:
                    model_index = self.reference_models.index(model)
                    current_proposer_response = [proposer_response[model_index]]
                    messages = (
                        [
                            {
                                "role": "system",
                                "content": self.judger_system_prompt
                            },
                            {"role": "user",
                             "content": self.construct_user_prompt(context, question, current_proposer_response)},
                        ])
                else:
                    messages = ([{"role": "system", "content": self.first_system_prompt},
                                 {"role": "user", "content": self.construct_user_prompt(context, question)}])
                response = await self.async_client.chat.completions.create(
                    model=self.judge_model if proposer_response else model,
                    messages=messages,
                    temperature=temperature_cur,
                    max_tokens=self.max_tokens,
                )
                print("Model: ", self.judge_model if proposer_response else model)
                break
            except (together.error.RateLimitError, ClientError) as e:
                print(e)
                await asyncio.sleep(sleep_time)
        return response.choices[0].message.content

    def store_generated_content(self, question, answer, context, is_final, temp, ground_truth=None, chunks_id=None,
                                prev_results=None, is_judge=False):
        """Store generated content to a specified file."""
        # Extract dataset name from persist_dir
        dataset_name = self.persist_dirs[0].split("/")[-1].split('_')[0]  # Get 'tech' from './dataset/tech_index'
        param = f"temp_{temp}_topk_{self.top_k}"
        # Determine model name
        if is_final:
            second_name = f"MoA_Rag_{param}_layers_{self.layers}_{self.timestamp_save}"
        else:
            second_name = f"Baselines_{param}_layers_{self.layers}_{self.timestamp_save}"  # Get specific model names

        base_path = f"./results/{dataset_name}/{second_name}/"
        os.makedirs(base_path, exist_ok=True)  # Create directory if it doesn't exist

        if not is_final and not is_judge:
            # Store baseline results for each model
            model_name = [ref_model.split("/")[-1] for ref_model in self.reference_models]
            for i, result in enumerate(answer):
                save_baseline_path = os.path.join(base_path, f"{model_name[i]}_{self.embed_model_names[i].split('/')[-1]}_output.jsonl")
                if self.baselines_base_path == None:
                    self.baselines_base_path = base_path
                entry = {
                    "question": question,
                    "answer": result,
                    "context": context[i],
                    "ground_truth": ground_truth,
                    "chunks_id": chunks_id[i],
                    "retrieval_model":self.embed_model_names[i].split('/')[-1]
                }
                with open(save_baseline_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")  # Append each model's output in JSONL format
        elif is_judge:
            model_name = [ref_model.split("/")[-1] for ref_model in self.reference_models]
            for i, result in enumerate(answer):
                save_judge_path = os.path.join(base_path, f"{model_name[i]}_judge.csv")
                if self.baselines_base_path is None:
                    self.baselines_base_path = base_path

                # Prepare the entry as a dictionary
                entry = {
                    "question": question,
                    "judge_result": result,
                    "context": context[i],
                    "ground_truth": ground_truth,
                    "chunks_id": chunks_id[i],
                    "retrieval_model": self.embed_model_names[i].split('/')[-1]
                }

                # Check if the file exists to write headers only once
                file_exists = os.path.isfile(save_judge_path)

                # Write to the CSV file
                with open(save_judge_path, "a", newline="", encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=entry.keys())
                    # Write header only if the file does not exist
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(entry)  # Append each model's output as a new row in CSV format
        else:
            # Store the final aggregated response
            aggregator_name = self.aggregator_model.split('/')[-1].split('-')[0]
            save_final_path = os.path.join(base_path, f"{aggregator_name}_final_response.jsonl")
            entry = {
                "question": question,
                "answer": answer,
                "context": context,
                "ground_truth": ground_truth,
                "chunks_id": chunks_id,
                "prev_results": prev_results,
                "retrieval_model": self.embed_model_names
            }
            with open(save_final_path, "a") as f:
                f.write(json.dumps(entry) + "\n")  # Append final response in JSONL format
            if self.final_base_path == None:
                self.final_base_path = base_path

    async def moa_process(self, question, ground_truth=None):
        """Run the multi-agent process with multiple layers and aggregate responses."""
        all_materials = []
        all_chunks_ids = []

        # Iterate over embed_model and persist_dir pairs
        for embed_model, persist_dir in zip(self.embed_models, self.persist_dirs):
            materials, chunks_id = retrieve_chunks(query_str=question, persist_dir=persist_dir,
                                                   embed_model=embed_model,
                                                   top_k=self.top_k)
            all_materials.append(materials)
            all_chunks_ids.append(chunks_id)

        # 产生第一层proposers的输出
        results = await asyncio.gather(
            *[self.run_llm(model, question, material, temperature_cur=self.temperature[0])
              for model, material in zip(self.reference_models, all_materials)]
        )

        self.store_generated_content(question, results, all_materials, is_final=False, temp=self.temperature[0],
                                     ground_truth=ground_truth, chunks_id=all_chunks_ids)

        # 第二层判决层对第一层的输出进行判决
        judge_comments = []
        for _ in range(1, self.layers - 1):
            judge_comments = await asyncio.gather(
                *[self.run_llm(model, question, material, proposer_response=results,
                               temperature_cur=self.temperature[_])
                  for model, material in zip(self.reference_models, all_materials)]
            )


        self.store_generated_content(question, judge_comments, all_materials, is_final=False, temp=self.temperature[_],
                                     ground_truth=ground_truth, chunks_id=all_chunks_ids, is_judge=True)
        select_results = []
        select_materials = []
        for i, comment in enumerate(judge_comments):
            if "Decision: Accept" in comment:
                select_results.append(results[i])
                if all_materials[i] not in select_materials:#对重复的context不予以加入
                    select_materials.append(all_materials[i])
            elif "Decision: Reject" in comment:
                select_results.append(f"Decision: Reject.{results[i]}")
            else:  # 这里留下用语义向量嵌入判断正向还是负向的选择
                select_results.append(results[i])

        # Run the final aggregation model
        final_materials = "\n".join([f"{i+1}. {material}" for i, material in enumerate(select_materials)])

        final_response = self.client.chat.completions.create(
            model=self.aggregator_model,
            messages=[
                {
                    "role": "system",
                    "content": self.aggregator_system_prompt
                },
                {"role": "user", "content": self.construct_user_prompt(final_materials, question, select_results)},
            ],
            stream=False,
            temperature=self.temperature[-1]
        )
        self.store_generated_content(question, final_response.choices[0].message.content, final_materials, is_final=True,
                                     temp=self.temperature[-1], ground_truth=ground_truth, chunks_id=all_chunks_ids,
                                     prev_results=select_results)

        return final_response.choices[0].message.content


# Define a function to process each record in the jsonl file
async def process_record(record):
    question = record.get('question', '')
    ground_truth = record.get('ground_truth', '')

    # Call the moa_process function and print the result
    await moa_rag.moa_process(question, ground_truth)
    # print(result)


# Read and process the jsonl file
async def process_jsonl(file_path, max_lines=None, begin_lines=0):
    # Calculate the effective total number of lines to read, considering constraints
    with open(file_path, 'r', encoding='utf-8') as file:
        # Count the total lines in the file
        total_lines_in_file = sum(1 for _ in file)

    # Calculate the starting and ending line indices based on begin_lines and max_lines
    start_line = min(begin_lines, total_lines_in_file)  # Ensure starting line is within the file's range
    end_line = min(total_lines_in_file, start_line + max_lines) if max_lines is not None else total_lines_in_file

    # The total number of lines to process
    total_lines_to_process = end_line - start_line

    with open(file_path, 'r', encoding='utf-8') as file:
        # Initialize progress bar
        for i, line in enumerate(tqdm(file, total=total_lines_to_process, desc="Processing JSONL")):
            # Skip lines before start_line
            if i < start_line:
                continue
            # Stop reading if we reach the end_line
            if i >= end_line:
                break
            # Parse each line as a JSON object
            record = json.loads(line.strip())
            # Process the record
            await process_record(record)


# layer1 中的模型就是baseline，其结果单独放到一个文件中
# uptrain
# 用的model写到文档里
#
# Usage Example:
if __name__ == "__main__":
    api_key = "" #改成你的together ai的API
    # temperature_layers= [0.7,0.8,0.6]
    temperature_layers = [0, 0, 0]  # 元素个数一定要和layers参数大小相吻合
    aggregator_model = "microsoft/WizardLM-2-8x22B"
    aggregator_model = 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
    #aggregator_model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    #aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    embed_model = ["BAAI/bge-small-en-v1.5","BAAI/bge-base-en-v1.5","BAAI/bge-large-en-v1.5"]
    # embed_model = 'thenlper/gte-small'
    # embed_model = 'yco/bilingual-embedding-base'
    index_dirs = ['./dataset/tech_index_bge-small-en-v1.5','./dataset/tech_index_bge-base-en-v1.5','./dataset/tech_index_bge-large-en-v1.5']
    # index_dir = './dataset/wiki_index_bge-small-en-v1.5'
    # index_dir = './dataset/TAT_index_bge-small-en-v1.5'

    moa_rag = MOA_RAG(api_key=api_key, layers=3, embed_model_names=embed_model,top_k=1, max_tokens=512, temperature=temperature_layers,
                      index_dirs=index_dirs,
                      aggregator_model=aggregator_model)
    # moa_rag.timestamp_save='2024-09-30 16_12_03'
    if moa_rag.timestamp_save == None:
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        moa_rag.timestamp_save = str(now)[:19].replace(':', '_')

    eval_after = True

    # Set the maximum number of lines to read
    max_lines_to_read = 10  # Replace this with the number of lines you want to read
    # source_path = './dataset/ragas-wiki/wiki_ragas_add_id.jsonl'
    source_path = './dataset/tech_QA/tech_ragas_bge-large-en-v1.5_add_id.jsonl'
    # source_path = './dataset/TAT_QA/TAT_ragas_add_id.jsonl'
    # Run the processing function for the jsonl file
    begin_lines = 0
    asyncio.run(process_jsonl(source_path, max_lines=max_lines_to_read, begin_lines=begin_lines))
    # evlaution after running
    moa_rag.close()
    if eval_after:
        process_all_jsonl_files(moa_rag.baselines_base_path)
        process_all_jsonl_files(moa_rag.final_base_path)
        # eval_target_file = moa_rag.final_base_path+'/'+aggregator_model.split('/')[-1].split('-')[0]+'_final_response.jsonl'
        # compute_mrr_and_hit_rate(source_path,eval_target_file)

# !/usr/bin/python
# -*- coding: utf-8 -*-
import os
import json
from datasets import Dataset
from ragas import evaluate
from langchain.chat_models import ChatOpenAI
from ragas.metrics import answer_similarity, faithfulness, answer_correctness, answer_relevancy, \
    noise_sensitivity_relevant, noise_sensitivity_irrelevant, summarization_score
import datetime
import warnings

# 忽略警告
warnings.filterwarnings('ignore')
os.environ[
    'OPENAI_API_KEY'] = 'sk-iBPKKjivWouQB7uL3781E2AbB6B94608B5Ff96Cc1aD64578'


os.environ['OPENAI_API_BASE'] = "https://api.xty.app/v1"


def read_jsonl(file_path, begin_lines=0, max_lines_to_read=10):
    """
    读取 JSONL 文件并返回指定样本的数据。

    Args:
        file_path (str): JSONL 文件路径
        begin_lines (int): 从哪一行开始读取，默认从第 0 行开始
        max_lines_to_read (int): 最大读取的行数，默认读取所有行

    Returns:
        dict: 包含问题、答案、上下文、ground truth 和 reference contexts 的数据字典
    """
    data_samples = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': [],
        'reference_contexts': []
    }

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            # 如果当前行号小于 begin_lines，跳过该行
            if line_num < begin_lines:
                continue
            # 如果已读取的行数超过 max_lines_to_read，停止读取
            if max_lines_to_read is not None and line_num >= begin_lines + max_lines_to_read:
                break

            sample = json.loads(line)
            data_samples['question'].append(sample.get('question'))
            data_samples['answer'].append(sample.get('answer'))

            # 按“Passage”分隔上下文并逐个添加
            if sample.get('context', '')!='None':
                contexts = sample.get('context', '').split('Passage')
                temp_context = []
                for context in contexts:
                    context = context.strip()
                    if context:  # 只添加非空上下文
                        temp_context.append(context)
                data_samples['contexts'].append(temp_context)  # 每个上下文作为单独的列表添加
            else:
                data_samples['contexts'].append([''])
            if 'response' in file_path.split('/')[-1]:
                data_samples['reference_contexts'].append(
                    sample.get('prev_results') if sample.get('prev_results') is not None else "no content")
            else:
                data_samples['reference_contexts'].append(temp_context)

            data_samples['ground_truth'].append(sample.get('ground_truth'))

    return data_samples


def save_results_to_csv(file_path, results_df, metadata):
    with open(file_path, 'w', encoding='utf-8') as f:
        # 写入元数据（如文件路径、时间戳、指标平均分数）
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        # 写入结果数据
        results_df.to_csv(f, index=False, encoding='utf-8')


def process_evaluation(file_path, begin_lines=0, max_lines_to_read=10):
    """
    Process the evaluation of a JSONL file and save the results to a CSV file.

    Args:
        file_path (str): Path to the input JSONL file.
    """
    # Read data samples from the JSONL file
    langchain_llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    data_samples = read_jsonl(file_path, begin_lines, max_lines_to_read)
    dataset = Dataset.from_dict(data_samples)
    print(dataset)

    # Perform evaluation using the specified metrics and language model
    result = evaluate(
        dataset,
        metrics=[answer_correctness, answer_similarity, faithfulness, answer_relevancy, summarization_score],
        # metrics=[answer_correctness],#, answer_similarity, faithfulness, answer_relevancy, summarization_score],
        #metrics=[faithfulness, answer_relevancy, summarization_score],

        llm=langchain_llm
    )
    print(result)

    # Convert result to pandas DataFrame
    df = result.to_pandas()

    # Add original data file path and timestamp
    timestamp = datetime.datetime.now().isoformat()

    # Calculate average metrics scores
    average_metrics_dict = dict(result)

    # Prepare metadata content
    metadata = {
        "File Path": file_path,
        "Evaluation Time": timestamp,
        # "Average Answer Correctness": average_metrics_dict[df.columns[-1]],

        # # "Average Answer Correctness": average_metrics_dict[df.columns[-2]],
        # # "Average Answer Similarity": average_metrics_dict[df.columns[-1]],
        "Average Answer Correctness": average_metrics_dict[df.columns[-5]],
        "Average Answer Similarity": average_metrics_dict[df.columns[-4]],
        "Average Faithfulness": average_metrics_dict[df.columns[-3]],
        "Average Answer Relevancy": average_metrics_dict[df.columns[-2]],
        # "Average Noise Sensitivity Relevant": average_metrics_dict[df.columns[-3]],
        # "Average Noise Sensitivity Irrelevant": average_metrics_dict[df.columns[-2]],
        "Average Summarization Score": average_metrics_dict[df.columns[-1]]
    }

    # Construct output CSV path
    base_dir = os.path.dirname(file_path)
    model_name = file_path.split('/')[-1]
    model_name = model_name.replace('.jsonl', f'_eval_result_{begin_lines}_{max_lines_to_read}.csv')
    # if 'response' in model_name:
    #     model_name = model_name.split('.')[0]
    output_csv_path = os.path.join(base_dir, model_name)

    # Save results to CSV
    save_results_to_csv(output_csv_path, df, metadata)


def process_all_jsonl_files(file_path, begin_lines=0, max_lines_to_read=10):
    # 遍历指定路径
    for root, dirs, files in os.walk(file_path):
        # 筛选出所有 .jsonl 文件
        jsonl_files = [f for f in files if f.endswith('.jsonl')]
        for jsonl_file in jsonl_files:
            jsonl_path = root + '/' + jsonl_file
            print(f"Processing file: {jsonl_path}")
            process_evaluation(jsonl_path, begin_lines=begin_lines, max_lines_to_read=max_lines_to_read)


# Usage Example:
if __name__ == "__main__":
    file_path = './results/tech_index/Baselines_temp_0.7_topk_1/Meta-Llama-3.1-8B-Instruct-Turbo_output.jsonl'
    # file_path = './results/TAT_index_bge-small-en-v1.5/Baselines_temp_0.7_topk_1/gemma-2-9b-it_output.jsonl'
    # file_path = './results/TAT_index_bge-small-en-v1.5/MoA_Rag_temp_0.6_topk_1_layers_3/final_response.jsonl'
    file_path = './results/TAT_index_bge-small-en-v1.5/Baselines_temp_0.7_topk_1/Meta-Llama-3.1-8B-Instruct-Turbo_output.jsonl'
    # file_path = './results/TAT_index_bge-small-en-v1.5/Baselines_temp_0.7_topk_1/Mistral-7B-Instruct-v0.3_output.jsonl'
    # file_path = './results/TAT_index_bge-small-en-v1.5/Baselines_temp_0.7_topk_1_layers_2/'
    file_path = './results/TAT_index_bge-small-en-v1.5/MoA_Rag_temp_0.8_topk_1_layers_2'
    # file_path = './results/TAT_index_bge-small-en-v1.5/70B'
    file_path = './results/TAT_index_bge-small-en-v1.5/Baselines_temp_0.7_topk_1_layers_2_2024-10-01 14_15_07'
    file_path = './results/tech/Baselines_temp_0_topk_1_layers_3_2024-10-11 18_48_32'
    #file_path = './results/tech/MoA_Rag_temp_0_topk_1_layers_3_2024-10-11 18_18_19'
    #process_all_jsonl_files(file_path,150,50)
    file_path = './ablation/tech/exp1'
    process_all_jsonl_files(file_path,0,200)

# !/usr/bin/python
# -*- coding: utf-8 -*-
from rouge import Rouge
import nltk

nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer
import os
import json
import csv
import datetime


def rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def calculate_bleu(prediction, ground_truth):
    # 将字符串分词
    reference = [word_tokenize(ground_truth)]
    candidate = word_tokenize(prediction)

    # 使用 SmoothingFunction 来避免 BLEU 分数为 0
    smooth = SmoothingFunction().method1

    # 计算 BLEU 分数
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smooth)

    return bleu_score


def calculate_meteor(prediction, ground_truth):
    # 将字符串分词
    reference = word_tokenize(ground_truth)
    candidate = word_tokenize(prediction)

    # 计算 METEOR 分数
    score = meteor_score([reference], candidate)

    return score


def calculate_bert_score(candidates, references, model_type='bert-base-uncased'):
    # 初始化 BERTScorer
    scorer = BERTScorer(model_type=model_type)

    # 计算 P, R, F1
    P, R, F1 = scorer.score(candidates, references)

    # 将结果转换为 Python 列表形式，方便输出
    precision_list = P.tolist()
    recall_list = R.tolist()
    f1_list = F1.tolist()

    return precision_list, recall_list, f1_list


# 计算 JSONL 文件中的指标并保存到 CSV
def evaluate_and_save_metrics(jsonl_path):
    # 加载 JSONL 文件
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 提取文件名及其目录
    file_dir = os.path.dirname(jsonl_path)
    file_name = os.path.basename(jsonl_path)
    file_name = file_name.split('.')[0]
    # 初始化分数列表
    rouge_scores, bleu_scores, meteor_scores, bert_scores = [], [], [], []

    # 提取所有文本用于 BERT-F1 批量计算
    answers = [item["answer"] for item in data]
    ground_truths = [item["ground_truth"] for item in data]

    # 计算 BERT-F1 分数
    _, _1, bert_f1_scores = calculate_bert_score(answers, ground_truths)

    # 遍历样本，计算每个样本的指标
    results = []
    for i, item in enumerate(data):
        question = item.get("question", "")
        answer = item.get("answer", "")
        ground_truth = item.get("ground_truth", "")

        # 计算其他指标
        rouge_l = rougel_score(answer, ground_truth)
        bleu = calculate_bleu(answer, ground_truth)
        meteor = calculate_meteor(answer, ground_truth)
        bert_f1 = bert_f1_scores[i]

        # 记录指标
        rouge_scores.append(rouge_l)
        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        bert_scores.append(bert_f1)

        # 添加结果到列表
        results.append([question, answer, ground_truth, rouge_l, bleu, meteor, bert_f1])

    # 计算平均值
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    avg_bert_f1 = sum(bert_scores) / len(bert_scores)

    # 构建 CSV 文件路径
    csv_file_path = os.path.join(file_dir, f"{file_name}_traditional_metrics.csv")

    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 写入 CSV 文件
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入文件名和评测时间
        writer.writerow(["File Name", jsonl_path])
        writer.writerow(["Evaluation Time", current_time])
        # 写入平均指标
        writer.writerow(["Average ROUGE-L", avg_rouge])
        writer.writerow(["Average BLEU", avg_bleu])
        writer.writerow(["Average METEOR", avg_meteor])
        writer.writerow(["Average BERT-F1", avg_bert_f1])
        # 写入每个样本的指标
        writer.writerow(["Question", "Answer", "Ground Truth", "ROUGE-L", "BLEU", "METEOR", "BERT-F1"])
        for row in results:
            writer.writerow(row)

    print(f"Metrics saved to {csv_file_path}")

def process_all_jsonl_files(file_path):
    # 遍历指定路径
    for root, dirs, files in os.walk(file_path):
        # 筛选出所有 .jsonl 文件
        jsonl_files = [f for f in files if f.endswith('.jsonl')]
        for jsonl_file in jsonl_files:
            jsonl_path = os.path.join(root, jsonl_file)
            print(f"Processing file: {jsonl_path}")
            # 调用 evaluate_and_save_metrics 函数
            evaluate_and_save_metrics(jsonl_path)

if __name__ == "__main__":
    file_path = './results/wiki_index_bge-small-en-v1.5/Baselines_temp_0.7_topk_1_layers_2_2024-10-01 16_29_21'
    #file_path = './results/tech_index_bge-small-en-v1.5/Baselines_temp_0.7_topk_1_layers_2_2024-10-02 09_07_26'
    #file_path = './results/tech_index_bge-small-en-v1.5/MoA_Rag_temp_0.8_topk_1_layers_2_2024-10-02 09_07_26'
    file_path = './results/tech_index_bge-small-en-v1.5/Baselines_temp_0.7_topk_1_layers_2_2024-10-01 17_16_23'
    #file_path = './results/tech_index_bge-small-en-v1.5/MoA_Rag_temp_0.8_topk_1_layers_2_2024-10-01 17_16_23'
    # file_path = './results/tech/Baselines_temp_0_topk_1_layers_3_2024-10-12 19_51_33'
    # process_all_jsonl_files(file_path)
    file_path = './results/tech/MoA_Rag_temp_0_topk_1_layers_3_2024-10-09 13_13_25'
    process_all_jsonl_files(file_path)
# jsonl_path = './results/wiki_index_bge-small-en-v1.5/MoA_Rag_temp_0.8_topk_1_layers_2_2024-10-01 16_29_21/WizardLM_final_response.jsonl'
# evaluate_and_save_metrics(jsonl_path)


# print(rougel_score('this is an apple','I think it is an apple'))
# print(calculate_bleu('this is an apple','I think it is an apple'))
# print(calculate_meteor('this is an apple','I think it is an apple'))
# print(calculate_bert_score(['this is an apple'],['I think it is an apple']))

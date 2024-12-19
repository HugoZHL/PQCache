import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset', nargs='+')
    parser.add_argument('--exp_name', type=str, default="default_exp")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    dataset_name_list = args.dataset
    exp_name = args.exp_name
    for dataset_name in dataset_name_list:
        if args.e:
            path = f"pred_e/{args.model}/{dataset_name}/"
        else:
            path = f"pred/{args.model}/{dataset_name}/"
        if exp_name is not None:
            path = "".join([path,f"{exp_name}/"])
        all_files = os.listdir(path)
        print("Evaluating on:", all_files)
        scores = dict()
        compressions = dict()
        for filename in all_files:
            if not filename.endswith("jsonl"):
                continue
            predictions, answers, lengths = [], [], []
            # all_input_tokens, all_compress_tokens = 0,0
            with open(f"{path}{filename}", "r", encoding="utf-8") as f:
                line_cnt = 0
                for line in f:
                    line_cnt += 1
                    data = json.loads(line)
                    predictions.append(data["pred"])
                    answers.append(data["answers"])
                    # all_input_tokens += data["input_tokens"]
                    # all_compress_tokens += data["compress_tokens"]
                    all_classes = data["all_classes"]
                    if "length" in data:
                        lengths.append(data["length"])
            if args.e:
                score = scorer_e(dataset_name, predictions, answers, lengths, all_classes)
            else:
                score = scorer(dataset_name, predictions, answers, all_classes)
            scores["_".join([dataset_name, filename])] = score
            # compressions["_".join([dataset_name, filename])] = all_compress_tokens / all_input_tokens
        if args.e:
            out_path = f"{path}result.json"
        else:
            out_path = f"{path}result.json"
        with open(out_path, "w") as f:
            json.dump({"score":scores,"compression":compressions}, f, ensure_ascii=False, indent=4)

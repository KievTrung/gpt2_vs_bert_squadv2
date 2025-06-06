"""Official evaluation script for SQuAD version 2.0.

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""
import argparse
import collections
import os
import re
import string
import sys
import numpy as np


ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

OPTS = None

def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                qid_to_has_ans[qa["id"]] = bool(qa["answers"]["text"])
    return qid_to_has_ans


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold) # get every tokens of truth ans
    pred_toks = get_tokens(a_pred) # get every tokens of pred ans
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks) # make common token couter dict of truth and pred
    num_same = sum(common.values()) # sum over all the common count value 
    if len(gold_toks) == 0 or len(pred_toks) == 0: # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                qid = qa["id"]
                gold_answers = [t for t in qa["answers"]["text"] if normalize_answer(t)]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = [""]
                if qid not in preds:
                    print(f"Missing prediction for {qid}")
                    continue
                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval[f"{prefix}_{k}"] = new_eval[k]


def plot_pr_curve(precisions, recalls, out_image, title):
    plt.step(recalls, precisions, color="b", alpha=0.2, where="post")
    plt.fill_between(recalls, precisions, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.title(title)
    plt.savefig(out_image)
    plt.clf()


def make_precision_recall_eval(scores, na_probs, num_true_pos, qid_to_has_ans, out_image=None, title=None):
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    true_pos = 0.0
    cur_p = 1.0
    cur_r = 0.0
    precisions = [1.0]
    recalls = [0.0]
    avg_prec = 0.0

    for i, qid in enumerate(qid_list):
        if qid_to_has_ans[qid]:
            true_pos += scores[qid]
        cur_p = true_pos / float(i + 1)
        cur_r = true_pos / float(num_true_pos)

        if i == len(qid_list) - 1 or na_probs[qid] != na_probs[qid_list[i + 1]]:
            # i.e., if we can put a threshold after this point
            avg_prec += cur_p * (cur_r - recalls[-1])
            precisions.append(cur_p)
            recalls.append(cur_r)
    if out_image:
        plot_pr_curve(precisions, recalls, out_image, title)
    return {"ap": 100.0 * avg_prec}


def run_precision_recall_analysis(main_eval, exact_raw, f1_raw, na_probs, qid_to_has_ans, out_image_dir):
    if out_image_dir and not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)
    num_true_pos = sum(1 for v in qid_to_has_ans.values() if v)
    if num_true_pos == 0:
        return
    pr_exact = make_precision_recall_eval(
        exact_raw,
        na_probs,
        num_true_pos,
        qid_to_has_ans,
        out_image=os.path.join(out_image_dir, "pr_exact.png"),
        title="Precision-Recall curve for Exact Match score",
    )
    pr_f1 = make_precision_recall_eval(
        f1_raw,
        na_probs,
        num_true_pos,
        qid_to_has_ans,
        out_image=os.path.join(out_image_dir, "pr_f1.png"),
        title="Precision-Recall curve for F1 score",
    )
    oracle_scores = {k: float(v) for k, v in qid_to_has_ans.items()}
    pr_oracle = make_precision_recall_eval(
        oracle_scores,
        na_probs,
        num_true_pos,
        qid_to_has_ans,
        out_image=os.path.join(out_image_dir, "pr_oracle.png"),
        title="Oracle Precision-Recall curve (binary task of HasAns vs. NoAns)",
    )
    merge_eval(main_eval, pr_exact, "pr_exact")
    merge_eval(main_eval, pr_f1, "pr_f1")
    merge_eval(main_eval, pr_oracle, "pr_oracle")


def parse_args():
    parser = argparse.ArgumentParser("Official evaluation script for SQuAD version 2.0.")
    parser.add_argument("--reference_file", metavar="references.pkl", help="truth ans pickle file.")
    parser.add_argument("--pred_file", metavar="pred.pkl", help="Model predictions.")
    parser.add_argument(
        "--out-file", "-o", metavar="eval.json", help="Write accuracy metrics to file (default is stdout)."
    )
    parser.add_argument(
        "--na-prob-file", "-n", metavar="na_prob.json", help="Model estimates of probability of no answer."
    )
    parser.add_argument(
        "--na-prob-thresh",
        "-t",
        type=float,
        default=1.0,
        help='Predict "" if no-answer probability exceeds this (default = 1.0).',
    )
    parser.add_argument(
        "--out-image-dir", "-p", metavar="out_images", default=None, help="Save precision-recall curves to directory."
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main():
    import pickle
    with open(OPTS.reference_file, 'rb') as ref, open(OPTS.pred_file, 'rb') as pred:
        references, predictions= pickle.load(ref), pickle.load(pred)

    na_probs = {p["id"]: p["no_answer_probability"] for p in predictions}
    dataset = [{"paragraphs": [{"qas": references}]}]
    predictions = {p["id"]: p["prediction_text"] for p in predictions}

    qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(dataset, predictions)
    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans, OPTS.na_prob_thresh)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, OPTS.na_prob_thresh)
    out_eval = make_eval_dict(exact_thresh, f1_thresh)

    if has_ans_qids:
        has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
        merge_eval(out_eval, has_ans_eval, "HasAns")
    if no_ans_qids:
        no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, "NoAns")

    run_precision_recall_analysis(out_eval, exact_raw, f1_raw, na_probs, qid_to_has_ans, OPTS.out_image_dir)

if __name__ == "__main__":
    OPTS = parse_args()
    if OPTS.out_image_dir:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    main()
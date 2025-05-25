import numpy as np
import torch
from tqdm import tqdm
import math


def find_answer_in_tokenize_sample(tokenize_samples, answers):
    offset_mapping = tokenize_samples["offset_mapping"]
    sample_idxs = tokenize_samples["overflow_to_sample_mapping"]
    start_idx = []
    end_idx = []

    # go through each encoded sample
    for i, offset in enumerate(offset_mapping):
        answer = answers[sample_idxs[i]]

        # check if ans exist
        if len(answer["answer_start"]) == 0:
            start_idx.append(0)
            end_idx.append(0)
        else:
            ans_start_char_idx = answer["answer_start"][0]
            ans_end_char_idx = ans_start_char_idx + len(answer["text"][0])

            # find context in feature
            idx = 0
            while tokenize_samples.sequence_ids(i)[idx] != 1:
                idx += 1
            context_start = idx
            try:
                while tokenize_samples.sequence_ids(i)[idx] == 1:
                    idx += 1
                context_end = idx - 1
            except Exception as e:
                context_end = idx - 1

            if (
                offset[context_start][0] > ans_start_char_idx
                or offset[context_end][1] < ans_end_char_idx
            ):
                start_idx.append(0)
                end_idx.append(0)
            else:
                # find the ans inside context
                idx = context_start
                while idx <= context_end and offset[idx][0] <= ans_start_char_idx:
                    idx += 1
                start_idx.append(idx - 1)

                while idx <= context_end and offset[idx][1] <= ans_end_char_idx:
                    idx += 1
                end_idx.append(idx)

    tokenize_samples["start_positions"] = start_idx
    tokenize_samples["end_positions"] = end_idx

    return tokenize_samples


def postprocessing_qa_prediction(
    start_logits,
    end_logits,
    samples,
    sample_to_features_,
    offset_mapping_,
    n_best=20,
    max_ans_len=30,
    null_threshold=0.0,
):
    pred_ans = []

    # go through each sample
    for sample in samples:
        sample_id = sample["id"]
        context = sample["context"]
        ans = []
        min_null_score = None

        # go through each feature of the sample
        for feature_idx in sample_to_features_[sample_id]:
            # apply softmax to start, end logit
            start_logit = start_logits[feature_idx]
            end_logit = end_logits[feature_idx]

            offsets = offset_mapping_[feature_idx]

            # find min unk score for all feature of sample
            feature_null_score = start_logit[0] + end_logit[0]
            if min_null_score is None or min_null_score > feature_null_score:
                min_null_score = feature_null_score

            # get the top n_best biggest probability
            start_idxs = np.argsort(start_logit[1:])[-1 : -n_best - 1 : -1].tolist()
            end_idxs = np.argsort(end_logit[1:])[-1 : -n_best - 1 : -1].tolist()

            # find the most probable ans in those 20 start, end logit
            for start_idx in start_idxs:
                for end_idx in end_idxs:
                    # skip ans not in context
                    if offsets[start_idx] is None or offsets[end_idx] is None:
                        continue

                    # skip ans that not in (0, max_ans_len)
                    if start_idx > end_idx or end_idx - start_idx + 1 > max_ans_len:
                        continue

                    ans.append(
                        {
                            "ans": context[offsets[start_idx][0] : offsets[end_idx][1]],
                            "score": start_logit[start_idx] + end_logit[end_idx],
                        }
                    )
        # get the best n ans based on score
        ans = sorted(ans, key=lambda x: x["score"], reverse=True)[:n_best]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid failure.
        if len(ans) == 0 or (len(ans) == 1 and ans[0]["ans"] == ""):
            ans.insert(0, {"ans": "empty", "score": 0})

        # find the best non-null ans
        i = 0
        while ans[i]["ans"] == "":
            i += 1
        best_non_null_pred = ans[i]

        # compare score diff with threshold
        score_diff = min_null_score - best_non_null_pred["score"]

        # convert score diff to probability via sigmoid
        score_diff_prob = 1 / (1 + math.exp(-score_diff))

        pred_ans.append(
            {
                "id": sample_id,
                "prediction_text": (
                    "" if score_diff > null_threshold else best_non_null_pred["ans"]
                ),
                "no_answer_probability": score_diff_prob,
            }
        )

    return pred_ans


def inferencing(
    accelerator_,
    model_,
    dataloader,
    dataset,
    sample_to_features_,
    offset_mapping_,
    metric=None,
):
    start_logits, end_logits, l = [], [], []
    d_size = dataloader.total_dataset_length

    is_eval = True if "answers" in dataset.features else False
    if is_eval and metric is None:
        raise ValueError("metric not define")

    iter = 100

    model_.eval()
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating..." if is_eval else "inferencing...")):
        with torch.no_grad():
            outputs = model_(**batch)
            if is_eval:
                loss_ = outputs.loss

        start_logits.append(accelerator_.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator_.gather(outputs.end_logits).cpu().numpy())

        if is_eval:
            s_ = {"b": i, "l": loss_.item()}
            l += [s_]
            if i == iter:
                print("b: {}, val_l: {}\n".format(s_["b"], s_["l"]), end="")
                iter += 300

    start_logits = np.concatenate(start_logits)[:d_size]
    end_logits = np.concatenate(end_logits)[:d_size]

    # model evaluate
    pred_ans_ = postprocessing_qa_prediction(
        start_logits,
        end_logits,
        samples=dataset,
        offset_mapping_=offset_mapping_,
        sample_to_features_=sample_to_features_,
    )
    if is_eval:
        metric_ = metric.compute(
            predictions=pred_ans_,
            references=[
                {"id": sample["id"], "answers": sample["answers"]} for sample in dataset],
        )
        print(metric_)

    return (pred_ans_, metric_, l) if is_eval else (pred_ans_)

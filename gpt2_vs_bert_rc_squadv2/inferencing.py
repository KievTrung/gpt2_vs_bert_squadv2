import os
from collections import defaultdict
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
)
from datasets import load_dataset
from accelerate import Accelerator
from .utils import inferencing
from .preprocessing import preprocess_eval_dataset
from .fine_tuning import prepare_eval_dataset
from . import config
import torch, json, argparse, gc, pickle
import evaluate


class QaModel:
    accelerator, metric = None, None

    def __init__(
        self, model_dir=None, tokenizer_dir=None
    ):
        model_card = config.c['model_card']
        if model_dir is None and tokenizer_dir is None:
            model_dir = os.path.join(config.c["model_path"], model_card)
            tokenizer_dir = os.path.join(model_dir, "tokenizer")
            assert os.path.isfile(os.path.join(model_dir, "model.safetensors")), f"model missing in {model_dir}"
            assert os.path.isfile(os.path.join(tokenizer_dir, "tokenizer.json")), f"tokenizer missing in {tokenizer_dir}"
            
        device = config.c["device"]
        precision = (
            None
            if device == "cpu"
            else "bf16" if torch.cuda.get_device_capability()[0] > 7 else "fp16"
        )
        if QaModel.accelerator is None:
            QaModel.accelerator = Accelerator(mixed_precision=precision)

        print('Loading model...')
        self.model = QaModel.accelerator.prepare(torch.compile(AutoModelForQuestionAnswering.from_pretrained(model_dir)).to(device))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        print("loaded model completed")

    def __del__(self):
        print("Deleting model...")
        self.model.cpu()
        del self.model
        del QaModel.accelerator
        torch.cuda.empty_cache()
        gc.collect()

    def preprocessing(self, samples):
        return preprocess_eval_dataset(
            samples,
            self.tokenizer,
            config.c["model_card"],
            config.c["max_length"],
            config.c["stride"],
        )

    def preparing(self, tokenized_samples, collate_fn):
        return prepare_eval_dataset(
            tokenized_samples, collate_fn, device=config.c["device"]
        )

    def clear_dataloader(self, dataloader):
        print("clear dataloader...")
        dataloader.dataset.clear()
        torch.cuda.empty_cache()
        del dataloader
        gc.collect()

    def inferencing(self, dataloader, samples, sample_to_features, offset_mapping):
        dataloader = QaModel.accelerator.prepare(dataloader)
        if "answers" in samples.features and QaModel.metric is None:
            QaModel.metric = evaluate.load("squad_v2")
        return inferencing(
            QaModel.accelerator,
            self.model,
            dataloader,
            dataset=samples,
            sample_to_features_=sample_to_features,
            offset_mapping_=offset_mapping,
            metric=QaModel.metric,
        )

    def pipeline(self, samples):
        tokenized_dataset = self.preprocessing(samples)
        dataloader, sample_to_features, offset_mapping = self.preparing(
            tokenized_dataset, collate_fn=default_data_collator
        )
        result = self.inferencing(dataloader, samples, sample_to_features, offset_mapping)
        self.clear_dataloader(dataloader)
        return result

    def aggregating_result(self, samples, pred_ans):
        pred_ans_ = defaultdict(list)
        for pred in pred_ans:
            pred_ans_[pred["id"]].append(
                {
                    "prediction_text": pred["prediction_text"],
                    "no_answer_probability": pred["no_answer_probability"],
                }
            )

        return (
            {
                sample["id"]: {
                    "context": sample["context"],
                    "question": sample["question"],
                    "true answers": sample["answers"],
                    "pred": sorted(
                        pred_ans_[sample['id']], key=lambda x: x["no_answer_probability"]
                    ),
                }
                for sample in samples
            }
            if "answers" in samples.features 
            else {
                sample["id"]: {
                    "context": sample["context"],
                    "question": sample["question"],
                    "pred": sorted(
                        pred_ans_[sample['id']], key=lambda x: x["no_answer_probability"]
                    ),
                }
                for sample in samples
            }
        )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dataset",
        "-i",
        default=os.path.join(config.c["dataset_path"], "test.json"),
        help="Specify test set file",
    )
    args = parser.parse_args()

    # load dataset
    print("Loading test set...")
    samples = load_dataset('json', data_files=args.input_dataset)['train']

    # find model
    print("Loading model...")
    model = QaModel()
    result = model.pipeline(samples)
    pred_ans = model.aggregating_result(samples, result[0] if isinstance(result, tuple) else result)

    # save result
    print('Saving result...')
    save_path = config.c['result_path'] + f'_{config.c["model_card"]}'
    os.makedirs(save_path, exist_ok=True)
    func = lambda x : os.path.join(save_path, x)
    with open(func("result.json"), "w") as f:
        json.dump(pred_ans, f)

    if len(result) != 1:
        with open(func("metric.json"), "w") as f:
            json.dump({'metric' : result[1], 'loss' : result[2]}, f)

    # clean up
    del model



if __name__ == "__main__":
    main()

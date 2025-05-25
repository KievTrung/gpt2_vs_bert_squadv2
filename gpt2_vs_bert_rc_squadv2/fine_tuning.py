from collections import defaultdict
from transformers import default_data_collator, get_scheduler
from transformers import AutoModelForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from .utils import inferencing
from datasets import load_from_disk
from tqdm import tqdm
from accelerate import Accelerator
import torch
import argparse, os, pickle, shutil, inspect, gc
from . import config
import evaluate


class Dataset_(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return (
            {
                "input_ids": self.data["input_ids"][idx],
                "attention_mask": self.data["attention_mask"][idx],
                "start_positions": self.data["start_positions"][idx],
                "end_positions": self.data["end_positions"][idx],
            }
            if "start_positions" in self.data
            else {
                "input_ids": self.data["input_ids"][idx],
                "attention_mask": self.data["attention_mask"][idx],
            }
        )
        
    def clear(self):
        self.data.clear()


def cls_dir(path):
    print(f"Clearing {path}...")
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def prepare_train_dataset(
    train_dataset_, collate_fn, batch_size=8, device="cpu", shuffle=False
):
    train_dataset_.convert_to_tensors("pt")
    train_dataset_.to(device)

    return DataLoader(
        Dataset_(train_dataset_),
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )


def prepare_eval_dataset(
    eval_dataset, collate_fn, batch_size=8, device="cpu", shuffle=False
):
    # map each sample to every feature in validation set
    sample_to_features = defaultdict(list)
    for idx, sample_id in enumerate(eval_dataset["sample_ids"]):
        sample_to_features[sample_id].append(idx)

    offset_mapping = eval_dataset.pop("offset_mapping")
    for k in list(eval_dataset.keys()):
        if k not in ("input_ids", "attention_mask", "start_positions", "end_positions"):
            eval_dataset.pop(k)
    eval_dataset.convert_to_tensors("pt")
    eval_dataset.to(device)

    eval_dataloader = DataLoader(
        Dataset_(eval_dataset),
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )
    return eval_dataloader, sample_to_features, offset_mapping


def main():
    def clean_up():
        nonlocal train_dataloader, eval_dataloader
        nonlocal tokenized_train_dataset, tokenized_dev_dataset
        nonlocal model, lr_scheduler, optimizer, accelerator

        # clean up code
        print("cleaning up...")
        tokenized_train_dataset.clear()
        tokenized_dev_dataset.clear()
        del train_dataloader
        del eval_dataloader

        model.cpu()
        del model
        del lr_scheduler
        del optimizer
        del accelerator

        torch.cuda.empty_cache()
        gc.collect()

    device = config.c["device"]
    model_card = config.c["model_card"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", default=3)
    parser.add_argument("--learning_rate", "-le", default=2e-5)
    parser.add_argument("--scheduler", "-s", default="linear")
    parser.add_argument("--warm_up", "-w", default=0.05)
    parser.add_argument("--max_grad_norm", "-mgn", default=1.0)

    precision = (
        None
        if device == "cpu"
        else "bf16" if torch.cuda.get_device_capability()[0] > 7 else "fp16"
    )
    parser.add_argument("--precision", "-p", default=precision)
    args = parser.parse_args()

    # load data
    path = config.c["preprocessed_dataset_path"]
    print("loading train, dev set...")
    try:
        fp = lambda x: os.path.join(path["dir"], path[x])
        with open(fp("train"), "rb") as f, open(fp("dev"), "rb") as f_:
            tokenized_train_dataset = pickle.load(f)
            tokenized_dev_dataset = pickle.load(f_)

        dev_set = load_from_disk(os.path.join(config.c["dataset_path"], "dev"))
    except Exception as e:
        print("Error loading dataset, detail:", e)
        return

    # prepare data loader
    print("preparing dataloader...")
    train_dataloader = prepare_train_dataset(
        tokenized_train_dataset, collate_fn=default_data_collator, device=device
    )
    eval_dataloader, sample_to_features, offset_mapping = prepare_eval_dataset(
        tokenized_dev_dataset, collate_fn=default_data_collator, device=device
    )

    start_e, stop_e = 0, int(args.epochs)
    steps_per_epoch = len(train_dataloader)
    total_steps = (stop_e - start_e) * steps_per_epoch

    # download model
    print("Downloading model...")
    model = AutoModelForQuestionAnswering.from_pretrained(model_card)
    if model_card.count("gpt") != 0:
        model.resize_token_embeddings(50304)
    print("Compiling model...")
    model = torch.compile(model).to(device)

    # load metric
    print("load metric...")
    metric__ = evaluate.load("squad_v2")

    # set hyper param
    print("Setting hyperparams...")
    fused_available = "fused" in inspect.signature(AdamW).parameters
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, fused=fused_available)

    lr_scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(args.warm_up * total_steps),
        num_training_steps=total_steps,
    )

    # define accelerator
    print("Defining accelerator...")
    accelerator = Accelerator(mixed_precision=args.precision)
    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = (
        accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader, eval_dataloader
        )
    )

    # check for any checkpoint
    cp = os.path.join(config.package, "checkpoint")
    print("Check for any checkpoint...")
    mode = "w"
    if os.path.isdir(cp) and len(os.listdir(cp)) != 0:
        print("Found checkpoint, loading...")
        mode = "a"
        accelerator.load_state(cp)
        with open(f"{cp}/epoch.pkl", "rb") as f:
            e = pickle.load(f)
            start_e, stop_e = e["start"], e["stop"]
        cls_dir(cp)

    # fine turning
    iter = 100
    best_score = 0
    result_path = os.path.join(config.package, f"fine_tune_result_{model_card}")
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, "train_loss.txt"), mode) as f, open(
        os.path.join(result_path, "val_loss.txt"), mode
    ) as f2, open(os.path.join(result_path, "eval.txt"), mode) as f3:
        try:
            print("Begin finetuning...")
            for epoch in range(start_e, stop_e):
                # train
                model.train()
                for i, batch in enumerate(tqdm(train_dataloader, desc="Training...")):
                    with accelerator.autocast():
                        outputs = model(**batch)
                        loss = outputs.loss

                    accelerator.backward(loss)
                    # clip gradient
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    s = f"e: {epoch}, b: {i}, l: {loss}\n"
                    f.write(s)
                    if i == iter:
                        print(s, end="")
                        iter += 500
                iter = 100

                # eval
                _, metric_, l = inferencing(
                    accelerator,
                    model,
                    eval_dataloader,
                    dev_set,
                    sample_to_features,
                    offset_mapping,
                    metric=metric__,
                )

                # save val loss
                for i in l:
                    f2.write("e: {}, b: {}, val_l: {}\n".format(epoch, i["b"], i["l"]))

                # save metric
                f3.write(f"epoch {epoch}, metric: {metric_}\n")

                # check if model performance has increased over epoch if not then early stop
                if metric_["f1"] > best_score:
                    print("New best score !!")
                    best_score = metric_["f1"]

                    # save best model
                    print("saving best model...")
                    accelerator.wait_for_everyone()
                    accelerator.unwrap_model(model).save_pretrained(
                        os.path.join(config.c["model_path"], model_card),
                        save_function=accelerator.save,
                    )

                # save check point
                print('Saving checkpoint...')
                accelerator.save_state(cp)
                with open(f"{cp}/epoch.pkl", "wb") as f_:
                    pickle.dump({"start": start_e + 1, "stop": stop_e}, f_)
            print("Finetuning completed...")
        except Exception as e:
            print("Exception:", e)
            clean_up()
            return

    # clear check point
    cls_dir(cp)
    clean_up()


if __name__ == "__main__":
    main()

from transformers import AutoTokenizer, AddedToken
from collections import defaultdict
from .utils import find_answer_in_tokenize_sample
from . import config
import os, argparse, pickle
from datasets import load_from_disk, load_dataset, concatenate_datasets


def build_squadv2_mini():
    s = config.c["squadv2_config"]
    train_total, dev_total, test_total = s["train_s"], s["dev_s"], s["test_s"]
    t_samples = train_total + dev_total + test_total

    nev_train, nev_dev, nev_test = s["neg_train_s"], s["neg_dev_s"], s["neg_test_s"]
    pos_train, pos_dev, pos_test = (
        train_total - nev_train,
        dev_total - nev_dev,
        test_total - nev_test,
    )

    p = lambda x, y: x / y
    p_nev_train, p_nev_dev, p_nev_test = (
        p(nev_train, t_samples),
        p(nev_dev, t_samples),
        p(nev_test, t_samples),
    )
    p_pos_train, p_pos_dev, p_pos_test = (
        p(pos_train, t_samples),
        p(pos_dev, t_samples),
        p(pos_test, t_samples),
    )

    # we dont have test set currently, so have to redistribute sample using all p above
    t_samples = train_total + dev_total

    b = lambda x, y: round(x * y)
    # define bin size
    bs_nev_train, bs_nev_dev, bs_nev_test = (
        b(p_nev_train, t_samples),
        b(p_nev_dev, t_samples),
        b(p_nev_test, t_samples),
    )
    bs_pos_train, bs_pos_dev, bs_pos_test = (
        b(p_pos_train, t_samples),
        b(p_pos_dev, t_samples),
        b(p_pos_test, t_samples),
    )

    print("total:", t_samples)
    print(
        f"b_nev_train: {bs_nev_train}({p_nev_train}), b_nev_dev: {bs_nev_dev}({p_nev_dev}), b_nev_test: {bs_nev_test}({p_nev_test})"
    )
    print(
        f"b_pos_train: {bs_pos_train}({p_pos_train}), b_pos_dev: {bs_pos_dev}({p_pos_dev}), b_pos_test: {bs_pos_test}({p_pos_test})"
    )

    # build article dict
    squad = load_dataset("squad_v2")
    train, dev = squad["train"], squad["validation"]
    articles = defaultdict(lambda: {"z": [], "nz": []})

    print("build article dict...")
    for i, sample in enumerate(train):
        articles[sample["title"]][
            "z" if len(sample["answers"]["text"]) == 0 else "nz"
        ].append(train.select([i]))

    for i, sample in enumerate(dev):
        articles[sample["title"]][
            "z" if len(sample["answers"]["text"]) == 0 else "nz"
        ].append(dev.select([i]))

    # fill out bins
    b_pos_train, b_nev_train = [], []
    b_pos_dev, b_nev_dev = [], []
    b_pos_test, b_nev_test = [], []

    print("filling bins ...")
    for title, sample in articles.items():
        if len(b_pos_train) <= bs_pos_train:
            b_pos_train += sample["nz"]
            b_nev_train += sample["z"]
        elif len(b_pos_test) <= bs_pos_test:
            b_pos_test += sample["nz"]
            b_nev_test += sample["z"]
        elif len(b_pos_dev) <= bs_pos_dev:
            b_pos_dev += sample["nz"]
            b_nev_dev += sample["z"]
        else:
            b_pos_test += sample["nz"]
            b_nev_test += sample["z"]

    print("Concatenating bins...")
    b_pos_train, b_nev_train = concatenate_datasets(b_pos_train), concatenate_datasets(
        b_nev_train
    )
    b_pos_dev, b_nev_dev = concatenate_datasets(b_pos_dev), concatenate_datasets(
        b_nev_dev
    )
    b_pos_test, b_nev_test = concatenate_datasets(b_pos_test), concatenate_datasets(
        b_nev_test
    )

    print(
        f"b_nev_train: {len(b_nev_train)}, b_nev_dev: {len(b_nev_dev)}, b_nev_test: {len(b_nev_test)}"
    )
    print(
        f"b_pos_train: {len(b_pos_train)}, b_pos_dev: {len(b_pos_dev)}, b_pos_test: {len(b_pos_test)}"
    )

    train_set = concatenate_datasets([b_pos_train, b_nev_train])
    dev_set = concatenate_datasets([b_pos_dev, b_nev_dev])
    test_set = concatenate_datasets([b_pos_test, b_nev_test])

    train_t_ar, train_nev_ar = len(set(train_set["title"])), len(
        set(b_nev_train["title"])
    )
    dev_t_ar, dev_nev_ar = len(set(dev_set["title"])), len(set(b_nev_dev["title"]))
    test_t_ar, test_nev_ar = len(set(test_set["title"])), len(set(b_nev_test["title"]))

    print(
        f"Train: \n\tTotal samples:{len(b_pos_train) + len(b_nev_train)}, neg samples: {len(b_nev_train)}, total articles: {train_t_ar}, neg articles: {train_nev_ar}"
    )
    print(
        f"Dev: \n\tTotal samples:{len(b_pos_dev) + len(b_nev_dev)}, nev samples: {len(b_nev_dev)}, total articles: {dev_t_ar}, neg articles: {dev_nev_ar}"
    )
    print(
        f"Test: \n\tTotal samples:{len(b_pos_test) + len(b_nev_test)}, nev samples: {len(b_nev_test)}, total articles: {test_t_ar}, neg articles: {test_nev_ar}"
    )

    return train_set, dev_set, test_set


def tokenize(example, tokenizer, model_card, max_length, stride):
    f_question = lambda questions: [
        f"{tokenizer.unk_token} question: " + question.strip() + " context: "
        for question in questions
    ]
    return tokenizer(
        (
            f_question(example["question"])
            if model_card.count("gpt2") != 0
            else example["question"]
        ),
        example["context"],
        max_length=max_length,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        stride=stride,
        padding="max_length",
    )


def preprocess_train_dataset(train_dataset, tokenizer, model_card, max_length, stride):
    print("tokenizing...")
    tokenize_dataset = tokenize(
        train_dataset, tokenizer, model_card, max_length, stride
    )
    print("fine answer in each feauture...")
    train_samples = find_answer_in_tokenize_sample(
        tokenize_dataset, train_dataset["answers"]
    )

    train_samples.pop("offset_mapping")
    train_samples.pop("overflow_to_sample_mapping")
    return train_samples


def preprocess_eval_dataset(val_dataset, tokenizer, model_card, max_length, stride):
    print("tokenizing...")
    val_samples = tokenize(val_dataset, tokenizer, model_card, max_length, stride)

    if 'answers' in val_dataset.features: 
        print("fine answer in each feauture...")
        val_samples = find_answer_in_tokenize_sample(
            val_samples, val_dataset["answers"]
        )

    print("processing...")
    sample_map = val_samples["overflow_to_sample_mapping"]
    feature_size = len(val_samples["input_ids"])

    # find sample id for each feature
    sample_ids = val_dataset["id"]
    f = lambda x: sample_ids[sample_map[x]]
    val_samples["sample_ids"] = list(map(f, [i for i in range(feature_size)]))

    # only include context in offset_mapping else set to None
    for i in range(feature_size):
        offset = val_samples["offset_mapping"][i]
        val_samples["offset_mapping"][i] = [
            o if val_samples.sequence_ids(i)[j] == 1 and j not in [0, 1] else None
            for j, o in enumerate(offset)
        ]

    val_samples.pop("overflow_to_sample_mapping")
    return val_samples


def main():
    path = config.c["dataset_path"]

    # create dataset if dataset not exit
    fs = lambda x, y: os.path.join(x, y)
    train_path, dev_path, test_path = (
        fs(path, "train"),
        fs(path, "dev"),
        fs(path, 'test.json'),
    )

    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        print("building squadv2 mini...")
        train_set, dev_set, test_set = build_squadv2_mini()

        print("building completed, saving ...")
        train_set.save_to_disk(train_path)
        dev_set.save_to_disk(dev_path)
        test_set.to_json(test_path)
    else:
        # load
        print("loading squadv2 mini...")
        train_set = load_from_disk(train_path)
        dev_set = load_from_disk(dev_path)

    # download tokenizer
    model_card, max_length, stride = (
        config.c["model_card"],
        config.c["max_length"],
        config.c["stride"],
    )

    tokenizer_path = os.path.join(config.c["model_path"], model_card, "tokenizer")
    
    if not os.path.isfile(os.path.join(tokenizer_path, "tokenizer.json")):
        print("downloading tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(model_card)

        # add special token if tokenizer is gpt2
        if model_card.count("gpt2") != 0:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens(
                {"unk_token": AddedToken("[CLS]", normalized=True, special=True)}
            )
            
        # save tokenizer
        print("saving tokenizer ...")
        tokenizer.save_pretrained(
            os.path.join(config.c["model_path"], model_card, "tokenizer")
        )
    else:
        print("loading tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


    path = config.c['preprocessed_dataset_path']
    save_path = path['dir']
    if not os.path.isdir(save_path):
        print("preprocessing train set...")
        train_dataset = preprocess_train_dataset(
            train_set, tokenizer, model_card, max_length, stride
        )
        print("preprocessing dev set...")
        dev_dataset = preprocess_eval_dataset(
            dev_set, tokenizer, model_card, max_length, stride
        )

        # save tokenized dataset
        print(f"saving to {save_path}...")
        os.makedirs(save_path, exist_ok=True)
        
        train_path, dev_path = os.path.join(save_path, path["train"]), os.path.join(save_path, path["dev"])
        print(train_path)
        print(dev_path)
        with open(train_path, "wb") as f, open(dev_path, "wb") as f_:
            pickle.dump(train_dataset, f)
            print('train set completed')
            
            pickle.dump(dev_dataset, f_)
            print('dev set completed')

if __name__ == "__main__":
    main()

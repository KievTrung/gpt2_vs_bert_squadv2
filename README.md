# GPT2 versus BERT in reading comprehension - SQUADv2
This repo showcase the performance different between gpt2 and bert in term of peforming reading task and extracting information on SQUADv2 dataset. 
To have a bigger picture on model performance, here are the models and their size that we performed evaluation on.
| Model            | Parameters |
|------------------|------------|
| Gpt2             | 124M       |
| Gpt2-medium      | 355M       |
| Bert-base-cased  | 110M       |
| Bert-large-cased | 340M       |

We want to know how good gpt2 perform the task, so we compared between its size and against bert which is an encoder model (in theory, better than gpt2 which is a decoder model). 
We keep the same hyper parameters for all models in training phase.

### How to use:
1. Install requirement package, run:
```
pip install -r requirements.txt
pip install -U datasets
```

2. Since squadv2 only come with **train and dev** set, so i have to combined and redistributed, splited into train, dev, test set [based on __table 2__ of squadv2 paper](https://arxiv.org/pdf/1806.03822). Run `preprocessing.py` to generate squadv2-mini:
```
python -m gpt2_vs_bert_rc_squadv2.preprocessing
```
Folder `/gpt2_vs_bert_rc_squadv2/dataset` will contain all splited dataset after completing.
Folder `/gpt2_vs_bert_rc_squadv2/model` will contain tokenizer and model weight tensors, configs

3. For fine-tune, run: 
```
python -m gpt2_vs_bert_rc_squadv2.fine_tuning
```
To adjusting: epoch `-e`, learning rate `-le`, scheduler `-s`, warm up `-w`, max_grad_norm `-mgn`. Or leave it as default.

4. For inferencing\evaluating, run:
```
python -m gpt2_vs_bert_rc_squadv2.inferencing 
```
This will run evaluating on `/gpt2_vs_bert_rc_squadv2/dataset/test.json` (generated from preprocessing step) \
To run on you own dataset, please format file as `test.json` and add option `--input_dataset ./your_test.json` \
For inferencing, you only need to provide `'id','context','question'` in `test.json`

5. To Use model in your script, please follow instruction in `demo.py`

### Get fine-tune model:
[Go here](https://drive.google.com/drive/folders/1-v5A0QOJHx0NdAzIHdauJZGCKbF6gXts?usp=sharing) to get fine-tune model for both `bert-large-cased` and `gpt2-medium` \
After downloaded, put both folder in `/gpt2_vs_bert_rc_squadv2/model/` \
Then, go to `config.py` and adjust:
```
model_card: #either `bert-large-cased` or `gpt2-medium`
'max_length':318
'stride':128
```
You can ready for inferencing or evaluating.


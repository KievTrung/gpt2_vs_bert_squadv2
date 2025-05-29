# GPT2 versus BERT in reading comprehension - SQUADv2
This repo showcase the performance different between gpt2 and bert in term of peforming reading task and extracting information on SQUADv2 dataset. i used **gpt2-medium** (355M) and **bert-large-cased** (340M). 
The package can also be adjust for different size of gpt2 and bert, **i had only tested on __small__ and __medium__ size** of the 2. 

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


import torch
import os

# global vars
package = './gpt2_vs_bert_rc_squadv2'
model_card = 'gpt2-medium'
# model_card = 'bert-large-cased'
# model_card = 'gpt2'
# model_card = 'bert-base-cased'
c = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dataset_path': os.path.join(package, 'dataset'),
    'model_path': os.path.join(package, 'model'), # /model-card
    'result_path': os.path.join(package, 'result'),
    'preprocessed_dataset_path': {
        'dir': os.path.join(package, f'dataset/preprocessed_dataset_{model_card}'),
        'train':'train.pkl',
        'dev':'dev.pkl',
        },
    'squadv2_config': {
        'train_s': 130319, #sample size
        'neg_train_s': 43498,
        'dev_s': 11873,
        'neg_dev_s': 5945,
        'test_s': 8862,
        'neg_test_s': 4332
    },
    'model_card': model_card,
    'max_length':318,
    'stride':128,
}
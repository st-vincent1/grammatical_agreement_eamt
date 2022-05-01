import pandas as pd
import numpy as np
import sentencepiece as spm
import os

from src.prep.utils import read_from_file
from src.models import Attributes

"""
Does the following:
1. Balances training data by removing most cases with context ",,<singular>,<informal>"
2. Creates amb test set
3. Creates dev and test sets for finetuning

"""


def encode_and_save(data, col, name):
    s = spm.SentencePieceProcessor(model_file='data/sentencepiece/spm.model')
    c = data[col].tolist()
    c_spm = s.encode(c, out_type='str', nbest_size=-1, alpha=0.5)
    c_spm = [' '.join(x) for x in c_spm]
    with open(f'data/finetune/en-pl.{name}.{col}', 'w+') as f:
        for line in c_spm:
            f.write(line + '\n')


def build_dev_and_test_for_finetuning() -> None:
    data = {}
    for split_set in ['dev', 'test']:
        data[split_set] = read_from_file(f'data/raw/en-pl.{split_set}', config='finetune')

    def build_balanced_set(data):
        attribs = Attributes()
        groups = {
            r'<sp:masculine>,[^,]*,[^,]*,[^,]*': 800,
            r'<sp:feminine>,[^,]*,[^,]*,[^,]*': 800,
            r'.*,<il:feminine>,<plural>,<informal>': 200,
            r'.*,<il:masculine>,<plural>,<informal>': 200,
            r'.*,,<plural>,<informal>': 200,
            r'.*,,<singular>,<informal>': 200,
            r'.*,<il:feminine>,<singular>,<informal>': 200,
            r'.*,<il:masculine>,<singular>,<informal>': 200,
            r'.*,<il:feminine>,<plural>,<formal>': 200,
            r'.*,<il:masculine>,<plural>,<formal>': 200,
            r'.*,<il:mixed>,<plural>,<formal>': 200,
            r'.*,<il:feminine>,<singular>,<formal>': 200,
            r'.*,<il:masculine>,<singular>,<formal>': 200,
            r',,,<formal>': 200,
        }
        balanced_set = pd.DataFrame(columns=['en', 'pl', 'cxt', 'marking'])
        temporary_group_df = pd.DataFrame(columns=['en', 'pl', 'cxt'])
        for pattern, count in groups.items():
            matched_rows = data[data.cxt.str.contains(pattern)].sample(frac=1, random_state=1)[
                           :count]
            temporary_group_df = pd.concat((temporary_group_df, matched_rows))

        for attr_type in attribs.type_list:
            matched_rows = temporary_group_df[temporary_group_df.cxt.str.contains(attr_type)]
            matched_rows['marking'] = attr_type
            balanced_set = pd.concat((balanced_set, matched_rows))
        return balanced_set

    for split_set in ['dev', 'test']:
        balanced_data = build_balanced_set(data[split_set])
        for col in ['pl', 'en', 'cxt']:
            encode_and_save(balanced_data, col, split_set)


def balance_training_data_for_finetuning():
    data = read_from_file(f'data/raw/en-pl.train', config='finetune')
    data['singular-informal'] = data.cxt.str.contains[',,<singular>,<informal>']
    leftover_samples = data[data['singular-informal']].sample(frac=0.05)
    balanced_training_data = pd.concat((
        data[data['singular-informal'] == False],
        leftover_samples
    ))
    for col in ['pl', 'en', 'cxt']:
        encode_and_save(balanced_training_data, col, 'train')


def create_amb_test_set() -> None:
    np.random.seed(1)
    data = read_from_file('data/raw/en-pl.test', config='finetune')
    data = data.loc[data.en.str.count(' ').add(1) > 5]

    test_set = data.loc[data.cxt == ',,,'].sample(n=1000)

    encode_and_save(test_set, 'en', 'amb'), encode_and_save(test_set, 'pl', 'amb')


if __name__ == '__main__':
    if not os.path.exists(f'data/finetune/en-pl.amb.pl'):
        create_amb_test_set()
    if not os.path.exists('data/finetune/en-pl.test.pl') or not os.path.exists('data/finetune/en-pl.dev.pl'):
        build_dev_and_test_for_finetuning()
    if not os.path.exists('data/finetune/en-pl.train.pl'):
        balance_training_data_for_finetuning()
import re

import numpy as np
import pandas as pd
from tqdm import tqdm


def preprocess(df):
    # Remove empty sentences
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(subset=['en', 'pl'], inplace=True)

    # Randomise row order
    df = df.sample(frac=1)
    # Preprocess data: 100 tokens max (to fit model)
    df['en_len'] = df['en'].str.count(' ').astype('int64')
    df['pl_len'] = df['pl'].str.count(' ').astype('int64')
    df = df.query('en_len < 100 & pl_len < 100')
    return df[['en', 'pl', 'cxt']]


def make_df(in_en, in_pl):
    en = pd.DataFrame(in_en, dtype="string")
    pl = pd.DataFrame(in_pl, dtype="string")
    df = pd.concat([en, pl], axis=1, join='outer', ignore_index=True)
    df.columns = ['en', 'pl']
    return df

def prepare_coherence_dev(data):
    """ Extract dev and test sets that satisfy the following constraints:
    - 996 sentence pairs per phenomenon
    - within that, e.g. 498 is spgen_m and 498 is spgen_f
    - each marks at least two phenomena

    Procedure:
    - find 498 with that phenomenon + at least one more
    - find the rest at random
    
    16 MAR CHANGES 
    - all of the above needs to hold, BUT, I also need to find sentences which satisfy the following constraints:
    - spgen: verb, noun, adj
    - ilgen: verb, noun, adjective, pronoun
    - ilnum: verb, pronoun
    - formality: verb, pronoun, honorific, request

    Additionally, sentences cannot be shorter than 5 tokens on both sides.

    The order is: request > honorific > noun > adjective > pronoun > verb

    27 April
    1. Dev/test data is now created differently:
    - still use marking for each phenomenon, but we obtain data by taking 200 examples each from each group
    - could use e.g. 600 for fem/masc speaker

    """

    groups = [
        ('spgen_m'),
        ('spgen_f'),
        ('ilgen_f', 'ilnum_p', 'form_i'),
        ('ilgen_m', 'ilnum_p', 'form_i'),
        ('ilnum_p', 'form_i'),
        ('ilgen_f', 'ilnum_s', 'form_i'),
        ('ilgen_m', 'ilnum_s', 'form_i'),
        ('ilgen_f', 'ilnum_p', 'form_f'),
        ('ilgen_m', 'ilnum_p', 'form_f'),
        ('ilgen_x', 'ilnum_p', 'form_f'),
        ('ilgen_f', 'ilnum_s', 'form_f'),
        ('ilgen_m', 'ilnum_s', 'form_f'),
        ('form_f')
    ]


    leftover_train = pd.DataFrame(columns=['en', 'pl', 'cxt'])
    dev = pd.DataFrame(columns=['en', 'pl', 'cxt'])
    test = pd.DataFrame(columns=['en', 'pl', 'cxt'])
    quants = {
        'ilgen': 332,
        'ilnum': 498,
        'spgen': 498,
        'form': 498
    }
    data[SETS] = data['cxt'].str.split(',', expand=True)
    data.replace('', np.nan, inplace=True)
    print(data)
    data['at_least_2'] = data[SETS].isna().sum(axis=1) <= 2
    for type_ in ['ilgen_x', 'spgen_m', 'spgen_f', 'ilgen_m', 'ilgen_f', 'ilnum_s', 'ilnum_p', 'form_i', 'form_f']:
        set_ = type_.split('_')[0]
        type_sents = data[(data[set_] == type_) & (data['at_least_2'])]
        type_sents['marking'] = type_
        leftover_train_, dev_, test_ = np.split(type_sents.sample(frac=1, random_state=1),
                                                [len(type_sents) - 2 * quants[set_], len(type_sents) - quants[set_]])
        leftover_train = pd.concat((leftover_train, leftover_train_))
        dev = pd.concat((dev, dev_))
        test = pd.concat((test, test_))
        data = pd.concat([data, type_sents, type_sents]).drop_duplicates(keep=False)
        # pick #quants of that phenomenon with at least 2, remove these rows from data, add to dev and test
        # print(len(type_sents))

    # with pd.option_context('display.max_rows', None, 'display.max_columns',
    #                        None):  # more options can be specified also
    #     print(dev)
    # print(test)
    # print(leftover_train)
    return leftover_train, dev, test


def save_leftovers(train, leftover_train, dev, test, prefix):

    train.append(leftover_train)
    train[SETS] = train['cxt'].str.split(',', expand=True)
    print(f'{train.shape=}')
    print(train)
    # Getting rid of some of the more popular scenarios because we don't need that many examples!
    train['ilnum_s_and_form_i'] = (train['ilnum'] == 'ilnum_s') & (train['form'] == 'form_i') & \
                                  (train['spgen'] == '') & (train['ilgen'] == '')

    print(f"{train[train['ilnum'] == 'ilnum_s'].shape=}")
    print(f"{train[train['form'] == 'form_i'].shape=}")
    print(f"{train[(train['form'] == 'form_i') & (train['ilnum'] == 'ilnum_s')].shape=}")

    # Removing all but 5% of ilnum_s and form_i
    train_ = train[train['ilnum_s_and_form_i']].sample(frac=0.05)
    train = pd.concat((train[train['ilnum_s_and_form_i'] == False], train_))
    print(f'{train.shape = }')
    print(f"{train[train['ilnum'] != ''].shape=}")
    print(f"{train[train['form'] != ''].shape=}")
    print(f"{train[train['spgen'] != ''].shape=}")
    print(f"{train[train['ilgen'] != ''].shape=}")


    dfs = {f'{prefix}.train.en': train['en'],
           f'{prefix}.train.pl': train['pl'],
           f'{prefix}.train.cxt': train['cxt'],
           f'{prefix}.dev.en': dev['en'],
           f'{prefix}.dev.pl': dev['pl'],
           f'{prefix}.dev.cxt': dev['cxt'],
           f'{prefix}.dev.marking': dev['marking'],
           f'{prefix}.test.en': test['en'],
           f'{prefix}.test.pl': test['pl'],
           f'{prefix}.test.cxt': test['cxt'],
           f'{prefix}.test.marking': test['marking']}
    return dfs


def train_spm(sentence_list: List[str]) -> None:
    train_data = 'data/sentencepiece/spm.training_data'
    with open('data/sentencepiece/spm.training_data', 'w+') as f:
        for line in sentence_list:
            f.write(line + "\n")
    spm.SentencePieceTrainer.train(input=train_data,
                                   model_prefix=f'data/sentencepiece/spm', vocab_size=32000,
                                   character_coverage=0.9995,
                                   input_sentence_size=len(sentence_list),
                                   shuffle_input_sentence=True)


def read_from_file(filename):
    with open(filename + '.en', 'r') as s, open(filename + '.pl', 'r') as t, open(filename + '.cxt', 'r') as c:
        raw_s = s.read().splitlines()
        raw_t = t.read().splitlines()
        raw_c = c.read().splitlines()
        df = pd.concat([pd.DataFrame(raw_s),
                        pd.DataFrame(raw_t),
                        pd.DataFrame(raw_c)], axis=1, join='outer', ignore_index=True)
        df.columns = ['en', 'pl', 'cxt']
    return df


def write_to_files(dfs, prefix=''):
    for k, v in dfs.items():
        with open(f'./data/qualitative/{prefix}{k}', 'w+') as f:
            f.writelines('\n'.join(v.tolist()))


def write_to_file(contents, filename):
    with open(filename + '.en', 'w+') as s, open(filename + '.pl', 'w+') as t:
        s.writelines('\n'.join(contents['en'].tolist()))
        t.writelines('\n'.join(contents['pl'].tolist()))

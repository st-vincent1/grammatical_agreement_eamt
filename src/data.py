import copy
import itertools
import logging
import pickle
import re
import random
import os

import numpy as np
import pandas as pd
import torch
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.vocab import build_vocab_from_iterator
from src.models import Attributes
from src.utils import CPU_Unpickler



def context_tokenize(x):
    return [a for a in x.split(',') if a]


class DataManager(object):
    def __init__(self, batch_size, device):
        self.text = Field(tokenize=lambda x: x.split(),
                          init_token='<sos>',
                          eos_token='<eos>',
                          lower=False,
                          batch_first=True)

        self.context = Field(tokenize=context_tokenize,
                             sequential=True,
                             use_vocab=True,
                             batch_first=True)

        self.fields = [('en', self.text), ('pl', self.text), ('cxt', self.context)]
        self.test_fields = self.fields + [('marking', self.context)]

        self.dataiter_fn = lambda dataset, train: BucketIterator(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=train,
            repeat=False,
            sort_key=lambda x: len(x.pl),
            sort_within_batch=False,
            device=device
        )


# Loading pretraining data
def load_pretraining_data(params):
    print("Loading data")
    batch_size = params.pt.batch_size
    device = torch.device(params.device)
    root = params.pt.data

    data = {}
    for s in ['train', 'dev']:
        df = {}
        for e in ['en', 'pl']:
            filepath = os.path.join(root, f'en-pl.{s}.bpe.{e}')
            with open(filepath) as f:
                df[e] = pd.Series(f.read().splitlines())
        data[s] = pd.concat([df['en'], df['pl']], axis=1)
        data[s].columns = ['en', 'pl']

    TEXT = Field(tokenize=lambda x: x.split(),
                 init_token='<sos>',
                 eos_token='<eos>',
                 lower=False,
                 batch_first=True)
    data_fields = [('en', TEXT), ('pl', TEXT)]

    dataiter_fn = lambda dataset, train: BucketIterator(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
        repeat=train,
        sort_key=lambda x: len(x.pl),
        sort_within_batch=False,
        device=device
    )
    train, dev = dataset_fn(data['train'], data['dev'], root, data_fields)
    tr_iters = dataiter_fn(train, True)
    de_iters = dataiter_fn(dev, False)

    if os.path.exists('data/vocab.pkl'):
        vocab = read_vocab('vocab.pkl')
        TEXT.vocab = vocab
    else:
        TEXT.build_vocab(train, max_size=16000, min_freq=5)
        vocab = TEXT.vocab
        save_vocab('vocab.pkl', vocab)

    params.eos_idx = vocab.stoi['<eos>']
    params.pad_idx = vocab.stoi['<pad>']
    print(f'Loaded vocab: {vocab.itos[9]=} | {vocab.itos[16]=} | {vocab.itos[225]=}')
    return params, tr_iters, de_iters, vocab


def build_context_vocab(term_list: iter, null=False):
    # Vocab expects each token as a list of 1 token otherwise splits into characters
    term_list = [[x] for x in term_list]
    context_vocab = build_vocab_from_iterator(term_list)
    # Remove special tokens as the method above does not let you ...
    del (context_vocab.stoi['<unk>'])
    context_vocab.itos = [x for x in context_vocab.itos if x not in ['<unk>', '<pad>']] + ['<pad>']
    if null:
        context_vocab.itos += ['<null>']
    # Fix stoi as special tokens were at the start
    context_vocab.stoi = {x: context_vocab.stoi[x] - 2 for x in context_vocab.stoi.keys()}
    if null:
        context_vocab.stoi.update({'<pad>': len(context_vocab.itos) - 2, '<null>': len(context_vocab.itos) - 1})
    else:
        context_vocab.stoi.update({'<pad>': len(context_vocab.itos) - 1})
    assert [x == context_vocab.stoi[context_vocab.itos[x]] for x in range(len(context_vocab.itos))]
    return context_vocab


def read_vocab(path, device=None):
    # read vocabulary pkl
    import pickle
    pkl_file = open(f'data/{path}', 'rb')
    vocab = pickle.load(pkl_file) if device != 'cpu' else CPU_Unpickler(pkl_file).load()
    pkl_file.close()
    return vocab


def save_vocab(path, vocab):
    import pickle
    print(f"Saving vocab to file {path}. . .")
    output = open(f'./data/{path}', 'wb+')
    pickle.dump(vocab, output)
    output.close()
    print(f"Saved.")
    return


def add_tags_to_vocab(vocab, new_terms):
    old_vocab_len = len(vocab)
    vocab.stoi.update({
        new_terms[k]: old_vocab_len + k for k in range(len(new_terms))
    })
    vocab.itos = vocab.itos + new_terms
    assert len(vocab) == old_vocab_len + len(new_terms)
    return vocab


def get_tags(row):
    """ Get tags from cxt row, shuffle them, pad with nulls"""
    types = row['cxt'].split(',')
    tags = [t_ for t_ in types if t_]
    random.shuffle(tags)
    finishing_tags = ['<pad>' for t_ in types if t_ == ''] + ['<null>']
    return ' '.join(tags), ' '.join(tags + finishing_tags)


def add_tags(df, tag_enc, tag_dec, splits):
    tags_enc = {x: [] for x in splits}
    tags_dec = {x: [] for x in splits}
    logging.info("Adding tags...")
    for split in splits:
        for idx, row in df[split].iterrows():
            tags_ = get_tags(row)
            tags_enc[split].append(tags_[0])
            if tag_dec: tags_dec[split].append(tags_[1])

    tags_enc = {x: pd.Series(tags_enc[x]) for x in tags_enc.keys()}
    tags_dec = {x: pd.Series(tags_dec[x]) for x in tags_dec.keys()}

    for split in splits:
        if tag_enc:
            df[split]['en'] = df[split]['en'] + ' ' + tags_enc[split]
            df[split]['en'] = df[split]['en'].str.strip()

        if tag_dec:
            df[split]['pl'] = tags_dec[split] + ' ' + df[split]['pl']
            df[split]['pl'] = df[split]['pl'].str.strip()
            # To make sure that they're shuffled in the same way!
            df[split]['cxt'] = tags_dec[split].replace(' ', ',', regex=True)
    return df


def count_words(df):
    l = (df['en'].str.count('▁').sum() + df['pl'].str.count('▁').sum()) / 2
    print(f"Total words in en and pl averaged: {l}.")


def make_df(en, pl, cxt, split):
    df = pd.concat([en, pl, cxt], axis=1, join='outer', ignore_index=True)
    df.columns = ['en', 'pl', 'cxt']
    # Adding extra ambivalent sentences to the set
    df[Attributes().attribute_list] = df['cxt'].str.split(',', expand=True)
    df.replace('', np.nan)
    if split == 'train':
        unann_count = df.query("cxt != ',,,'").shape[0] // 4  # get 25% of size of annotated corpus
        print(f'{unann_count = }')
        extra_unann = df.query("cxt == ',,,'").sample(frac=1)[:unann_count]  # grab unann_count sentences from unann
        df = pd.concat((df.query("cxt != ',,,'"), extra_unann), ignore_index=True)  # add them to the dataframe
    return df


def load_files_into_dfs(params, sets):
    root = params.ft.data
    dfs = {}
    elm = ['en', 'pl', 'cxt']

    for s, e in itertools.product(sets, elm):
        filename = f'en-pl.{s}.bpe.{e}'
        filepath = os.path.join(root, filename)
        df = pd.read_csv(filepath, sep='\t', index_col=None, header=None, skip_blank_lines=False)
        dfs[f'{s}.{e}'] = df
    return {f'{s}': make_df(dfs[f'{s}.en'], dfs[f'{s}.pl'], dfs[f'{s}.cxt'], s) for s in sets}


def add_labels(dfs: dict, cxt_vocab, seed, alpha: float) -> dict:
    attrs = Attributes()
    # Change seed to make sure re-labelling is different, but still dependent on the original seed so it is reproducible
    np.random.seed(seed)

    # For alpha amb, we pick a subset of random attributes
    def labels_alpha_amb(length):
        # todo print what this does
        contexts = list(zip(
            *(list(np.random.choice(types + [''] * len(types), length))
              for types in attrs.types.values())))
        return [','.join(cs) for cs in contexts]

    # Turn empty context into nans
    dfs['train'].replace({'': np.nan}, inplace=True)
    dfs['dev'].replace({'': np.nan}, inplace=True)

    non_empty_change = dfs['train'].loc[dfs['train']['cxt'] != ',,,'].sample(frac=alpha).index
    empty_change = dfs['train'].loc[dfs['train']['cxt'] == ',,,'].sample(frac=alpha).index

    def drop_random_vals(x):
        y = x.split(',')
        z = [random.choice([i, '']) for i in y]
        return ','.join(z)

    dfs['train'].loc[non_empty_change, 'cxt'] = dfs['train'].loc[non_empty_change, 'cxt'].apply(
        lambda x: drop_random_vals(x))
    dfs['train'].loc[empty_change, 'cxt'] = labels_alpha_amb(len(empty_change))

    dfs['train'] = dfs['train'].replace({'': np.nan})
    return dfs


def add_labels_test(dfs: dict) -> dict:
    attrs = Attributes()

    # Reset columns in marking; not np.nan because of join used later
    dfs['isolated'][attrs.attribute_list] = ''

    for att in attrs.attribute_list:
        for type_ in attrs.types[att]:
            dfs['isolated'].loc[dfs['isolated']['marking'] == type_, att] = type_

        for setting in ['full', 'isolated']:
            # Gather context from attributes
            dfs[setting]['cxt'] = dfs[setting][attrs.attribute_list].agg(','.join, axis=1)

    for set_ in dfs.keys():
        dfs[set_] = dfs[set_].replace({'': np.nan})
        dfs[set_] = dfs[set_].reset_index(drop=True)
    return dfs


def dataset_fn(train, dev, root, fields, prefix='load'):
    count_words(train)
    count_words(dev)
    train.to_csv(os.path.join(root, f"{prefix}_train.csv"), index=False, header=None)
    dev.to_csv(os.path.join(root, f"{prefix}_dev.csv"), index=False, header=None)
    train, dev = TabularDataset.splits(
        path=root,
        train=f'{prefix}_train.csv',
        validation=f'{prefix}_dev.csv',
        format='csv',
        fields=fields)
    return train, dev


def load_finetuning_data(params, vocab, context_vocab):
    print(f"Loading {params.config}")
    data_manager = DataManager(params.ft.batch_size, torch.device(params.device))
    root = params.paths.data

    loaded_dfs = load_files_into_dfs(params, ['train', 'dev'])

    labelled_dfs = add_labels(loaded_dfs, context_vocab, params.seed, params.alpha)

    # Add tags to source/target sentences if config requires it
    if 'tag' in params.config:
        labelled_dfs = add_tags(labelled_dfs, 'enc' in params.config, 'dec' in params.config, ['train', 'dev'])

    labelled_dfs['train'] = labelled_dfs['train'][['en', 'pl', 'cxt']]
    labelled_dfs['dev'] = labelled_dfs['dev'][['en', 'pl', 'cxt']]

    data_manager.context.vocab = context_vocab

    train, dev = dataset_fn(labelled_dfs['train'], labelled_dfs['dev'], root, data_manager.fields, params.config)
    train_iters = data_manager.dataiter_fn(train, True)
    dev_iters = data_manager.dataiter_fn(dev, False)

    data_manager.text.vocab = vocab
    params.eos_idx = vocab.stoi['<eos>']
    params.pad_idx = vocab.stoi['<pad>']

    print(f'Loaded vocab: {vocab.itos[9]=} | {vocab.itos[16]=} | {vocab.itos[225]=}')

    if 'tag' not in params.config:
        print(f'Context vocab: {context_vocab.itos}\n{context_vocab.stoi}')
    
    return params, train_iters, dev_iters, vocab


def add_random_labels(df, fill_marking=False):
    """
    Adds 5 columns to df: ['SpGender'..] and 'cxt'. Attributes are random.
    :param df: dataframe of 2 columns: en, pl
    :return: same dataframe with 5 extra columns
    """
    attrs = Attributes()
    n = len(df)
    context = []
    for i in range(n):
        types = []
        for att in attrs.attribute_list:
            types.append(random.choice(attrs.types[att]))
        context.append(','.join(types))
    df['cxt'] = pd.Series(context)
    df[attrs.attribute_list] = df['cxt'].str.split(',', expand=True)
    if fill_marking:
        df['marking'] = ''
    df.replace('', np.nan)
    print(df)
    return df


def reverse_labels(df):
    attrs = Attributes()
    df_ = copy.deepcopy(df)
    for att in attrs.attribute_list:
        df_[att] = df_[att].apply(lambda x: attrs.reverse_map[x])
    df_['cxt'] = df_[attrs.attribute_list].agg(','.join, axis=1)
    return df_


def load_test(params, vocab, context_vocab):
    data_manager = DataManager(params.ft.batch_size, torch.device(params.device))
    data_manager.context.vocab = context_vocab
    attrs = Attributes()

    root = params.ft.data

    # Load files, assuming "marking" has context only for the relevant attribute
    fields = ['en', 'pl', 'cxt', 'marking']
    data = {}
    for e in fields:
        filename = f'en-pl.test.bpe.{e}'
        filepath = os.path.join(root, filename)
        df = pd.read_csv(filepath, sep='\t', index_col=None, header=None, skip_blank_lines=False)
        data[e] = df

    data = pd.concat([data['en'], data['pl'], data['cxt'], data['marking']],
                     axis=1,
                     join='outer',
                     ignore_index=True)
    data.columns = ['en', 'pl', 'cxt', 'marking']
    data[attrs.attribute_list] = data['cxt'].str.split(',', expand=True)
    data.replace('', np.nan)

    dfs = {
        'full': data,
        'isolated': copy.deepcopy(data)
    }

    """ Loading amb data. """
    amb = {}
    for suffix in ['en', 'pl']:
        filename = f'en-pl.amb.bpe.{suffix}'
        filepath = os.path.join(root, filename)
        df = pd.read_csv(filepath, sep='\t', index_col=None, header=None, skip_blank_lines=False)
        amb[suffix] = df

    dfs['amb'] = pd.concat([amb['en'], amb['pl']], axis=1, join='outer', ignore_index=True)
    dfs['amb'].columns = ['en', 'pl']
    dfs['amb'] = add_random_labels(dfs['amb'], fill_marking=True)
    dfs['amb_rev'] = reverse_labels(copy.deepcopy(dfs['amb']))

    # Adds labels according to set.
    labelled = add_labels_test(dfs)

    count_words(labelled['full'])

    """Adding tags if required - should be done at the very end though, when phens are filled in everywhere."""
    if 'tag' in params.config:
        labelled = add_tags(labelled, 'enc' in params.config, 'dec' in params.config, labelled.keys())

    def prepare_test_iter(x):
        x.to_csv(os.path.join(root, "test.csv"), index=False)
        test = TabularDataset(
            path=os.path.join(root, 'test.csv'),
            skip_header=True,
            format='csv',
            fields=data_manager.test_fields)
        return data_manager.dataiter_fn(test, False)

    iters = {
        name: prepare_test_iter(labelled[name]) for name in dfs.keys()
    }

    data_manager.text.vocab = vocab
    params.eos_idx = vocab.stoi['<eos>']
    params.pad_idx = vocab.stoi['<pad>']
    print(f'Loaded vocab: {vocab.itos[9]=} | {vocab.itos[16]=} | {vocab.itos[225]=}')
    return params, iters, vocab

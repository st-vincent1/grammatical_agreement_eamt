import os
import argparse
import numpy as np
import pandas as pd
import sentencepiece as spm

from utils import read_from_file, preprocess, save_leftovers, write_to_files, prepare_coherence_dev, train_spm


def read_data():
    train = read_from_file('data/en-pl.train')
    dev = read_from_file('data/en-pl.dev')
    test = read_from_file('data/en-pl.test')
    return train, dev, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    np.random.seed(1)

    train, dev, test = read_data()

    # Train SPM model if not trained yet
    if not os.exists('data/sentencepiece/spm.model'):
        train_spm(train['en'] + train['pl'])
    s = spm.SentencePieceProcessor(model_file='data/sentencepiece/spm.model')


    def spm_encode(df):
        def encode_column(col):
            c = df[col].tolist()
            c_spm = s.encode(c, out_type='str', nbest_size=-1, alpha=0.5)
            c_spm = [' '.join(x) for x in c_spm]
            return c_spm

        en_spm, pl_spm = encode_column('en'), encode_column('pl')
        new_df = pd.concat([pd.DataFrame(en_spm), pd.DataFrame(pl_spm)],
                           axis=1,
                           join='outer',
                           ignore_index=True)
        new_df.columns = ['en', 'pl']
        return new_df


    train_spm, dev_spm, test_spm = map(lambda x: spm_encode(x), (train, dev, test))
    print("Preprocessing.")
    train, dev, test = (preprocess(x) for x in (train_spm, dev_spm, test_spm))

    # leftover_train, dev_, test_ = prepare_coherence_dev(test)
    # os = save_leftovers(train, leftover_train, dev_, test_, 'os')

import pandas as pd
import numpy as np
import sentencepiece as spm

np.random.seed(1)
data = {}
for x in ['en', 'pl', 'context']:
    df = pd.read_csv(f'data/working/upgrade.test.{x}', sep='delimiter', index_col=None, header=None, skip_blank_lines=False)
    data[x] = df

data = pd.concat([data['en'], data['pl'], data['context']], axis=1, join='outer', ignore_index=True)
data.columns = ['en', 'pl', 'context']
data = data.loc[data.src.str.count(' ').add(1) > 5]
test_set = data.loc[data.context == ',,,'].sample(n=1000)

# Encode with spm
s = spm.SentencePieceProcessor(model_file='data/sentencepiece/upgrade_spm.model')

def encode_column(col):
    c = test_set[col].tolist()
    c_spm = s.encode(c, out_type='str', nbest_size=-1, alpha=0.5)
    c_spm = [' '.join(x) for x in c_spm]
    return c_spm

en_spm, pl_spm = encode_column('en'), encode_column('pl')

with open('data/preprocessed/amb.test.en', 'w+') as f:
    for line in en_spm:
        f.write(line + '\n')

with open('data/preprocessed/amb.test.pl', 'w+') as f:
    for line in pl_spm:
        f.write(line + '\n')

import pandas as pd
from pprint import pprint
if __name__ == '__main__':
    df_baseline = pd.read_csv('QUALITATIVE_BASELINE', delimiter='|', header=None)
    df_finetune = pd.read_csv('QUALITATIVE_FINETUNED', delimiter='|', header=None)
    print(df_baseline)
    df_baseline.columns = ['src', 'base_hyp', 'ref', 'base_score']
    df_finetune.columns = ['src',  'ft_hyp', 'ref' , 'cxt', 'ft_score']

    print(df_baseline)
    df = pd.merge(df_baseline, df_finetune, on=['src', 'ref'])
    df['score_difference'] = df['ft_score'] - df['base_score']
    df['src_len'] = df['src'].str.split().apply(len)
    df = df.query('src_len > 10')
    df = df[['src', 'ref', 'cxt', 'base_hyp', 'ft_hyp', 'base_score', 'ft_score', 'score_difference']]
    df = df.sort_values(by=['score_difference'])
    pprint(df[:100])
    df[:150].to_csv('worst_150.csv', index=False)
    pprint(df[-100:])
    df[-150:].to_csv('best_150.csv', index=False)

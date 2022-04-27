"""Need to calculate:
- average of no_label_agr_fem and average std (average variances then std)
- same to no_label_agr_fem
- same to agrs"""

import json

import numpy as np

# read in every file before means and add with_fem_label_dep - no_label_dep
res = {
}

for method in ['tag_enc', 'tag_enc_dec', 'tag_dec', 'embedding', 'emb_bias', 'emb_sum']:
    for data_type in ['alpha']:
        with open(f'out/{method}_{data_type}.json') as json_file:
            data = json.load(json_file)
            f = data['with_fem_label_agr'] / (np.add(data['with_fem_label_agr'], data['with_fem_label_masc_agr'])) * 100
            m = data['with_masc_label_agr'] / (
                np.add(data['with_masc_label_agr'], data['with_masc_label_fem_agr'])) * 100
            fm = np.concatenate((f, m))
            if method not in res.keys():
                res[method] = fm
            else:
                res[method] += fm
print(res)
for key in res.keys():
    res[key] = (np.mean(res[key]), np.std(res[key]))

with open('out/agreement_diff.result', 'w+') as f:
    json.dump(res, f, indent=4)

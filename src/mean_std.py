import argparse
import json
import sys

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--filename')

args = parser.parse_args()
filename = args.filename
try:
    with open(f'out/{filename}') as json_file:
        data = json.load(json_file)
except FileNotFoundError:
    print("File does not exist.")
    sys.exit(0)

#for key in data.keys():
#    data[key] = data[key][-3:]
# Needed for results table
data['agreement'] = []
data['iso_agreement'] = []
data['bias_overall'] = []
for key in data.keys():
    if 'agreement_' in key and 'iso' not in key:
        data['agreement'] = data['agreement'] + data[key]
    if 'iso_agreement_' in key:
        data['iso_agreement'] = data['iso_agreement'] + data[key]
    if '_bias' in key:
        data['bias_overall'] = data['bias_overall'] + data[key]

for key in data.keys():
    data[key] = (np.mean(data[key]), 2*np.std(data[key]))

with open(f'out/{filename[:-5]}_means.json', 'w+') as f:
    json.dump(data, f, indent=4)

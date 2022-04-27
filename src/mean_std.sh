#!/bin/bash

for setting in emb_enc emb_dec tag_enc_dec tag_enc tag_dec out_bias emb_pw_sum emb_enc_dec emb_add; do
  python mean_std.py --filename ${setting}_biases_results.json
done

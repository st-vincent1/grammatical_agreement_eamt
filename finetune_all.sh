#!/bin/bash

set -e

for CONFIG in emb_enc emb_sos emb_enc_sos tag_enc tag_dec tag_enc_dec out_bias emb_pw_sum emb_add; do
  python finetune.py --config $CONFIG
done
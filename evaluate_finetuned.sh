#!/bin/bash

# run evaluation script to get outputs
for CONFIG in emb_enc emb_enc_dec emb_dec tag_enc tag_enc_dec tag_dec out_bias emb_add emb_pw_sum; do
  python evaluate_finetuned.py --config ${CONFIG} --print-hypotheses out/finetuned/en-pl.${CONFIG}.out --print-references out/finetuned/en-pl.test.ref
done

# Paired bootstrap resampling: only run after evaluation was finished and you checked which system is performing best with chrF++

sacrebleu out/finetuned/en-pl.test.ref -i out/finetuned/en-pl.tag_enc.out out/finetuned/en-pl.*.out -m bleu chrf --chrf-word-order 2 --paired-bs

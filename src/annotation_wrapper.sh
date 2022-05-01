#!/bin/bash

# Split data into groups of 2M, annotate, concatenate contexts
PREFIX=data/finetune/en-pl.train.chunk
for LANG in en pl; do
	split -l 2000000 -d data/pretrain/en-pl.train.${LANG} ${PREFIX}_
	for chunk in {00..05}; do
		mv ${PREFIX}_${chunk} ${PREFIX}_${chunk}.${LANG}
	done
done
cp data/raw/en-pl.{dev,test}.* data/finetune/

# Pause for annotating with parallel jobs

# Concatenate results
#cp data/pretrain/en-pl.train.* data/finetune/
#
#OUT=data/finetune/en-pl.train.cxt
#> $OUT
#for chunk in {00..05}; do
#	cat ${PREFIX}_${chunk}.cxt >> $OUT
#done
#
## Remove chunks
#rm data/finetune/*chunk*

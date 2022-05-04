#!/bin/bash

# When no prefix is given, annotates dev and test by default
python src/prep/annotate.py

# ANNOTATING TRAINING DATA
# --> IF PARALLEL PROCESSING IS POSSIBLE, USE THE FOLLOWING SCRIPT FOR QUICKER ANNOTATION OF TRAINING DATA
#PREFIX=data/raw/en-pl.train.chunk


# 1. Split data into groups of 2M, annotate, concatenate contexts

#for LANG in en pl; do
#	split -l 2000000 -d data/pretrain/en-pl.train.${LANG} ${PREFIX}_
#	for chunk in {00..05}; do
#		mv ${PREFIX}_${chunk} ${PREFIX}_${chunk}.${LANG}
#	done
#done

# 2. Pause for annotating with parallel jobs (run python src/prep/annotate.py for separate chunks on separate processes)

# 3. Concatenate results

# OUT=data/raw/en-pl.train.cxt
#> $OUT
#for chunk in {00..05}; do
#	cat ${PREFIX}_${chunk}.cxt >> $OUT
#done

# 4. Remove chunks
#rm data/raw/*chunk*


# --> IF PARALLEL PROCESSING IS NOT POSSIBLE, SIMPLY PREPROCESS ALL AT ONCE

python src/prep/annotate.py --prefix data/raw/en-pl.train

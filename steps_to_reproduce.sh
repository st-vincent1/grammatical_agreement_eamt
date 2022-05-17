#!/bin/bash

# Install
# python 3.8.10
# pip install sentencepiece sacrebleu
# pytorch 1.7.1 torchtext 0.8.1
# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 torchtext=0.8.1 -c pytorch

# spacy
# conda install spacy
# spacy model (morfeusz)
# python -m pip install lib/pl_spacy_model_morfeusz_big-0.1.0.tar.gz
# Run prepare_pretraining_data.sh
# Run pretrain.py which pretrains model

# Run annotation wrapper bash src/annotation_wrapper.sh
# Run python -m src.prep.annotate which annotates files from data/raw and moves the result to data/finetune

# Run postprocess_annotated_data.py which prepares eval data and finalises annotation

# Run finetune.py to finetune models
# 6. Once model is pretrained, evaluate it on bleu
# 7. Write a running script to finetune model with each config

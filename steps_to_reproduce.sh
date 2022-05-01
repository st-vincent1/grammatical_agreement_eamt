#!/bin/bash

# Install
# sentencepiece
# pytorch 1.7.1
# torchtext 0.8.1
# spacy
# spacy model (morfeusz)

# Run prepare_pretraining_data.sh
# Run pretrain.py which pretrains model

# Run python -m src.prep.annotate which annotates files from data/raw and moves the result to data/finetune

# Run finetune.py to finetune models
# 6. Once model is pretrained, evaluate it on bleu
# 7. Write a running script to finetune model with each config

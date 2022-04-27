#!/bin/bash


# 1. prepare data (in eamt_data_prep) [this includes bicleaner]
# 2. Copy data here, to data/raw
# 3. run src/data_prep/encode_and_split.py which
#   a. trains spm
#   b. encodes data with spm
#   c. preprocesses it (removes sequences that won't fit into the model)
# 4. Run pretrain.py which pretrains model
# 5. In the meantime, run src/data_prep/annotate_corpus.py which annotates the corpus (non-spm'd of course)
#     this makes copies of train dev test data into a data/finetuning folder
#     then produces context files in that folder
#     finally, run src/data_prep/encode_and_split.py which will encode the files ready for finetuning
# 6. Once model is pretrained, evaluate it on bleu
# 7. Write a running script to finetune model with each config
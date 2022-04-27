# Controlling the gender of the speaker (SpGen): code

## Description
This repository contains code associated with the publication ... .
## Table of Contents

## Installation

### Installation instructions

1. Clone this repository.
2. Navigate to the repository.
3. Install requirements by running `pip install requirements.txt` (recommended in a sterile environment)
4. Download data and models (optional).
5. To pretrain/train models, run `python pretrain.py [-f] [-g]` or `python train.py`.
6. To evaluate downloaded models against downloaded test sets, run `python evaluate.py`.

### Downloading data and models

1. Clone and navigate to the repository.
    - To download the models, run `bash download_models.sh`.
    - To download all data (training and test), run `bash download_data.sh -all`.
    - To download just test data for reproducing results, run `bash download_data.sh -test`.
3. Run `python test_dl.py` to make sure that the files have downloaded correctly.

### Requirements

Full requirements can be found in `requirements.txt` but are satisfied by installing:

```python==3.8.10
pytorch==1.7.1
torchtext==0.8.1
sentencepiece
sacrebleu==1.5.1
cudatoolkit==10.2
torchvision==0.8.2
pandas==1.2.4
spacy==2.2.4
keras-gpu==2.3.1
tensorflow-gpu==2.2.0
```
To run evaluation using Morfeusz2 & Spacy (todo add link here), the spacy model must also be installed. 
(todo add instructions)

## Results

All tests done on a test set of 3 000 sentences, comprising of a 1 000 female-speaking, a 1 000 male speaking and a 1
000 ambiguous sentences.

General model performance (on mf_test):

| Model       | BLEU        | chrF++      | Accuracy    |
| ----------- | ----------- | ----------- | ----------- |
| Baseline    | Title       | Title       | Title       |
| Transformer    | Title       | Title       | Title       |

### Training

- Mixed precision training as
  per [PyTorch's guidelines](https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples)
- Adam optimizer with fixed learning rate (no established value yet) & fixed weight decay
- batch size 128 but simulated to be 4096 as shown beneficial (cite paper)

## Reproducing results

To reproduce results reported in the paper:

1. Download the models and test data as per instructions above.
2. Run ```python evaluate.py Transformer.pt```

## Contributing

If you find a bug or wish to request a feature, please submit
an [https://github.com/st-vincent1/Transformer/issues/new/choose](Issue).

## Credits

## License
# Installation instructions

First, clone the project:

`git clone https://github.com/st-vincent1/grammatical_agreement_eamt && cd grammatical_agreement_eamt`\

We recommend that the research is reproduced in a new Anaconda environment:

Create a conda environment with Python 3.8.10
`conda create --name grammatical_agreement_eamt python=3.8.10 && conda activate grammatical_agreement_eamt`\

1. Install PyTorch 1.7.1. and related libraries (make sure to specify correct CUDA version)
`conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 torchtext=0.8.1 -c pytorch`\
2. Install dependencies for the detection tool:
    a. `conda install -c conda-forge spacy==2.2.4`
    b. `pip install tensorflow==2.2.0 keras==2.3.1 morfeusz2 sentencepiece sacrebleu mosestokenizer pycld3`
    c. `python -m pip install lib/pl_spacy_model_morfeusz_big-0.1.0.tar.gz`

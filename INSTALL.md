# Installation instructions

First, clone the project:
```
git clone https://github.com/st-vincent1/grammatical_agreement_eamt && cd grammatical_agreement_eamt
```

We recommend that the research is reproduced in a new Anaconda environment:
```
conda create --name grammatical_agreement_eamt python=3.8.10 && conda activate grammatical_agreement_eamt
```

Install PyTorch 1.7.1. and related libraries (make sure to specify correct CUDA version)
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 torchtext=0.8.1 -c pytorch
```

Install dependencies for the detection tool:
```
conda install -c conda-forge spacy==2.2.4
pip install tensorflow==2.2.0 keras==2.3.1 morfeusz2 sentencepiece sacrebleu mosestokenizer pycld3
```
Run `mkdir -p lib/`, download `pl_spacy_model_morfeusz_big-0.1.0.tar.gz` from [the bottom of this list](http://zil.ipipan.waw.pl/SpacyPL?action=AttachFile&do=view&target=pl_spacy_model_morfeusz_big-0.1.0.tar.gz) and put in the `lib/` directory. Then install
```
python -m pip install lib/pl_spacy_model_morfeusz_big-0.1.0.tar.gz
```

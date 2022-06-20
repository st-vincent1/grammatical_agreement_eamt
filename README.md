# Controlling Extra-Textual Attributes about Dialogue Participants: A Case Study of English-to-Polish Neural Machine Translation

## Description
This repository contains code associated with the publication ["Controlling Extra-Textual Attributes about Dialogue Participants: A Case Study of English-to-Polish Neural Machine Translation", EAMT2022](https://aclanthology.org/2022.eamt-1.15/).
## Table of Contents

## Installation

For installation instructions, head to `INSTALL.md`

## Downloading or preparing data

The data can be downloaded, cleaned and prepared from scratch by following the instructions below:
1. `bash prepare_pretraining_data.sh` (download and prepare raw data)
2. `bash src/annotation_wrapper.sh` (annotate training data)
4. `python -m src.prep.annotate` (annotate dev/test data)
5. `python postprocess_annotated_data.py`

Alternatively, it can be downloaded [here](todo data link). Once downloaded, data should be inserted at the root of this directory.

## Downloading or training models

Models can be trained by following the following instructions:

1. Pretrain a general model: `python pretrain.py`
2. Either run `python finetune.py --config C`, where C is one of `[emb_enc, emb_sos, emb_enc_sos, tag_enc, tag_dec, tag_enc_dec, emb_pw_sum, emb_add, out_bias]`\
or run `bash finetune_all.sh` which will finetune all architectures in turn.

## Using the Annotation Tool

To use the annotation tool on your custom dataset, run
```python -m src.prep.annotate --prefix P```
where `P` is the path to your dataset, e.g. `data/custom_data`. The tool assumes that a source file `P.en` and a target file `P.pl` exists. Annotation will be stored in `P.cxt`.

## Contributing

If you find a bug or wish to request a feature, please submit
an [Issue](https://github.com/st-vincent1/grammatical_agreement_eamt/issues/new/choose).

## License

The contents of this repository have been released under MIT license.

If you use the resources provided, please include the following citation in your work:
Vincent et al. 2022, [Controlling Extra-Textual Attributes about Dialogue Participants: A Case Study of English-to-Polish Neural Machine Translation](https://aclanthology.org/2022.eamt-1.15) (EAMT 2022)

## References

### Corpora
 - P. Lison and J. Tiedemann, 2016, OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles. In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016)\
     - Data originally sourced from [OpenSubtitles.org](http://www.opensubtitles.org/)
 - Koehn, P. (2005). Europarl: A Parallel Corpus for Statistical Machine Translation. Conference Proceedings: The Tenth Machine Translation Summit, 79–86. http://mt-archive.info/MTS-2005-Koehn.pdf
 
### Annotation tool
 - Ryszard Tuora and Łukasz Kobyliński, "Integrating Polish Language Tools and Resources in spaCy". In: Proceedings of PP-RAI'2019 Conference, 16-18.10.2019, Wrocław, Poland.\
 - Witold Kieraś, Marcin Woliński. Morfeusz 2 – analizator i generator fleksyjny dla języka polskiego. Język Polski, XCVII(1):75–83, 2017.\

### NMT
 - Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32(NeurIPS).

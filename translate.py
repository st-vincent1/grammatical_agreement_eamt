from argparse import ArgumentParser
from os import path

import sentencepiece as spm
import torch
from tqdm import tqdm

from src.models import Transformer
from src.trainer import preprocess
from src.utils import load_params, tensor2text
from src.data import load_pretrain, load_train_dev, load_test

def write_to_file(params, data, filename):
    with open(path.join(params.paths.data, filename), 'w+') as f:
        for line in data:
            f.write(line + '\n')


def translate(params, vocab, data_iter, model, set_):
    data = {'fem': data_iter.fem_iter,
            'masc': data_iter.masc_iter,
            'amb': data_iter.amb_iter}
    genders = {'fem': 1, 'masc': 0, 'amb': 1}

    def translate_iter(set_='train', type_='fem'):
        iter = data[type_]
        gender = genders[type_]
        translated = []
        for batch in tqdm(iter):
            inp_tokens, _, inp_lengths, _ = preprocess(batch, params.eos_idx)
            raw_gender = torch.full_like(inp_tokens[:, 0], gender)
            with torch.no_grad():
                tokens = model(
                    inp_tokens,
                    None,
                    inp_lengths,
                    raw_gender,
                    generate=True,
                    beam_size=5)
            if params.detokenise:
                sp = spm.SentencePieceProcessor(
                    model_file=params.spm)
                detok = tensor2text(vocab, tokens, sp)
            else:
                detok = tensor2text(vocab, tokens)
            translated = translated + detok
        write_to_file(params, translated, f'bert_acc_{set_}.{type_}.Transformer.out')

    for type in ['fem', 'masc', 'amb']:
        translate_iter(set_, type)


def main():
    params = load_params()
    parser = ArgumentParser()
    parser.add_argument('-c', default='', help='Transformer model filename.')
    parser.add_argument('--detok', default='False', action='store_true', help='Detokenise data?')
    args = parser.parse_args()
    params, train_iter, dev_iter, vocab = load_pretrain(params)
    params.detokenise = args.detok
    c_dev = params.c.device
    model_c = Transformer(params, vocab, fn='Transformer')
    model_c.to(c_dev)
    state_dict = path.join(params.paths.save, args.c)
    if 'ckpt' in state_dict:
        checkpoint = torch.load(state_dict, map_location=c_dev)
        model_c.load_state_dict(checkpoint['model_C_state_dict'])
        del checkpoint
    else:
        model_c.load_state_dict(torch.load(path.join(params.paths.save, args.c)))
    model_c.eval()
    #    translate(params, vocab, train_iter, model_c, 'train')
    #    translate(params, vocab, dev_iter, model_c, 'dev')
    translate(params, vocab, test_iter, model_c, 'test')


if __name__ == '__main__':
    main()

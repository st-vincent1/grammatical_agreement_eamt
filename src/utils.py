import json
import os
import random
from types import SimpleNamespace
from tqdm import tqdm

import numpy as np
import torch
import pickle
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_params():
    with open("data/params.json", 'r') as j:
        params = json.loads(j.read(), object_hook=lambda d: SimpleNamespace(**d))
        return params


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def tensor2text(vocab, tensor, sp=None, ignore_tags=True):
    base_vocab_len = 15945
    tensor = tensor.cpu().numpy()
    text, cxt = [], []

    index2word = vocab.itos
    eos_idx = vocab.stoi['<eos>']
    unk_idx = vocab.stoi['<unk>']
    sos_idx = vocab.stoi['<sos>']
    pad_idx = vocab.stoi['<pad>']
    null_idx = vocab.stoi['<null>']
    tagging = False
    if len(vocab) > base_vocab_len:
        # Tags are being used
        tagging = True
    for sample in tensor:
        context_found = []
        sample_filtered = []
        prev_token = None
        # null = False
        for idx in list(sample):
            # if idx == null_idx:
            #     null = True
            if tagging \
                    and ignore_tags \
                    and idx in list(range(base_vocab_len, base_vocab_len + 10)):
                context_found.append(index2word[idx])
                continue
            # if idx == pad_idx and null == False:
            #     context_found.append(index2word[idx])
            #     continue
            if idx == eos_idx:
                break
            if idx in [unk_idx, sos_idx, pad_idx, null_idx] or idx == prev_token:
                continue
            prev_token = idx
            sample_filtered.append(index2word[idx])
        context = ' '.join(context_found)
        sample = ' '.join(sample_filtered)
        cxt.append(context)
        text.append(sample)
    # Decode sentencepiece
    if sp is not None:
        text = [sp.decode(t.split()) for t in text]
    # output = []
    # for (a, b) in zip(cxt, text):
    #     output.append(a + ' || ' + b)
    # return output
    return text


def inference(model, data_iter, vocab, cxt_vocab, sp, evaluate=False):
    sources, references, hypotheses, context, marked = [], [], [], [], []

    for batch in tqdm(data_iter):
        inp_tokens, ref_tokens, inp_lengths, types, marking = preprocess(
            batch, vocab.stoi['<eos>'], model.config, evaluate=evaluate)
        with torch.no_grad():
            raw_tokens = model(
                inp_tokens,
                None,
                inp_lengths,
                types,
                generate=True,
                beam_size=1,
                tag_dec=('dec' in model.config and 'tag' in model.config)
            )
        sources += tensor2text(vocab, inp_tokens.cpu(), sp)
        references += tensor2text(vocab, ref_tokens.cpu(), sp)
        hypotheses += tensor2text(vocab, raw_tokens.cpu(), sp)

        context += tensor2cxt(cxt_vocab, types.cpu()) if model.config != 'pretrain' else None
        marked += tensor2cxt(cxt_vocab, marking.cpu()) if model.config != 'pretrain' else None

    text = [(a, b, c, d, e) for (a, b, c, d, e) in zip(sources, references, hypotheses, context, marked) if len(b) > 0]
    return zip(*text)


def preprocess(batch, eos_idx, config, evaluate=False):
    """
    :param batch: batch to extract sentences from
    :param eos_idx:
    :return: tokens and lengths
    """
    device = torch.device('cuda:0')
    inp_tokens, out_tokens = batch.en.to(device), batch.pl.to(device)
    if config != 'pretrain':
        cxt = batch.cxt.to(device)
        if evaluate:
            return inp_tokens, out_tokens, get_lengths(inp_tokens, eos_idx), cxt, batch.marking.to(device)
        return inp_tokens, out_tokens, get_lengths(inp_tokens, eos_idx), cxt
    return inp_tokens, out_tokens, get_lengths(inp_tokens, eos_idx), None


def tensor2cxt(vocab, tensor):
    if tensor is None:
        return 'Context found in tags'
    tensor = tensor.detach().cpu().numpy()
    cxt = []
    for sample in tensor:
        cxt_ = []
        for idx in list(sample):
            if idx != vocab.stoi['<pad>']:
                cxt_.append(vocab.itos[idx])
        cxt.append(' '.join(cxt_))
    return cxt


def get_lengths(tokens, eos_idx):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths + 1  # +1 for <eos> token
    return lengths

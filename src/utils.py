import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import pickle
import io
import re


class Coherence(object):
    def __init__(self):
        self.types = {
            'spgen': ['<spgen_f>', '<spgen_m>'],
            'ilgen': ['<ilgen_f>', '<ilgen_m>', '<ilgen_x>'],
            'ilnum': ['<ilnum_s>', '<ilnum_p>'],
            'form': ['<form_f>', '<form_i>']
        }
        self.attribs = list(self.types.keys())
        self.reverse_map = {
            '<spgen_f>': '<spgen_m>',
            '<spgen_m>': '<spgen_f>',
            '<ilgen_f>': '<ilgen_m>',
            '<ilgen_m>': '<ilgen_f>',
            '<ilgen_x>': '<ilgen_f>',
            '<ilnum_s>': '<ilnum_p>',
            '<ilnum_p>': '<ilnum_s>',
            '<form_f>': '<form_i>',
            '<form_i>': '<form_f>',
            '': ''
        }
    def get_tag_list(self):
        return [a for b in self.types.values() for a in b]
    def type_to_attr(self, type_):
        assert isinstance(type_, str)
        t1 = re.sub(r'<|>', '', type_)
        return t1[:-2]

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
    tensor = tensor.cpu().numpy()
    text = []
    cxt = []
    index2word = vocab.itos
    eos_idx = vocab.stoi['<eos>']
    unk_idx = vocab.stoi['<unk>']
    sos_idx = vocab.stoi['<sos>']
    pad_idx = vocab.stoi['<pad>']
    null_idx = vocab.stoi['<null>']
    gender_tags = False
    if len(vocab) > 32004:
        # Tags are being used
        gender_tags = True
    for sample in tensor:
        context_found = []
        sample_filtered = []
        prev_token = None
        for idx in list(sample):
            if gender_tags \
                    and ignore_tags \
                    and idx in list(range(32004, 32004 + 10)):
                context_found.append(index2word[idx])
                continue
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

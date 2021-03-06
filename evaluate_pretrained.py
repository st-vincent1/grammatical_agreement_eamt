from argparse import ArgumentParser
from os import path

import torch
import json

from src.evaluator import evaluate
from src.data import load_test, read_vocab, build_context_vocab
from src.models import Transformer, Attributes
from src.utils import load_params, set_seed


def main():
    params = load_params()
    parser = ArgumentParser()

    # Model paths
    parser.add_argument('--model_path', default='pretrained_best.pt')

    args = parser.parse_args()
    set_seed(1)

    params.config = 'pretrain'

    model_path = path.join(params.paths.save, args.model_path)
    device = torch.device(params.device)

    def load_model(params, vocab, state_dict, device):
        model = Transformer(params, vocab)
        model = model.to(device)
        model.config = params.config
        model.load_state_dict(torch.load(state_dict))
        model = model.eval()
        return model

    vocab = read_vocab(params.paths.vocab, device)

    # This is not needed but building it will save many exceptions in load_test
    tag_list = Attributes().type_list
    cxt_vocab = build_context_vocab(tag_list)

    model = load_model(params, vocab, model_path, device)

    params, iters, vocab = load_test(params, vocab, cxt_vocab)

    results = evaluate(params, model, iters, vocab, cxt_vocab, baseline=True)
    with open('out/baseline_results.json', 'w+') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()

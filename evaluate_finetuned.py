import json
import re
from argparse import ArgumentParser
from os import path

import torch

from eval.evaluator import evaluate
from src.data import load_test, read_vocab, build_context_vocab, load_train_dev, add_tags_to_vocab
from src.models import Transformer
from src.trainer import validate_ft
from src.utils import load_params, set_seed, Attributes


def main():
    params = load_params()
    parser = ArgumentParser()

    # Model paths
    parser.add_argument('--model_path', default='', required=True)
    args = parser.parse_args()
    set_seed(1)

    print("Trying to infer model name.")
    model_name = re.findall(r'(\w+_)+\d', args.model_path)[0][:-1]
    print(f"Model name inferred: {model_name}.")
    params.config = model_name

    model_path = path.join(params.paths.save, args.model_path)
    device = torch.device(params.device)

    vocab = read_vocab('upgrade_vocab.pkl', device)
    model = Transformer(params, vocab)
    model.to(device)

    # Adding null because it's necessary for tag_(enc_?)dec
    tag_list = Attributes().get_tag_list() + ['<null>']
    if 'tag' in params.config:
        print("Resizing embedding layer & vocabulary to fit extra gender tokens. . .")
        model.expand_embeddings(len(vocab) + len(tag_list), device)
        if 'dec' in params.config:
            # Need to expand pos_embeddings by 1 as well, to make space for <null>
            model.max_length += 1
            model.expand_pos_embeddings(model.max_length, device)
        vocab = add_tags_to_vocab(vocab, tag_list)
        model.gender_tags = [vocab.stoi[tag] for tag in tag_list]

        # Experimental but should work
        cxt_vocab = vocab
    elif 'emb' in params.config:
        cxt_vocab = build_context_vocab(tag_list)
        model.add_coher_embedding(len(cxt_vocab), device)
    elif params.config == 'out_bias':
        cxt_vocab = build_context_vocab(tag_list)
        model.decoder.generator.config = 'out_bias'
        model.add_bias_embedding(len(cxt_vocab), device)

    model.cxt_vocab = cxt_vocab
    model.load_state_dict(torch.load(model_path))
    model.config = params.config
    model.eval()
    print(f"{model.decoder.generator.config=}")

    params, iters, vocab = load_test(params, vocab, cxt_vocab)
    results = evaluate(params, model, iters, vocab, cxt_vocab, device)

    try:
        with open(f'out/{model.config}_results.json') as json_file:
            data = json.load(json_file)
        for key in data.keys():
            data[key] = data[key] + results[key]
    except FileNotFoundError:
        data = results

    with open(f'out/{model.config}_results.json', 'w+') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    main()

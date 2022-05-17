from argparse import ArgumentParser
from os import path

import torch
from torch import optim
from src.data import load_finetuning_data, add_tags_to_vocab, read_vocab, save_vocab, build_context_vocab
from src.models import Transformer
from src.trainer import train
from src.utils import load_params, set_seed
from src.models import Attributes
import logging


def main():
    params = load_params()
    parser = ArgumentParser()
    parser.add_argument('--model_path', default='pretrained_best.pt')

    parser.add_argument('--config', required=True, choices=('emb_add',
                                                            'emb_enc',
                                                            'emb_dec',
                                                            'emb_enc_dec',
                                                            'emb_pw_sum',
                                                            'tag_enc',
                                                            'tag_dec',
                                                            'tag_enc_dec',
                                                            'out_bias'))
    # Vocab
    parser.add_argument('--save_vocab', default='', help='If expanding embeddings, store vocab under given file path. '
                                                         'Useful if vocab is not built yet '
                                                         'as it must be given from file during inference later.')

    parser.add_argument('--seed', default=1, type=int, help='Random seed.')
    parser.add_argument('--alpha', default=0.50, type=float, help='Regularisation parameter.')
    args = parser.parse_args()
    params.seed = args.seed
    params.alpha = args.alpha
    set_seed(args.seed)

    params.config = args.config

    model_path = path.join(params.paths.save, args.model_path)
    device = torch.device(params.device)

    vocab = read_vocab('vocab.pkl', device)
    model = Transformer(params, vocab)
    model.to(device)

    model.load_state_dict(torch.load(model_path))
    # Adding null because it's necessary for tag_(enc_?)dec
    tag_list = Attributes().type_list + ['<null>']

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {pytorch_total_params} trainable parameters.")
    if 'tag' in params.config:
        logging.info("Resizing embedding layer & vocabulary to fit extra gender tokens. . .")
        model.expand_embeddings(len(vocab) + len(tag_list), device)

        if 'dec' in params.config:
            # Need to expand pos_embeddings by 1 as well, to make space for <null>
            #check if both of these make sense
            model.max_length += 1
            model.expand_pos_embeddings(model.max_length, device)
        vocab = add_tags_to_vocab(vocab, tag_list)

        if args.save_vocab:
            save_vocab(args.save_vocab, vocab)
        model.cxt_vocab = vocab

    elif 'emb' in params.config:
        cxt_vocab = build_context_vocab(tag_list)
        model.cxt_vocab = cxt_vocab
        model.add_coher_embedding(len(cxt_vocab), device)
    elif params.config == 'out_bias':
        cxt_vocab = build_context_vocab(tag_list)
        model.decoder.generator.config = 'out_bias'
        model.cxt_vocab = cxt_vocab
        model.add_bias_embedding(len(cxt_vocab), device)
    optimizer = optim.Adam(model.parameters(), lr=params.ft.lr, weight_decay=params.ft.L2, betas=(0.9, 0.98))
    model.config = params.config

    print(f"{model.decoder.generator.config=}")
    params, train_iter, dev_iter, vocab = load_finetuning_data(params, vocab, model.cxt_vocab)
    train(params, vocab, model, train_iter, dev_iter, optimizer, cxt_vocab=model.cxt_vocab)



if __name__ == '__main__':
    main()

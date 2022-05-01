from src.utils import load_params, set_seed
from sys import exit
from src.data import load_pretraining_data
from os import path
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler
from src.trainer import pretrain
from src.models import Transformer


def start_pretraining(params, train_iter, dev_iter, vocab):
    print(f"Pretraining...")
    scaler = GradScaler()
    model = Transformer(params, vocab)
    model.to(torch.device(params.device))
    model.config = 'pretrain'
    optimizer = optim.Adam(model.parameters(), lr=params.pt.lr, weight_decay=params.pt.L2, betas=(0.9, 0.98))
    save_path = path.join(params.paths.save, f'pretrained_0.pt')
    pretrain(params, vocab, model, optimizer, train_iter, dev_iter, save_path, scaler)


def main():
    set_seed(1)
    params = load_params()
    params.config = 'pretrain'
    loaded_data = load_pretraining_data(params)
    start_pretraining(*loaded_data)


if __name__ == '__main__':
    main()

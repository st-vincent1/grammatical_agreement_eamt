import logging
import os
import re
import time

import numpy as np
import sentencepiece as spm
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_
from sacrebleu.metrics import BLEU, CHRF

from src.data import load_finetuning_data
from src.utils import inference, preprocess

logging.basicConfig(level=logging.INFO)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def pretrain(params, vocab, model, optimizer, train_iters, dev_iters, save_path, scaler) -> None:
    iter_step = 0
    assert params.config == 'pretrain'

    def pretrain_epoch(iter_step):
        iter_n = params.pt.total_batch // params.pt.batch_size
        optimizer.zero_grad()
        model.train()
        accum_loss = 0
        his_loss = []
        for i, batch in enumerate(train_iters):
            if (i + 1) % params.pt.eval_every == 0:
                break
            iter_step += 1
            loss = step(params, model, scaler, batch)
            accum_loss += loss
            if (i + 1) % iter_n == 0:
                # Step the optimizer after processing n batches
                opt_step(model, optimizer, scaler)
                accum_loss /= iter_n
                his_loss.append(accum_loss)
                if (i + 1) % (iter_n * 100) == 0:
                    logging.info(f"Step [{iter_step}]: loss {accum_loss:.2f}.")
                accum_loss = 0
        avrg_loss = np.mean(his_loss)
        return avrg_loss, iter_step

    # Assume that save path for the model is load path with different epoch number
    patience = params.pt.patience
    best_path = re.sub(r'(\d+)\.pt', rf'best.pt', save_path)
    best_chrf = -1
    for ep in range(1, params.pt.max_epoch + 1):
        start_time = time.time()
        train_loss, iter_step = pretrain_epoch(iter_step)
        bleu, chrf = validate(params, vocab, _, model, dev_iters)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if chrf >= best_chrf:
            best_chrf = chrf
            logging.info(f"New record for chrF: {best_chrf:.2f}. Saving model...")
            torch.save(model.state_dict(), best_path)
            patience = params.pt.patience
        else:
            patience -= 1
            logging.info(f'Model not saved. Remaining patience: {patience}')
            if patience == 0:
                break
        logging.info(
            f'[Epoch: {ep}/{params.pt.max_epoch}] Time: {epoch_mins}m {epoch_secs}s | '
            f'Train loss: {train_loss:.3f} | BLEU: {bleu:.3f} | chrF: {chrf:.3f}')


def train(params, vocab, model, train_iters, dev_iters, optimizer, cxt_vocab=None):
    scaler = GradScaler()
    model.train()
    save_path = os.path.join(params.paths.save, f'{params.config}.pt')
    his_loss = []

    iter_n = params.ft.total_batch // params.ft.batch_size
    accum_loss = 0
    best_chrf = -1
    batch_iters = iter(train_iters)
    start_time = time.time()

    for k in range(1, params.ft.max_steps + 1):
        for _ in range(iter_n):
            try:
                batch = next(batch_iters)
            except StopIteration:
                # Reload data, labelling will be re-done so change seed
                params.seed = params.seed + 10
                _, train_iters, _, _ = load_finetuning_data(params, vocab, model.cxt_vocab)
                batch_iters = iter(train_iters)
                batch = next(batch_iters)
            loss = step(params, model, scaler, batch)
            accum_loss += loss
        his_loss.append(accum_loss / iter_n)
        accum_loss = 0
        opt_step(model, optimizer, scaler=scaler)
        if k % params.ft.log_steps == 0:
            avrg_loss = np.mean(his_loss)
            # Time the epoch
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            start_time = time.time()
            logging.info(f'    [Iteration {k}] ({epoch_mins}m {epoch_secs}s): {avrg_loss:.2f}')
            his_loss = []
        if k % params.ft.eval_steps == 0:
            model.eval()
            torch.save(model.state_dict(), re.sub('.pt', f'_{k}.pt', save_path))
            _, chrf = validate(params, vocab, cxt_vocab, model, dev_iters)
            if chrf > best_chrf:
                logging.info(f"Saving new model.")
                torch.save(model.state_dict(), save_path)
                best_chrf = chrf
            model.train()
            end_time = time.time()
            eval_mins, eval_secs = epoch_time(start_time, end_time)
            logging.info(f'   [Evaluation]: ({eval_mins}m {eval_secs}s)')
            start_time = time.time() 
    # Saving last model for inspection
    torch.save(model.state_dict(), re.sub('.pt', '_last.pt', save_path))

def compute_loss(loss_fn, log_probs, out_tokens, token_mask, batch_size):
    loss = loss_fn(log_probs.transpose(1, 2), out_tokens) * token_mask
    loss = loss.sum() / batch_size
    return loss


def step(params, model, scaler, batch):
    pad_idx = params.pad_idx
    eos_idx = params.eos_idx

    loss_fn = nn.NLLLoss(reduction='none')

    inp_tokens, out_tokens, inp_lengths, types, _ = preprocess(batch, eos_idx, params.config)
    batch_size = inp_tokens.size(0)
    token_mask = (out_tokens != pad_idx).float()
    with autocast():
        log_probs = model(
            inp_tokens,
            out_tokens,
            inp_lengths,
            types,
            generate=False
        )
        loss = compute_loss(loss_fn, log_probs, out_tokens, token_mask, batch_size)
    scaler.scale(loss).backward()
    return loss.item()


def opt_step(model, optimizer, scaler):
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), 5)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    return

def validate(params, vocab, cxt_vocab, model, dev_iters):
    bleu = BLEU()
    chrf = CHRF(word_order=2)
    sp = spm.SentencePieceProcessor(model_file=params.spm)

    src, ref, hyp, types_, _ = inference(model, dev_iters, vocab, cxt_vocab, sp)
    bleu_score = bleu.corpus_score(hyp, [ref]).score
    chrf_score = chrf.corpus_score(hyp, [ref]).score
    for _ in range(8):
        idx = np.random.randint(len(hyp))
        logging.info(f"\n{'*' * 40}\n[source    ] {src[idx]}\n[hypothesis] {hyp[idx]}\n[reference ] {ref[idx]}\n")
        if model.config != 'pretrain':
            logging.info(f"[context   ] {types_[idx]}\n{'*' * 40}")
    logging.info(f"Scores:\n   BLEU: {bleu_score:.2f}\n   chrF++: {chrf_score:.2f}\n")
    return bleu_score, chrf_score

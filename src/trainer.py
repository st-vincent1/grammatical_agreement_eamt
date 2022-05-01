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
from tqdm import tqdm

from src.data import load_finetuning_data
from src.utils import tensor2text, tensor2cxt, get_lengths

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
        bleu, chrf = calculate_metrics(params, dev_iters, model, vocab)
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
        #    f'[Epoch: {ep}/{params.pt.max_epoch}] Time: {epoch_mins}m {epoch_secs}s | Val. loss: {dev_loss:.3f} | '
        logging.info(
            f'[Epoch: {ep}/{params.pt.max_epoch}] Time: {epoch_mins}m {epoch_secs}s | '
            f'Train loss: {train_loss:.3f} | BLEU: {bleu:.3f} | chrF: {chrf:.3f}')


def train(params, vocab, model, train_iters, dev_iters, optimizer, global_step=0, cxt_vocab=None):
    scaler = GradScaler()
    model.train()
    save_path = os.path.join(params.paths.save, f'{params.config}_{params.seed}.pt')
    his_loss = []

    iter_n = params.ft.total_batch // params.ft.batch_size
    accum_loss = 0
    best_chrf = -1
    global_step = max(0, global_step - 1)
    start_time = time.time()
    batch_iters = iter(train_iters)
    for k in range(global_step + 1, params.ft.max_steps + 1):
        for _ in range(iter_n):
            try:
                batch = next(batch_iters)
            except StopIteration:
                # Reload data, labelling will be re-done so change seed
                params.seed = params.seed + 10
                _, train_iters, _, _ = load_train_dev(params, vocab, model.cxt_vocab)
                batch_iters = iter(train_iters)
                batch = next(batch_iters)
            loss = step(params, model, scaler, batch)
            # Inspecting weights
            # torch.set_printoptions(threshold=10_000)
            # print(f"{model.decoder.generator.coher_emb.weight[-8:] = }")
            accum_loss += loss
        his_loss.append(accum_loss / iter_n)
        accum_loss = 0
        opt_step(model, optimizer, scaler=scaler)
        if k % params.ft.log_steps == 0:
            avrg_loss = np.mean(his_loss)
            # Time the epoch
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'[iter {k}] ({epoch_mins}m {epoch_secs}s): {avrg_loss:.2f}')
            # print(f'{model.decoder.generator.gender_embed.weight=}')
            start_time = time.time()
            his_loss = []
        if k % params.ft.eval_steps == 0:
            model.eval()
            _, chrf = validate_ft(params, vocab, cxt_vocab, model, dev_iters)
            if chrf > best_chrf:
                print(f"Saving new model.")
                torch.save(model.state_dict(), f'{save_path}_best.pt')
                best_chrf = chrf
            model.train()
            # save model
        # if k % params.ft.ckpt_steps == 0:
        #     torch.save({
        #         'epoch': k + 1,
        #         'model_C_state_dict': model.state_dict(),
        #         'optimizer_C_state_dict': optimizer.state_dict(),
        #     }, f'{save_path}_{k}_ckpt.pt')


def preprocess(batch, eos_idx, config, evaluate=False):
    """
    :param batch: batch to extract sentences from
    :param eos_idx:
    :return: tokens and lengths
    """
    # todo make this independent of cuda...
    inp_tokens, out_tokens = batch.en.to(torch.device('cuda:0')), batch.pl.to(torch.device('cuda:0'))
    if config != 'pretrain':
        cxt = batch.cxt.to(torch.device('cuda:0'))
        if evaluate:
            return inp_tokens, out_tokens, get_lengths(inp_tokens, eos_idx), cxt, batch.marking.to(
                torch.device('cuda:0'))
        return inp_tokens, out_tokens, get_lengths(inp_tokens, eos_idx), cxt
    return inp_tokens, out_tokens, get_lengths(inp_tokens, eos_idx), None


def compute_loss(loss_fn, log_probs, out_tokens, token_mask, batch_size):
    loss = loss_fn(log_probs.transpose(1, 2), out_tokens) * token_mask
    loss = loss.sum() / batch_size
    return loss


def step(params, model, scaler, batch):
    pad_idx = params.pad_idx
    eos_idx = params.eos_idx

    loss_fn = nn.NLLLoss(reduction='none')

    # Collect a minibatch
    inp_tokens, out_tokens, inp_lengths, types = preprocess(batch, eos_idx, params.config)
    batch_size = inp_tokens.size(0)
    token_mask = (out_tokens != pad_idx).float()
    with autocast():
        log_probs, _ = model(
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


def translate_batch(params, batch, model, vocab, sp, device=torch.device('cuda')):
    eos_idx = params.eos_idx
    inp_tokens, out_tokens, inp_lengths, _ = preprocess(batch, eos_idx, params.config)
    inp_tokens = inp_tokens.to(device)
    out_tokens = out_tokens.to(device)
    inp_lengths = inp_lengths.to(device)
    with torch.no_grad():
        with autocast():
            hyp_tokens, _ = model(
                inp_tokens,
                out_tokens,
                inp_lengths,
                None,
                generate=True,
                beam_size=5)
    src_text = tensor2text(vocab, inp_tokens.cpu(), sp)
    trg_text = tensor2text(vocab, out_tokens.cpu(), sp)
    hyp_text = tensor2text(vocab, hyp_tokens.cpu(), sp)
    print(trg_text[3], hyp_text[3])
    return src_text, trg_text, hyp_text


def calculate_metrics(params, data_iter, model, vocab, device=torch.device('cuda')):
    chrf = CHRF(word_order=2)
    bleu = BLEU()
    sp = spm.SentencePieceProcessor(model_file=params.spm)

    ref_text_acc, hyp_text_acc = [], []

    for batch in data_iter:
        _, ref_text, hyp_text = translate_batch(
            params, batch, model, vocab, sp, device)
        ref_text_acc.append(ref_text)
        hyp_text_acc.append(hyp_text)

    # flatten lists to remove batch division
    def flatten_list(lst):
        return [item for sublist in lst for item in sublist]

    ref, hyp = flatten_list(ref_text_acc), flatten_list(hyp_text_acc)

    # Removing instances where ref is empty or sacrebleu will be sad
    text = [(a, b) for (a, b) in zip(ref, hyp) if len(a) > 0]
    ref, hyp = zip(*text)
    bleu_score = bleu.corpus_score(hyp, [ref]).score
    chrf_score = chrf.corpus_score(hyp, [ref]).score

    return bleu_score, chrf_score


def inference(model, data_iter, vocab, cxt_vocab, sp, device):
    """
    Main inference function.
    :param model: Model used to perform inference.
    :param data_iter: Test data.
    :param raw_style:
    :param vocab:
    :param sp:
    :return:
    """
    src, ref, raw, types_ = [], [], [], []

    for batch in tqdm(data_iter):
        inp_tokens, ref_tokens, inp_lengths, types = preprocess(batch, vocab.stoi['<eos>'], model.config)
        with torch.no_grad():
            raw_tokens, _ = model(
                inp_tokens,
                None,
                inp_lengths,
                types,
                generate=True,
                beam_size=5,
                tag_dec=('dec' in model.config and 'tag' in model.config)
            )
        src += tensor2text(vocab, inp_tokens.cpu(), sp)
        ref += tensor2text(vocab, ref_tokens.cpu(), sp)
        raw += tensor2text(vocab, raw_tokens.cpu(), sp)
        types_ += tensor2cxt(cxt_vocab, types.cpu())
    # Removing instances where ref is empty or sacrebleu will be sad
    text = [(a, b, c, d)
            for (a, b, c, d) in zip(src, ref, raw, types_) if len(b) > 0]
    src, ref, raw, types_ = zip(*text)

    return src, ref, raw, types_


def validate_ft(params, vocab, cxt_vocab, model, dev_iters):
    bleu = BLEU()
    chrf = CHRF(word_order=2)
    sp = spm.SentencePieceProcessor(model_file=params.spm)

    src, ref, raw, types_ = inference(model, dev_iters, vocab, cxt_vocab, sp, device=params.ft.device)
    bleu_score = bleu.corpus_score(raw, [ref]).score
    chrf_score = chrf.corpus_score(raw, [ref]).score
    for _ in range(5):
        idx = np.random.randint(len(raw))
        print('*' * 40)
        print('[src  ]', src[idx])
        print('[raw  ]', raw[idx])
        print('[ref  ]', ref[idx])
        print('[types]', types_[idx])
        print('*' * 40)
    print(f"Scores:\n   BLEU: {bleu_score:.3f}\n   chrF++: {chrf_score:.3f}\n")
    return bleu_score, chrf_score

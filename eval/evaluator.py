from sacrebleu.metrics import BLEU, CHRF
import sentencepiece as spm
# import spacy
import torch

from tqdm import tqdm
import numpy as np

from src.trainer import preprocess
from src.utils import get_lengths, tensor2text, tensor2cxt, Attributes


def infer(model, data_iter, vocab, cxt_vocab, sp):
    """
    Main inference function.
    Optional functionality: alongside with the hypothesis list, return a list of types that were used (main).
    So just gather marking.

    :param model: Model used to perform inference.
    :param data_iter: Test data.
    :param vocab:
    :param sp:
    :return:
    """
    en = []
    ref = []
    hyp = []
    mark = []
    cxt = []
    for batch in tqdm(data_iter):
        inp_tokens, ref_tokens, inp_lengths, types, marking = preprocess(batch, vocab.stoi['<eos>'], model.config, evaluate=True)
        with torch.no_grad():
            tokens, _ = model(
                inp_tokens,
                None,
                inp_lengths,
                types,
                generate=True,
                beam_size=5,
                tag_dec=('dec' in model.config and 'tag' in model.config)
            )
        en += tensor2text(vocab, inp_tokens.cpu(), sp)
        ref += tensor2text(vocab, ref_tokens.cpu(), sp)
        hyp += tensor2text(vocab, tokens.cpu(), sp)
        if cxt_vocab is not None:
            mark += tensor2cxt(cxt_vocab, marking.cpu())
            cxt += tensor2cxt(cxt_vocab, types.cpu())

    # Removing instances where ref or hyp is empty or sacrebleu will be sad
    if cxt_vocab is not None:
        text = [(a, b, c, d, e)
                for (a, b, c, d, e) in zip(en, ref, hyp, mark, cxt) if len(b) > 0]

    else:
        text = [(a, b, c)
                for (a, b, c) in zip(en, ref, hyp) if len(b) > 0]

    return zip(*text)

def evaluate(params, model, iters, vocab, cxt_vocab, device, baseline=False):
    """
    Produce scores for evaluation:
    - chrF++ for isolated and full context scenario
    - BLEU for isolated and full context scenario
    - Agreement to specific attributes
    - AmbID
    :param params:
    :param model:
    :param iters:
    :param vocab:
    :param cxt_vocab:
    :param device:
    :param baseline:
    :return:
    """
    chrf = CHRF(word_order=2)
    bleu = BLEU()
    def compute_metrics(refs, raw, rev=None):
        bleu_raw = bleu.corpus_score(raw, [refs]).score
        chrf_raw = chrf.corpus_score(raw, [refs]).score
        if rev:
            bleu_rev = bleu.corpus_score(rev, [refs]).score
            chrf_rev = chrf.corpus_score(rev, [refs]).score
            return bleu_raw, chrf_raw, bleu_rev, chrf_rev
        return bleu_raw, chrf_raw

    sp = spm.SentencePieceProcessor(model_file=params.spm)
    hyps, refs, marks, cxts = {}, {}, {}, {}
    results = {}
    
    attrs = Attributes()
    detector = Detector()

    attrib_counts = {x: [[], []] for x in attrs.attribute_list}
    if baseline:
        print("Getting scores for the pretrained model.")
        src, ref, hyp, marks, _ = infer(model, iters['raw'], vocab, cxt_vocab, sp, device)

        results['bleu'], results['chrf'] = compute_metrics(ref, hyp)
        agr_corr_raw, agr_incorr_raw = detector.calculate_type_agreement(hyp, src, marks)
        for att in attrs.attribute_list:
           results[f'agreement_{att}'] = agr_corr_raw[att] / (agr_corr_raw[att] + agr_incorr_raw[att]) * 100
        print(results)
        return results

    # print("Generating hypotheses for full and isolated contexts.")
    src, ref, hyps['raw'], marks['raw'], cxts['raw'] = infer(model, iters['raw'], vocab, cxt_vocab, sp, device)

    results['bleu_raw'], results['chrf_raw'] = compute_metrics(ref, hyps['raw'])
    src, ref, hyps['iso_raw'], marks['iso_raw'], cxts['iso_raw'] = infer(model, iters['iso_raw'], vocab, cxt_vocab, sp, device)
    for reff, hyp, mark in zip(ref, hyps['raw'], marks['raw']):
        attrib = attrs.type_to_attr(mark)
        attrib_counts[attrib][0] += [reff]
        attrib_counts[attrib][1] += [hyp]
    for attrib in attrs.attribute_list:
        results[f'bleu_{attrib}'], results[f'chrf_{attrib}'] = compute_metrics(attrib_counts[attrib][0], attrib_counts[attrib][1])
        
    results['bleu_iso'], results['chrf_iso'] = compute_metrics(ref, hyps['iso_raw'])
    print("Generating ambivalent hyps for AmbID")
    _, amb_ref, hyps['amb_raw'], _, _ = infer(model, iters['amb'], vocab, cxt_vocab, sp, device)
    _, _, hyps['amb_rev'], _, _ = infer(model, iters['amb_rev'], vocab, cxt_vocab, sp, device)
    # Removing the one empty instance - but doing it right
    hyps['amb_raw'], hyps['amb_rev'] = zip(*[(a, b) for (a, b) in zip(hyps['amb_raw'], hyps['amb_rev']) if a and b])
    results['ambchrf'] = compute_metrics(hyps['amb_raw'], hyps['amb_rev'])[1]

    detector = Detector()
    # Calculate agreement for full context & isolated hypotheses.
    for pref in ['', 'iso_']:
       agr_corr_raw, agr_incorr_raw = detector.calculate_type_agreement(hyps[f'{pref}raw'], src, marks[f'{pref}raw'])
       for att in attrs.attribute_list:
           results[f'{pref}agreement_{att}'] = agr_corr_raw[att] / (agr_corr_raw[att] + agr_incorr_raw[att]) * 100

    for key in results.keys():
        results[key] = [results[key]]
    print(results)
    return results


def evaluate_baseline(params, base_model, test_iters, vocab, device):
    """Code to get scores for pretrained model."""
    chrf = CHRF(word_order=2)
    bleu = BLEU()
    sp = spm.SentencePieceProcessor(model_file=params.spm)
    print(f"Params in baseline: {sum(p.numel() for p in base_model.parameters())}")

    src, ref, hyp = infer(base_model, test_iters, vocab, None, sp)
    bleu = bleu.corpus_score(hyp, [ref]).score
    chrf = chrf.corpus_score(hyp, [ref]).score
    print(f"Baseline scores:\n   BLEU: {bleu:.2f}\nchrF++: {chrf:.2f}")
    return

from sacrebleu.metrics import BLEU, CHRF
import sentencepiece as spm

from src.utils import inference
from src.models import Detector, Attributes
import logging

import pandas as pd
import re

logging.basicConfig(level=logging.INFO)


def evaluate(params, model, iters, vocab, cxt_vocab, baseline=False):
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
    srcs, hyps, refs, marks, cxts = {}, {}, {}, {}, {}
    results = {}
    def generate_chrf_for_groups(df):
        chrf = CHRF(word_order=2)
        if 'dec' in params.config: 
            df.cxts = df.cxts.apply(attribs.sort_group)
        for group in attribs.groups.keys():
            group = re.sub(r',|\.|\*|\[|\]|\^', ' ', group).strip()
            pattern = fr"^{group}$"
            if 'sp' not in group:
                pattern = rf'^(<sp:feminine> |<sp:masculine> |){pattern[1:]}'
            else:
                pattern = rf'{pattern[:-1]}.*$'
            group_hypotheses = df[df.cxts.str.contains(pattern)]
            results[f'chrF++:{group}'] = chrf.corpus_score(group_hypotheses['hyps'].tolist(),
                                                           [group_hypotheses['refs'].tolist()]).score
    attribs = Attributes()
    detector = Detector()

    attrib_counts = {x: [[], []] for x in attribs.attribute_list}

    for setting in ['isolated', 'full']:
        logging.info(f"Generating scores for {setting}.")
        srcs[setting], refs[setting], hyps[setting], cxts[setting], marks[setting] = inference(model, iters[setting],
                                                                                               vocab, cxt_vocab, sp,
                                                                                               evaluate=True)
        results[f'BLEU:{setting}'], results[f'chrF++:{setting}'] = compute_metrics(refs[setting], hyps[setting])

        for reference, hypothesis, marking, context in zip(refs[setting], hyps[setting], marks[setting], cxts[setting]):
            attrib = attribs.identify_from_type(marking)
            attrib_counts[attrib][0] += [reference]
            attrib_counts[attrib][1] += [hypothesis]
        for attrib in attribs.attribute_list:
            results[f'{attrib}:BLEU:{setting}'], results[f'{attrib}:chrf++:{setting}'] = compute_metrics(
                attrib_counts[attrib][0], attrib_counts[attrib][1])
        logging.info(f"Inspecting {setting}")
        agr_corr_hyp, agr_incorr_hyp = detector.calculate_type_agreement(hyps[setting], srcs[setting], marks[setting])
        for att in attribs.attribute_list:
            results[setting] = agr_corr_hyp[att] / (agr_corr_hyp[att] + agr_incorr_hyp[att]) * 100
    if not baseline:
        logging.info("Generating ambivalent hyps for AmbID.")
        _, refs['amb'], hyps['amb'], _, _ = inference(model, iters['amb'], vocab, cxt_vocab, sp, evaluate=True)
        _, _, hyps['amb_rev'], _, _ = inference(model, iters['amb_rev'], vocab, cxt_vocab, sp, evaluate=True)
        # Removing the one empty instance - but doing it right
        hyps['amb'], hyps['amb_rev'] = zip(*[(a, b) for (a, b) in zip(hyps['amb'], hyps['amb_rev']) if a and b])
        results['AmbID'] = compute_metrics(hyps['amb'], hyps['amb_rev'])[1]

    # For plotting purposes, calculate chrF++ scores for each group# for key in results.keys():
    df = pd.DataFrame({'hyps': hyps['full'], 'refs': refs['full'], 'cxts': cxts['full']})
    
    generate_chrf_for_groups(df)

    logging.info(results)
    return results, refs, hyps


def evaluate_baseline(params, base_model, test_iters, vocab):
    """Code to get scores for pretrained model."""
    chrf = CHRF(word_order=2)
    bleu = BLEU()
    sp = spm.SentencePieceProcessor(model_file=params.spm)
    print(f"Params in baseline: {sum(p.numel() for p in base_model.parameters())}")

    src, ref, hyp, _, _ = inference(base_model, test_iters, vocab, None, sp, evaluate=True)
    bleu = bleu.corpus_score(hyp, [ref]).score
    chrf = chrf.corpus_score(hyp, [ref]).score
    print(f"Baseline scores:\n   BLEU: {bleu:.2f}\nchrF++: {chrf:.2f}")
    return

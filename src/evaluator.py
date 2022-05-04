from sacrebleu.metrics import BLEU, CHRF
import sentencepiece as spm

from src.utils import inference
from src.models import Detector, Attributes
import logging

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
    hyps, refs, marks, cxts = {}, {}, {}, {}
    results = {}

    attrs = Attributes()
    detector = Detector()

    attrib_counts = {x: [[], []] for x in attrs.attribute_list}
    if baseline:
        logging.info("Getting scores for the pretrained model.")
        src, ref, hyp, _, marks = inference(model, iters['full'], vocab, cxt_vocab, sp, evaluate=True)

        results['BLEU'], results['chrF'] = compute_metrics(ref, hyp)
        agr_corr_hyp, agr_incorr_hyp = detector.calculate_type_agreement(hyp, src, marks)
        for att in attrs.attribute_list:
            results[f'Agreement to {att}'] = agr_corr_hyp[att] / (agr_corr_hyp[att] + agr_incorr_hyp[att]) * 100
        logging.info(results)
        return results

    # print("Generating hypotheses for full and isolated contexts.")
    src, ref, hyps['full'], cxts['full'], marks['full'] = inference(model, iters['full'], vocab, cxt_vocab, sp, evaluate=True)

    results['BLEU:full'], results['chrF:full'] = compute_metrics(ref, hyps['full'])
    src, ref, hyps['isolated'], cxts['isolated'], marks['isolated'] = inference(model, iters['isolated'], vocab, cxt_vocab, sp, evaluate=True)

    for reff, hyp, mark, cxt in zip(ref, hyps['full'], marks['full'], cxts['full']):
        # if '<sp' in mark:
        #     logging.info(f"Reference: {reff}\nHypothesis: {hyp}\nMarking:{mark}\nContext: {cxt}")
        attrib = attrs.identify_from_type(mark)
        attrib_counts[attrib][0] += [reff]
        attrib_counts[attrib][1] += [hyp]
    for attrib in attrs.attribute_list:
        results[f'bleu_{attrib}'], results[f'chrf_{attrib}'] = compute_metrics(attrib_counts[attrib][0],
                                                                               attrib_counts[attrib][1])

    results['BLEU:isolated'], results['chrF:isolated'] = compute_metrics(ref, hyps['isolated'])
    logging.info("Generating ambivalent hyps for AmbID.")
    _, amb_ref, hyps['amb_hyp'], _, _ = inference(model, iters['amb'], vocab, cxt_vocab, sp, evaluate=True)
    _, _, hyps['amb_rev'], _, _ = inference(model, iters['amb_rev'], vocab, cxt_vocab, sp, evaluate=True)
    # Removing the one empty instance - but doing it right
    hyps['amb_hyp'], hyps['amb_rev'] = zip(*[(a, b) for (a, b) in zip(hyps['amb_hyp'], hyps['amb_rev']) if a and b])
    results['AmbID'] = compute_metrics(hyps['amb_hyp'], hyps['amb_rev'])[1]

    # Calculate agreement for full context & isolated hypotheses.
    for setting in ['full', 'isolated']:
        logging.info(f"Inspecting {setting}")
        agr_corr_hyp, agr_incorr_hyp = detector.calculate_type_agreement(hyps[setting], src, marks[setting])
        for att in attrs.attribute_list:
            results[setting] = agr_corr_hyp[att] / (agr_corr_hyp[att] + agr_incorr_hyp[att]) * 100

    for key in results.keys():
        results[key] = [results[key]]
    logging.info(results)
    return results


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

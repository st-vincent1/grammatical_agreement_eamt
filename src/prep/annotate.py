from src.models import Detector
from src.prep.utils import read_from_file
import logging
from tqdm import tqdm
from argparse import ArgumentParser


def compute_pr(answers, golden_answers):
    stats = {
        'false_positive': 0,
        'false_negative': 0,
        'true_positive': 0,
        'true_negative': 0
    }
    for predicted, actual in zip(answers, golden_answers):
        predicted = predicted.split(',')
        actual = actual.split(',')
        stats['true_positive'] += sum([actual[i] == predicted[i] == '' for i in range(len(predicted))])
        stats['true_positive'] += sum([actual[i] == predicted[i] and actual[i] != '' for i in range(len(predicted))])
        stats['false_negative'] += sum([actual[i] != '' and predicted[i] == '' for i in range(len(predicted))])
        stats['false_positive'] += sum([actual[i] == '' and predicted[i] != '' for i in range(len(predicted))])

    precision = stats['true_positive'] / (stats['true_positive'] + stats['false_positive']) * 100
    recall = stats['true_positive'] / (stats['true_positive'] + stats['false_negative']) * 100
    logging.info(f"{precision = }; {recall = }")
    return (2 * precision * recall) / (precision + recall)


def evaluate_on_sample(prefix: str, detector: Detector) -> float:
    annotations = annotate_corpus(prefix, detector)
    with open(f"{prefix}.gold") as f:
        gold_annotations = f.read().splitlines()

    return compute_pr(annotations, gold_annotations)


def annotate_corpus(prefix: str, detector: Detector) -> None:
    # Typical annotation: '<sp:feminine>,<il:masculine>,<plural>,<formal>'
    data = read_from_file(prefix, config='pretrain')
    annotations = detector.annotate(data)
    with open(f'{prefix}.cxt', 'w+') as f:
        f.write("\n".join(annotations) + "\n")
    return annotations


if __name__ == '__main__':
    detector = Detector()
    parser = ArgumentParser()
    parser.add_argument('--prefix', default=None)
    args = parser.parse_args()
    # Evaluate on a sample
    f1 = evaluate_on_sample('data/detector_dev/detector_sample', detector)

    assert f1 > 99, "Threshold not reached; quitting"
    logging.info('--- Threshold reached. Annotating corpus...')
    if args.prefix is None:
        for split_set in ['dev', 'test']:
            annotate_corpus(f'data/finetune/en-pl.{split_set}', detector)
    else:
        annotate_corpus(args.prefix, detector)

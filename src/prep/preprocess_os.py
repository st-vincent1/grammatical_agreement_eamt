import argparse

from utils import clean_corpus, make_working_splits, write_to_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default='os', required=True)
    parser.add_argument("-en", required=True)
    parser.add_argument("-pl", required=True)
    args = parser.parse_args()
    parsed_en, parsed_pl = clean_corpus(args.en, args.pl, args.corpus)
    train, test = make_working_splits(parsed_en, parsed_pl)
    write_to_file(train, f'data/working/upgrade_train_{args.corpus}')
    write_to_file(test, f'data/working/upgrade_test_{args.corpus}')

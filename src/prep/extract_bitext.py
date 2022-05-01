import xml.etree.ElementTree as ET
from argparse import ArgumentParser
import os
import sys
from tqdm import tqdm
from utils import *
import random


def parse_documents():
    pairname = "en-pl"

    align_filename = f"data/raw/OpenSubtitles/en-pl/en-pl.xml"
    """
    Part 1: Parse alignments
    """
    align_tree = ET.parse(align_filename)
    collection = align_tree.getroot()
    # Identify aligned files
    path_to_xml = 'data/raw/OpenSubtitles/xml'
    path_to_output = f"data"

    test_ids = []
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)
    for document in tqdm(collection):
        # Should not parse if on the
        en_file = os.path.join(path_to_xml, document.attrib['fromDoc'][:-3])
        pl_file = os.path.join(path_to_xml, document.attrib['toDoc'][:-3])
        try:
            en_tree = ET.parse(en_file)
            en_root = en_tree.getroot()
            en_subtitles = parse_subtitles(en_root)
            pl_tree = ET.parse(pl_file)
            pl_root = pl_tree.getroot()
            pl_subtitles = parse_subtitles(pl_root)
        except FileNotFoundError:
            print("Error when parsing source file")
            continue

        pairs_to_parse = []
        for alignment in document:
            # if it is a pair and it has the overlap of at least 0.9
            if 'overlap' in alignment.attrib.keys() and float(alignment.attrib['overlap']) > 0.9:
                en, pl = alignment.attrib['xtargets'].split(';')
                en, pl = en.split(), pl.split()
                pairs_to_parse.append((en, pl))

        check = random.random()
        if check <= 0.96:
            split_set = 'train'
        elif check <= 0.98:
            split_set = 'dev'
        else:
            split_set = 'test'
        if split_set != 'train':
            test_ids += [en_file, pl_file]
        """
        # Print context and main sentences to files
        # """
        for en, pl in pairs_to_parse:
            write_to_file_extractor(os.path.join(path_to_output, f"en-pl.{split_set}.en"), en_subtitles, en)
            write_to_file_extractor(os.path.join(path_to_output, f"en-pl.{split_set}.pl"), pl_subtitles, pl)
    with open('data/opensubtitles_test.ids', 'w+') as f:
        for line in test_ids:
            f.write(line + "\n")
if __name__ == '__main__':
    parse_documents()

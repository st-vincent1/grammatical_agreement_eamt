from typing import List, Tuple, Union, Dict
import re
from argparse import ArgumentParser
from mosestokenizer import MosesDetokenizer, MosesPunctuationNormalizer
from tqdm import tqdm
from pprint import pprint
from dataclasses import dataclass
from utils import bert_nsp_formatter
import cld3

REMOVE_TOKEN = "@Remove@"

x: int = 5

@dataclass
class SententialPreprocessor:
    languages: List[str]
    detok: Dict[str, MosesDetokenizer]
    norm: Dict[str, MosesPunctuationNormalizer]

    def preprocess(self, sent: str, lang: str) -> str:
        return self.lang_check(
            self.manual_clean(
                self.remove_trailing_dashes(
                    self.moses_clean(
                        sent, self.detok[lang], self.norm[lang])), lang), lang)

    @classmethod
    def init_from_langs(cls, languages):
        detok = {}
        norm = {}
        for lang in languages:
            detok[lang] = MosesDetokenizer(lang)
            norm[lang] = MosesPunctuationNormalizer(lang)
        return cls(languages, detok, norm)

    @staticmethod
    def remove_trailing_dashes(sentence: str) -> str:
        if sentence == REMOVE_TOKEN: return sentence
        return re.sub(r'^ *-* *', '', sentence)

    @staticmethod
    def moses_clean(sentence: str, detokenize: MosesDetokenizer, normalize: MosesPunctuationNormalizer) -> str:
        if sentence == REMOVE_TOKEN or sentence == "": return sentence
        return normalize(detokenize(sentence.split()))

    @staticmethod
    def manual_clean(sentence: str, lang: str) -> str:
        def common_errors(sentence: str, lang: str) -> str:
            """ https://github.com/rbawden/PrepCorpus-OpenSubs/blob/master/scripts-opensubs/clean-up-subs.py """
            if lang == "en":
                sentence = sentence.replace('l"m', "I'm")
                sentence = re.sub('([^ ])l" ?II', r"\1I'll", sentence)
                sentence = re.sub("(^| )l([',.-])", r"\1I\2", sentence)
                sentence = sentence.replace(" l...l", " I...I")
                sentence = re.sub("l\-[lI](['\- ])", r"I-I\1", sentence)
                sentence = re.sub("'lI[\- ]", "'ll ", sentence)
                sentence = re.sub("^lsn't", "Isn't", sentence)
                sentence = re.sub(r"^l ", "I ", sentence)
                sentence = re.sub(r"^l[tsnf] ", "Is ", sentence)
                sentence = re.sub(r"^lt'", "It'", sentence)
                sentence = sentence.replace(" l ", " I ")
                sentence = re.sub("[\- ]I'lI[\- ]", " I'll ", sentence)

                for word, replacement in [("belIeve", "believe"),
                                          ("feelIng", "feeling"),
                                          ("welI", "well"),
                                          ("wheelI've", "wheel I've"),
                                          ("LeguelIec", "Leguellec"),
                                          ("CampbelI", "Campbell"),
                                          ("telIJoe", "tell Joe"),
                                          ("tllI", "till"),
                                          ("y'alI", "y'all"),
                                          ("ﬀ", "ff")]:
                    sentence = sentence.replace(word, replacement)

            # problem of I transcribed as l
            sentence = re.sub(r"^l([NnLmDdRrTtSsKkFf])", r"i\1", sentence)
            sentence = sentence.replace("¡", "i")

            return sentence

        if sentence == REMOVE_TOKEN: return sentence
        # Rule-based clean of regular errors
        brax = lambda text: re.sub(" *[\(\[\{].*?[\)\]\}] *", "", text)
        speakers = lambda text: re.sub("^[A-Z]+\: ", "", text)
        single_quot = lambda text: re.sub(" \"$", "", text)
        any_letter = lambda text, lang: re.search(r"[a-zA-Z]", text) if lang != 'ru' else re.search(r'[А-я]+', text)
        star = lambda text: re.sub("(^ *\* *)|(\* ){1,2}| \*$", "", text)
        red_flag_chars = '♪/~'
        s = single_quot(speakers(brax(star(sentence))))
        if not any_letter(s, 'en'):
            return REMOVE_TOKEN
        if any(rfc in s for rfc in red_flag_chars):
            return REMOVE_TOKEN
        # Fix tokenizer mistakes in German
        s = common_errors(s, lang)
        return s

    @staticmethod
    def lang_check(sentence, lang):
        """ Language prediction-based clean. Is only reliable for English but English is the main offender so it works
        well in this case. Removes sentence if it's in English (and monolingual corpus is in another language)
        """
        if sentence == REMOVE_TOKEN: return sentence
        pred = cld3.get_language(sentence)
        if pred.is_reliable and lang != 'en' and pred.language == 'en':
            return REMOVE_TOKEN
        return sentence


def bitext_parse(prefix: str, tgt_lang: str) -> None:
    src_file, tgt_file = f'{prefix}.en', f'{prefix}.{tgt_lang}'
    with open(src_file) as s, open(tgt_file) as t:
        src = s.read().splitlines()
        tgt = t.read().splitlines()
    bitext = list(zip(src, tgt))
    sent_prep = SententialPreprocessor.init_from_langs(['en', tgt_lang])
    with open(f"{prefix}.preproc.en", 'w+') as s_out, open(f"{prefix}.preproc.{tgt_lang}", 'w+') as t_out:
        for i in tqdm(range(len(bitext))):
            s, t = sent_prep.preprocess(bitext[i][0], 'en'), sent_prep.preprocess(bitext[i][1], tgt_lang)
            if s != REMOVE_TOKEN and t != REMOVE_TOKEN and len(s) > 1 and len(t) > 1:
                s_out.write(s + "\n")
                t_out.write(t + "\n")
    return


if __name__ == '__main__':
    """ This script can either preprocess a monolingual dataset (documents of sentence per line, separated by newline)
        or a bitext (two paired documents). args.prefix denotes the prefix of the parsed file (e.g. for monolingual file
        "file.ru" you'd provide "--prefix file --language ru. For bitext parsing the assumption is that en is the source
        language always. 
        
        Except for bits copied from R. Bawden's script, everything here is language-independent 
        (though might work better on Latin-alphabet languages etc.)"""
    parser = ArgumentParser()
    parser.add_argument('-p', '--prefix', required=True,
                        help='Prefix of files (paths relative to master directory) to parse.')
    args = parser.parse_args()

    bitext_parse(args.prefix, 'pl')

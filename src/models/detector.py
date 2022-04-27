import spacy


class Detector(object):
    def __init__(self):
        try:
            nlp = spacy.load('pl_spacy_model_morfeusz_big')
            self.nlp = nlp
        except ValueError:
            assert hasattr(self, 'nlp')
            pass
        with open('data/stopwords', 'r') as f:
            stopwords = f.read().splitlines()
        self.stopwords = stopwords

    def parse_sentence(self, sentence):
        try:
            parsed = nlp(sentence)
        except KeyboardInterrupt:
            raise
        except:
            print("Parser threw an error.")
            return ""
        return parsed

    def calculate_type_agreement(self, sents, en_sents, type_):
        coh = Coherence()
        inv_type_ = [coh.reverse_map[x] for x in type_]
        # A list of bools depending on whether ith sentence agreed to the ith type
        correct = [self.identify_context(sents[i], en_sents[i], type_[i]) for i in tqdm(range(len(sents)))]
        incorrect = [self.identify_context(sents[i], en_sents[i], inv_type_[i]) for i in tqdm(range(len(sents)))]
        corr = {
            x: np.sum(np.array(correct)
                      & np.array([t_ in coh.types[x] for t_ in type_]))
            for x in coh.attribs
        }
        incorr = {
            x: np.sum(np.array(incorrect)
                      & np.array([t_ in coh.types[x] for t_ in inv_type_]))
            for x in coh.attribs
        }
        print(corr, incorr)
        return corr, incorr

    def identify_context(self, sentence, en_sentence, type_):
        print(type_)
        sentence = self.parse_sentence(sentence)
        if 'spgen' in type_:
            x = check_speaker_gender(sentence, en_sentence, type_, self.stopwords)
            return x
        else:
            il_identify_context = il_id(sentence, en_sentence, self.stopwords)
            return type_ in il_identify_context

    @staticmethod
    def check_speaker_gender(sentence, src_sentence, types, stopnouns):
        if 'I' not in src_sentence.split():
            return types
        for token in sentence:
            token_feats = token._.feats.split(':')
            head_feats = token.head._.feats.split(':')
            if token.head.pos_ not in ['NOUN', 'VERB', 'ADJ']: continue

            if 'sg' in token_feats and 'pri' in token_feats:
                # Past tense and future tense verbs
                if token.head.pos_ == 'VERB' and token.dep_ in ['aux:clitic', 'aux', 'aux:pass']:
                    return gender_check(token, types)

                # Nouns
                if token.head.pos_ == 'NOUN' and 'inst' in head_feats:
                    if token.dep_ in ['aux:clitic', 'cop']:
                        if no_adp(sentence, token.i, token.head.i):
                            if token.head.orth_ not in stopnouns:
                                return gender_check(token, types)

                # Adjectives
                if token.head.pos_ == 'ADJ':
                    if token.dep_ in ['aux:clitic', 'aux:pass', 'cop', 'obl:cmpr', 'obl']:
                        return gender_check(token, types)
        return types

    @staticmethod
    def no_det(sentence, token):
        """'państwo poszli' vs 'ci państwo poszli'. The latter must be recognised as wrong."""
        """Fails on 'pan takze"""
        for t in sentence:
            if t.head == token and t.dep_ == 'det':
                return False
        return True

    @staticmethod
    def no_appos(sentence, token):
        """'państwo poszli' vs 'ci państwo poszli'. The latter must be recognised as wrong."""
        for t in sentence:
            if t.head == token and t.dep_ == 'appos' \
                    and 'gen' not in t._.feats.split(':'):
                return False
        return True

    @staticmethod
    def no_nation(sentence):
        if re.findall('(countr|nation|land|state|kingdom|realm|econom|elsewhere|rule)|\b', sentence.lower()):
            return False
        return True

    @staticmethod
    def check_if_formal(sentence, src_sentence, types):
        # todo validate performance against hard cases (e.g. with the "pani" etc)
        for token in sentence:
            if token.orth_.lower() == 'proszę' and not re.findall(r'please|ask', src_sentence.lower()):
                types['FORM'] = 'form'

            if token.lemma_.lower() in ['pan', 'pani'] and no_det(sentence, token) and no_appos(sentence, token):
                types['FORM'] = 'form'
                # Check gender of interlocutor
                types = gender_check(token, types, 'ILGEN')
                # Check number of interlocutor
                types['ILNUM'] = re.findall(r'sg|pl', token._.feats)[0]
                return types, True

            elif token.lemma_.lower() == 'pański':
                types['FORM'] = 'form'
                types['ILNUM'] = 'sg'
                types['ILGEN'] = 'masc'
                return types, True

            if token.lemma_ == 'państwo' and no_det(sentence, token) and no_nation(src_sentence):
                types['FORM'] = 'form'
                types['ILNUM'] = 'pl'
                types['ILGEN'] = 'mix'
                return types, True
        return types, False

    @staticmethod
    def check_interlocutor(sentence, types, stopnouns):
        # todo implement some sort of majority vote
        for token in sentence:
            token_feats = token._.feats.split(':')
            head_feats = token.head._.feats.split(':')
            for number in ('sg', 'pl'):
                if number in head_feats and 'sec' in head_feats:
                    if token.head.pos_ in ['VERB', 'PRON']:
                        types['ILNUM'] = number
                        types['FORM'] = 'inf'
                        if token.pos_ == 'ADJ' and number in token_feats:
                            if token.dep_ in ['xcomp:pred', 'nsubj', 'conj', 'nsubj', 'iobj', 'xcomp',
                                              'amod', 'vocative', 'obl:cmpr']:
                                types = gender_check(token, types, 'ILGEN')

                        if token.pos_ == 'NOUN':
                            if token.dep_ == 'vocative' or (token.dep_ in ['appos', 'obj'] and 'voc' in token_feats):
                                ner = [a.text for a in sentence.ents]
                                if token.orth_ not in ner:
                                    types = gender_check(token, types, 'ILGEN')
            continue_check = False
            # Your/yours
            if token.lemma_.lower() == 'twój':
                types['ILNUM'] = 'sg'
                types['FORM'] = 'inf'
            if token.lemma_.lower() == 'wasz':
                types['ILNUM'] = 'pl'
                types['FORM'] = 'inf'
            for number in ('sg', 'pl'):
                if 'sec' in token_feats and number in token_feats:
                    # print(token.pos_, ' || ', token, ' || ', token.head, ' || ',  sentence, '\n')
                    if not (token.orth_ == 'ś' and sentence[token.i - 1].orth_ in ['czym', 'kim']):
                        types['ILNUM'] = number
                        types['FORM'] = 'inf'
                        continue_check = True
            if continue_check:
                # Past tense and future tense verbs
                if token.head.pos_ == 'VERB' and token.dep_ in ['aux:clitic', 'aux', 'aux:pass']:
                    types = gender_check(token.head, types, 'ILGEN')
                # Nouns
                if token.head.pos_ == 'NOUN':
                    if token.dep_ in ['aux:clitic', 'cop']:
                        if no_adp(sentence, token.i, token.head.i):
                            if token.head.orth_ not in stopnouns:
                                types = gender_check(token.head, types, 'ILGEN')
                # Adjectives
                if token.head.pos_ == 'ADJ':
                    # First 3 come from SPGEN, obl:cmpr is "takiemu jak ty"
                    # niczym w porownaniu do mocy fails this todo
                    if token.dep_ in ['aux:clitic', 'aux:pass', 'cop', 'obl:cmpr', 'obl']:
                        types = gender_check(token.head, types, 'ILGEN')
                # print(sentence, types)
        return types

    def sent_id(self, sents, stopnouns) -> dict:
        """
        Go through the sentence and identify markers for all phenomena. return them in a list.
        Markers:
        SPGEN_fem, SPGEN_masc
        ILGEN_fem, ILGEN_masc
        ILNUM_sg, ILNUM_pl
        FORM_form, FORM_inf
        Assumption: if no type is returned for a given phenomenon, then the sentence is ambivalent w.r.t. the phenomenon
        :param sentence:
        :param wordlist:
        :param nlp:
        :param stopnouns:
        :return:
        """
        sentence, src_sentence = sents['trg'], sents['src']
        types = {
            'SPGEN': None,
            'ILGEN': None,
            'FORM': None,
            'ILNUM': None
        }
        parsed = self.parse_sentence(sentence)

        """1. Check SPGEN."""
        types = self.check_speaker_gender(parsed, src_sentence, types, stopnouns)

        """2. Check formality. If sentence is matched as formal, then return the right features and quit."""
        # Lemma suggests formal addressing and no_det makes sure that there are no determinants (e.g. lady vs this lady)
        types, sent_is_formal = self.check_if_formal(parsed, src_sentence, types)
        if sent_is_formal:
            return types
        """3. If sentence did not match as formal, then keep looking for other interlocutor features. 
        If found, annotate sentence as informal."""
        types = self.check_interlocutor(parsed, types, stopnouns)
        return types

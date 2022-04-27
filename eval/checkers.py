import re


def no_adp(parsed, i, j):
    for x in range(i, j):
        if parsed[x].pos_ == 'ADP' and parsed[x].head == parsed[j]:
            return False
    return True


def gender_check(token):
    if re.findall(r'm[123]', token._.feats):
        return 'm'
    if 'f' in token._.feats:
        return 'f'
    return ''


def check_form_deps(sentence, fem_deps, masc_deps):
    for token in sentence:
        if token.lemma_.lower() in ['pan', 'pani']:
            if gender_check(token) == 'f':
                #print(f"Fem found. {token} || {sentence}")
                fem_deps += 1
            elif gender_check(token) == 'm':
                #print(f"Masc found. {token} || {sentence}")
                masc_deps += 1
        elif token.lemma_.lower() == 'pański':
            #print(f"Masc found. {token} || {sentence}")
            masc_deps += 1
    return fem_deps, masc_deps


def check_gender_deps(sentence, missing_val, fem_deps, masc_deps):
    """
    :param sentence:
    :param missing_val: ('sg', 'pri') if checking for interlocutor, else ('sg', 'sec')
    :param fem_deps:
    :param masc_deps:
    :return:
    """
    for token in sentence:
        token_feats = token._.feats.split(':')
        head_feats = token.head._.feats.split(':')
        for number, pers in missing_val + [('sg', 'ter'), ('pl', 'ter')]:
            if number in head_feats and pers in head_feats:
                if token.head.pos_ in ['VERB', 'PRON']:
                    if token.pos_ == 'ADJ' and number in token_feats:
                        if token.dep_ in ['xcomp:pred', 'nsubj', 'conj', 'iobj', 'xcomp', 'amod', 'vocative']:
                            if gender_check(token) == 'f':
                                #print(f"Fem found. {token} || {sentence}")
                                fem_deps += 1
                            elif gender_check(token) == 'm':
                                #print(f"Masc found. {token} || {sentence}")
                                masc_deps += 1

                    if token.pos_ == 'NOUN':
                        if token.dep_ == 'vocative' or (token.dep_ in ['appos', 'obj'] and 'voc' in token_feats):
                            if gender_check(token) == 'f':
                                #print(f"Fem found. {token} || {sentence}")
                                fem_deps += 1
                            elif gender_check(token) == 'm':
                                #print(f"Masc found. {token} || {sentence}")
                                masc_deps += 1
        continue_check = False
        for number, pers in missing_val + [('sg', 'ter'), ('pl', 'ter')]:
            if pers in token_feats and number in token_feats:
                if not (token.orth_ == 'ś' and sentence[token.i - 1].orth_ in ['czym', 'kim']):
                    continue_check = True
        if continue_check:
            # Past tense and future tense verbs
            if token.head.pos_ == 'VERB' and token.dep_ in ['aux:clitic', 'aux', 'aux:pass']:
                if gender_check(token.head) == 'f':
                    #print(f"Fem found. {token.head} || {sentence}")
                    fem_deps += 1
                elif gender_check(token.head) == 'm':
                    #print(f"Masc found. {token.head} || {sentence}")
                    masc_deps += 1
            # Nouns
            if token.head.pos_ == 'NOUN':
                if token.dep_ in ['aux:clitic', 'cop']:
                    if no_adp(sentence, token.i, token.head.i):
                        if gender_check(token.head) == 'f':
                            #print(f"Fem found. {token.head} || {sentence}")
                            fem_deps += 1
                        elif gender_check(token.head) == 'm':
                            #print(f"Masc found. {token.head} || {sentence}")
                            masc_deps += 1
            # Adjectives
            if token.head.pos_ == 'ADJ':
                if token.dep_ in ['aux:clitic', 'aux:pass', 'cop', 'obl:cmpr', 'obl']:
                    if gender_check(token.head) == 'f':
                        #print(f"Fem found. {token.head} || {sentence}")
                        fem_deps += 1
                    elif gender_check(token.head) == 'm':
                        #print(f"Masc found. {token.head} || {sentence}")
                        masc_deps += 1
    return fem_deps, masc_deps

def no_det(sentence, token):
    """'państwo poszli' vs 'ci państwo poszli'. The latter must be recognised as wrong."""
    """Fails on 'pan takze"""
    for t in sentence:
        if t.head == token and t.dep_ == 'det':
            return False
    return True


def no_appos(sentence, token):
    """'państwo poszli' vs 'ci państwo poszli'. The latter must be recognised as wrong."""
    for t in sentence:
        if t.head == token and t.dep_ == 'appos' \
                and 'gen' not in t._.feats.split(':'):
            return False
    return True


def no_nation(sentence):
    if re.findall('(countr|nation|land|state|kingdom|realm|econom|elsewhere|rule)|\b', sentence.lower()):
        return False
    return True


def check_form(sentence, en_sentence):
    answers = []
    for token in sentence:
        if token.orth_.lower() == 'proszę' and not re.findall(r'please|ask', en_sentence.lower()):
            answers += ['<form_f>']
        if token.lemma_.lower() in ['pan', 'pani'] and no_det(sentence, token) and no_appos(sentence, token):
            num = re.findall(r'sg|pl', token._.feats)[0]
            answers += ['<form_f>', f'<ilnum_{num[0]}>']
            # Check gender of interlocutor
            g = gender_check(token)
            if g:
                return answers + [f'<ilgen_{g}>'], True

        elif token.lemma_.lower() == 'pański':
            return ['<form_f>', '<ilnum_s>', '<ilgen_m>'], True

        if token.lemma_ == 'państwo' and no_det(sentence, token) and no_nation(en_sentence):
            return ['<form_f>', '<ilnum_p>', '<ilgen_x>'], True
    return [], False


def spgen_id(sentence, type_, stopwords):
    pttn = 'f' if type_ == '<spgen_f>' else 'm'
    for token in sentence:
        token_feats = token._.feats.split(':')
        head_feats = token.head._.feats.split(':')
        if token.head.pos_ not in ['NOUN', 'VERB', 'ADJ']: continue
        if 'sg' in token_feats and 'pri' in token_feats:
            # Past tense and future tense verbs
            if token.head.pos_ == 'VERB' and token.dep_ in ['aux:clitic', 'aux', 'aux:pass']:
                if gender_check(token.head) == pttn:
                    return True

            # Nouns
            if token.head.pos_ == 'NOUN' and 'inst' in head_feats:
                if token.dep_ in ['aux:clitic', 'cop']:
                    if no_adp(sentence, token.i, token.head.i):
                        if token.head.lemma_.lower() not in stopwords:
                            if gender_check(token.head) == pttn:
                                return True

            # Adjectives
            if token.head.pos_ == 'ADJ':
                if token.dep_ in ['aux:clitic', 'aux:pass', 'cop']:
                    if gender_check(token.head) == pttn:
                        return True
    return False


def il_id(sentence, en_sentence, stopwords):
    form_id = check_form(sentence, en_sentence)
    if form_id[1] == True:
        return form_id[0]
    answers = []
    for token in sentence:
        token_feats = token._.feats.split(':')
        head_feats = token.head._.feats.split(':')
        for number in ('sg', 'pl'):
            if number in head_feats and 'sec' in head_feats:
                if token.head.pos_ in ['VERB', 'PRON']:
                    answers = answers + [f'<ilnum_{number[0]}>', '<form_i>']

                    if token.pos_ == 'ADJ' and number in token_feats:
                        if token.dep_ in ['xcomp:pred', 'nsubj', 'conj', 'iobj', 'xcomp', 'amod', 'vocative']:
                            g = gender_check(token)
                            if g:
                                return answers + [f'<ilgen_{g}>']

                    if token.pos_ == 'NOUN':
                        if token.dep_ == 'vocative' or (token.dep_ in ['appos', 'obj'] and 'voc' in token_feats):
                            ner = [a.text for a in sentence.ents]
                            if token.orth_ not in ner:
                                if token.lemma_.lower() not in stopwords:
                                    g = gender_check(token)
                                    if g:
                                        return answers + [f'<ilgen_{g}>']

        continue_check = False
        # Your/yours
        if token.lemma_.lower() == 'twój':
            answers = answers + [f'<ilnum_s>', '<form_i>']

        if token.lemma_.lower() == 'wasz' \
                and token.head.orth_.lower() not in ['wysokość', 'świątobliwość', 'mość'] \
                and token.head.orth_[0].islower():
            answers = answers + [f'<ilnum_p>', '<form_i>']
        for number in ('sg', 'pl'):
            if 'sec' in token_feats and number in token_feats:
                if not (token.orth_ == 'ś' and sentence[token.i - 1].orth_ in ['czym', 'kim']) \
                        and token.pos_ in ['VERB', 'AUX', 'PRON']:
                    answers = answers + [f'<ilnum_{number[0]}>', '<form_i>']
                    continue_check = True

        if continue_check:
            # Past tense and future tense verbs
            if token.head.pos_ == 'VERB' and token.dep_ in ['aux:clitic', 'aux', 'aux:pass']:
                g = gender_check(token.head)
                if g:
                    return answers + [f'<ilgen_{g}>']
            # Nouns
            if token.head.pos_ == 'NOUN':
                if token.dep_ in ['aux:clitic', 'cop']:
                    if no_adp(sentence, token.i, token.head.i):
                        if token.head.lemma_.lower() not in stopwords:
                            g = gender_check(token.head)
                            if g:
                                return answers + [f'<ilgen_{g}>']
            # Adjectives
            if token.head.pos_ == 'ADJ':
                if token.dep_ in ['aux:clitic', 'aux:pass', 'cop', 'obl:cmpr', 'obl']:
                    g = gender_check(token.head)
                    if g:
                        return answers + [f'<ilgen_{g}>']
    return answers

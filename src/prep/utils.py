import math
from typing import List
import logging
import re
import pandas as pd
import sentencepiece as spm

logging.basicConfig(level=logging.INFO)


def get_blacklist(path: str) -> List[str]:
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print("No blacklist file found (we need one before corpus is extracted!). Quitting")
        exit(1)
    return lines


def time_converter(time_str):
    time_str = time_str.replace(',', ':').replace('.', ':').replace(' ', '')
    time_str = re.split(r'[^0-9]', time_str)
    # Bugproofing
    if len(time_str) < 4:
        time_str.append('000')
    try:
        hours, mins, secs, msecs = list(time_str)
    except:
        print("Can't unpack values correctly")
        hours, mins, secs, msecs = ['00', '00', '00', '00']
    msecs = int(msecs) + int(hours) * 3600000 + int(mins) * 60000 + int(secs) * 1000

    return msecs


def parse_subtitles(tree_root, return_type=dict()):
    """
    Extract subtitles from xml files as text
    :param tree_root: root of the xml tree
    :return: subtitles : a dictionary where key is subtitle ID and value is text and timestamps
    """
    time_start = -1
    sub_count = 0
    group_buffer = []
    # Making a nan array to store subs
    subtitles = dict() if return_type == dict() else []
    for sub in tree_root:
        if sub.tag == 's':
            # Check for time start
            if sub[0].tag == 'time':
                time_start = time_converter(sub[0].attrib['value'])
                sub_count = 1
            else:
                sub_count += 1
            if sub[-1].tag == 'time':
                time_end = time_converter(sub[-1].attrib['value'])
            else:
                time_end = -1
            # Collecting subtitles
            single_buffer = ""
            for element in sub:
                if element.tag == 'w':
                    single_buffer = single_buffer + ' ' + element.text
            group_buffer.append((single_buffer, sub.attrib['id']))
            # Subtitles collected. Flush with time stamps if done
            if time_end != -1:
                duration = time_end - time_start
                fragment = math.floor(duration / sub_count)
                # Assigning time fragments to subs
                stamp = time_start
                for single_sub, sub_id in group_buffer:
                    if return_type == dict():
                        subtitles[sub_id] = (single_sub, stamp, stamp + fragment - 80)
                    else:
                        subtitles.append((single_sub, stamp, stamp + fragment - 80))
                    stamp = stamp + fragment + 80
                group_buffer = []
    # Bugproofing: if last sub is not closed
    if group_buffer:
        print(type(subtitles))
        time_end = time_start + 1000
        duration = time_end - time_start
        fragment = math.floor(duration / sub_count)
        for single_sub, sub_id in group_buffer:
            print(f'{single_sub=}', f'{sub_id=}')
            if return_type == dict():
                subtitles[sub_id] = (single_sub, stamp, stamp + fragment - 80)
            else:
                subtitles.append((single_sub, stamp, stamp + fragment - 80))
            stamp = stamp + fragment + 80
        group_buffer = []
    return subtitles


def write_to_file_extractor(filename, subs, indices):
    """

    :param filename: name of file to write to
    :param subs: dictionary containing subtitles
    :param indices: a list of idcs (str) to access subtitles from subs
    :return:
    """
    with open(filename, 'a+') as f:
        buffer = ''
        for index in indices:
            try:
                buffer = buffer + subs[index][0]
            except KeyError:
                buffer = buffer + "-"
        f.write(buffer + '\n')
    return


def preprocess(df):
    # Remove empty sentences
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(subset=['en', 'pl'], inplace=True)

    # Randomise row order
    df = df.sample(frac=1)
    # Preprocess data: 100 tokens max (to fit model)
    df['en_len'] = df['en'].str.count(' ').astype('int64')
    df['pl_len'] = df['pl'].str.count(' ').astype('int64')
    df = df.query('en_len < 100 & pl_len < 100')
    return df[['en', 'pl']]


def make_df(in_en, in_pl):
    en = pd.DataFrame(in_en, dtype="string")
    pl = pd.DataFrame(in_pl, dtype="string")
    df = pd.concat([en, pl], axis=1, join='outer', ignore_index=True)
    df.columns = ['en', 'pl']
    return df

# def save_leftovers(train, leftover_train, dev, test, prefix):
#     train.append(leftover_train)
#     train[SETS] = train['cxt'].str.split(',', expand=True)
#     print(f'{train.shape=}')
#     print(train)
#     # Getting rid of some of the more popular scenarios because we don't need that many examples!
#     train['ilnum_s_and_form_i'] = (train['ilnum'] == 'ilnum_s') & (train['form'] == 'form_i') & \
#                                   (train['spgen'] == '') & (train['ilgen'] == '')
#
#     print(f"{train[train['ilnum'] == 'ilnum_s'].shape=}")
#     print(f"{train[train['form'] == 'form_i'].shape=}")
#     print(f"{train[(train['form'] == 'form_i') & (train['ilnum'] == 'ilnum_s')].shape=}")
#
#     # Removing all but 5% of ilnum_s and form_i
#     train_ = train[train['ilnum_s_and_form_i']].sample(frac=0.05)
#     train = pd.concat((train[train['ilnum_s_and_form_i'] == False], train_))
#     print(f'{train.shape = }')
#     print(f"{train[train['ilnum'] != ''].shape=}")
#     print(f"{train[train['form'] != ''].shape=}")
#     print(f"{train[train['spgen'] != ''].shape=}")
#     print(f"{train[train['ilgen'] != ''].shape=}")
#
#     dfs = {f'{prefix}.train.en': train['en'],
#            f'{prefix}.train.pl': train['pl'],
#            f'{prefix}.train.cxt': train['cxt'],
#            f'{prefix}.dev.en': dev['en'],
#            f'{prefix}.dev.pl': dev['pl'],
#            f'{prefix}.dev.cxt': dev['cxt'],
#            f'{prefix}.dev.marking': dev['marking'],
#            f'{prefix}.test.en': test['en'],
#            f'{prefix}.test.pl': test['pl'],
#            f'{prefix}.test.cxt': test['cxt'],
#            f'{prefix}.test.marking': test['marking']}
#     return dfs


def train_spm(sentence_list: List[str]) -> None:
    train_data = 'data/sentencepiece/spm.training_data'
    with open('data/sentencepiece/spm.training_data', 'w+') as f:
        print(len(sentence_list))
        for line in sentence_list:
            f.write(line + "\n")
    spm.SentencePieceTrainer.train(input=train_data,
                                   model_prefix=f'data/sentencepiece/spm', vocab_size=16000,
                                   character_coverage=0.9995,
                                   input_sentence_size=len(sentence_list),
                                   shuffle_input_sentence=True)


def read_from_file(filename, config='pretrain'):
    with open(filename + '.en', 'r') as s, open(filename + '.pl', 'r') as t:
        df = pd.concat([pd.DataFrame(s.read().splitlines()),
                        pd.DataFrame(t.read().splitlines())], axis=1, join='outer', ignore_index=True)
    df.columns = ['en', 'pl']
    if config != 'pretrain':
        with open(filename + '.cxt', 'r') as c:
            lines = c.read().splitlines()
            df = pd.concat([df, pd.DataFrame(lines)], axis=1, join='outer', ignore_index=True)
        df.columns = ['en', 'pl', 'cxt']
    return df


def write_to_files(dfs, prefix=''):
    for k, v in dfs.items():
        with open(f'./data/qualitative/{prefix}{k}', 'w+') as f:
            f.writelines('\n'.join(v.tolist()))


def write_to_file(contents, filename):
    with open(filename + '.en', 'w+') as s, open(filename + '.pl', 'w+') as t:
        s.writelines('\n'.join(contents['en'].tolist()))
        t.writelines('\n'.join(contents['pl'].tolist()))

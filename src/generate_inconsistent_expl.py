# -*- coding: utf-8 -*-

__license__ = "Apache License"
__version__ = "2021.1"
__date__ = "25 08 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"


import typing
import re
import requests
import os
import argparse
import pandas as pd
import copy
import inflect
import nltk
import json

from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from tqdm import tqdm
from typing import Text

# 0. Patterns
not_p = re.compile(' not |Not | no ')
cannot_p = re.compile(" cannot | can't")
is_p = re.compile(" isn't ")
does_p = re.compile(" doesn't ")
do_p = re.compile(" don't ")
are_p = re.compile(" aren't ")
double_space = re.compile('\s\s+')
space_dot = re.compile('\s\.')

# 1. Download NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def return_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../resources/esnli',
                        help='directory path where reverse datasets are located')

    parser.add_argument('--save_dir', type=str, default='../resources/esnli',
                        help='save directory path to save generated inconsistent explanations')

    parser.add_argument('--dataset', type=str, default='esnli',
                        choices=['esnli', 'cose1.0'])
    return parser.parse_args()


def is_contain_number(word: Text):
    if re.findall(pattern='\\d+', string=word):
        return True
    return False


def extract_instance(df: pd.DataFrame):
    new_outp = []
    for id_, datum in tqdm(df.iterrows(), total=len(df)):
        subject, relation, object = datum['subject'], datum['relation'], datum['object']
        if is_contain_number(subject) or is_contain_number(object):
            continue
        elif len(subject) <= 2 or len(object) <= 2:
            continue
        elif subject == object:
            continue
        elif len(subject.split(' ')) > 3 or len(object.split(' ')) > 3:
            continue
        else:
            save_dict_inst = {
                    "subject": subject,
                    "relation": relation,
                    "object": object
                }
            new_outp.append(json.dumps(save_dict_inst))
    new_outp = list(set(new_outp))
    new_outp = [json.loads(d) for d in new_outp]
    return new_outp


def load_cn():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    cn_path = os.path.join(dir_path, '../resources/conceptnet_en_5.7.0.tsv')
    cn = pd.read_csv(cn_path, sep='\t')

    # Build search dictionary
    search_dict = {}
    for relation in ['Antonym', 'DistinctFrom', 'HasA', 'IsA', 'Synonym']:
        partial_df_ = cn[cn['relation'] == relation]
        partial_df_ = partial_df_.dropna()
        for id_, datum in tqdm(partial_df_.iterrows(), total=len(partial_df_)):
            subject, relation, object = datum['subject'], datum['relation'], datum['object']
            if is_contain_number(subject) or is_contain_number(object):
                continue
            elif len(subject) <= 2 or len(object) <= 2:
                continue
            elif subject == object:
                continue
            elif len(subject.split(' ')) > 3 or len(object.split(' ')) > 3:
                continue
            else:
                if search_dict.get(subject):
                    if search_dict.get(subject).get(relation):
                        search_dict[subject][relation].append(object)
                    else:
                        search_dict[subject][relation] = [object]
                else:
                    search_dict[subject] = {}
                    if search_dict.get(subject).get(relation):
                        search_dict[subject][relation].append(object)
                    else:
                        search_dict[subject][relation] = [object]
    return search_dict


def add_negation(text: typing.Text, tgt_word: typing.Text):
    def check_be_cond(a_part: typing.Text):
        special_words = ['they', 'there', 'it', 'that']
        tags_a = pos_tag(word_tokenize(a_part))
        if tags_a[-1][1].startswith('NN') or tags_a[-1][1] in ['WP', 'WDT']:
            return True
        if tags_a[-1][0].lower() in special_words:
            return True
        return False

    def check_have_cond(b_part: typing.Text):
        tags = pos_tag(word_tokenize(b_part))
        word, tag = tags[0]
        if tag.startswith('NN'):
            return True
        if tag in ['CD', 'DT']:
            return True
        return False

    splits = text.split(tgt_word)
    outp = list()
    for i in range(len(splits) - 1):
        if tgt_word in [' is ', ' are ']:
            a_part = splits[i]
            if check_be_cond(a_part):
                original_phrase = splits[i] + f' {tgt_word.strip()} ' + splits[i + 1]
                repl_phrase1 = splits[i] + f' {tgt_word.strip()} not ' + splits[i + 1]
                repl_phrase2 = splits[i] + f" {tgt_word.strip()}n't " + splits[i + 1]

                text1_ = text.replace(original_phrase, repl_phrase1)
                text2_ = text.replace(original_phrase, repl_phrase2)
                outp += [text1_, text2_]

        if tgt_word in [' has ', ' have ']:
            b_part = splits[i + 1]
            if check_have_cond(b_part):
                original_phrase = splits[i] + f' {tgt_word.strip()} ' + splits[i + 1]
                if tgt_word == ' has ':
                    repl_phrase1 = splits[i] + f' does not have ' + splits[i + 1]
                    repl_phrase2 = splits[i] + f" doesn't have " + splits[i + 1]
                else:
                    repl_phrase1 = splits[i] + f' do not have ' + splits[i + 1]
                    repl_phrase2 = splits[i] + f" don't have " + splits[i + 1]

                text1_ = text.replace(original_phrase, repl_phrase1)
                text2_ = text.replace(original_phrase, repl_phrase2)
                outp += [text1_, text2_]
    return outp


def remove_negation(text: typing.Text):
    not_p = re.compile(' not |Not | no ')
    cannot_p = re.compile(" cannot | can't")
    is_p = re.compile(" isn't ")
    does_p = re.compile(" doesn't ")
    do_p = re.compile(" don't ")
    are_p = re.compile(" aren't ")

    def replace_pattern(text: typing.Text, pattern, repl_word: typing.Text):
        outp = list()
        if pattern.findall(text):
            for p_ in pattern.findall(text):
                splits = text.split(p_)
                tmp_ = []
                for i in range(len(splits) - 1):
                    original_text = splits[i] + p_ + splits[i + 1]
                    repl_text = splits[i] + repl_word + splits[i + 1]
                    print()
                    tmp_.append(text.replace(original_text, repl_text))
                outp += list(set(tmp_))
        return outp

    outp = []
    if not_p.findall(text):
        outp += replace_pattern(text, not_p, ' ')
    if cannot_p.findall(text):
        outp += replace_pattern(text, cannot_p, ' can ')
    if is_p.findall(text):
        outp += replace_pattern(text, is_p, ' is ')
    if does_p.findall(text):
        outp += replace_pattern(text, does_p, ' does ')
    if do_p.findall(text):
        outp += replace_pattern(text, do_p, ' do ')
    if are_p.findall(text):
        outp += replace_pattern(text, are_p, ' are ')
    return outp


# 1. Negation
def negation(text: typing.Text):
    outp = []
    # 1) remove negation
    neg_removed = remove_negation(text)
    if neg_removed:
        outp += neg_removed

    # 2) add negation
    if ' is ' in text:
        outp += add_negation(text, ' is ')

    if ' are ' in text:
        outp += add_negation(text, ' are ')

    if ' has ' in text:
        outp += add_negation(text, ' has ')

    if ' have ' in text:
        outp += add_negation(text, ' have ')
    return outp


# 2. Antonym
# Use only Adverb and Adjective for antonyms
def antonyms_wn(word: typing.Text):
    """
    Extract antonym from WordNet
    :param word:
    :return:
    """
    def check_pos(antonym_synsets: typing.List):
        pos_list = [wn.ADV, wn.ADJ, 's']  # s: satelite-adj
        cnt = 0
        for p in pos_list:
            if p in [ss.pos() for ss in antonym_synsets]:
                cnt += 1
        if cnt:
            return True
        else:
            return False

    antonyms = set()
    for ss in wn.synsets(word):
        for lemma in ss.lemmas():
            any_pos_antonyms = [antonym.name() for antonym in lemma.antonyms()]
            for antonym in any_pos_antonyms:
                antonym_synsets = wn.synsets(antonym)
                if not check_pos(antonym_synsets):
                    continue
                antonyms.add(antonym)
    return antonyms


# use conceptnet 5.7.0
def extract_from_cn(word: typing.Text, relation: typing.Text, search_dict: typing.Dict):
    assert relation in ['Antonym', 'DistinctFrom', 'HasA', 'IsA', 'Synonym']
    outp = set()
    rm_pattern = re.compile('a |A |an |An |')
    response = search_dict.get(word)
    if response is not None:
        result_ = response.get(relation)
        if result_ is not None:
            outp = set([rm_pattern.sub('', o) for o in result_])
    return outp


def get_synonyms_from_google(word: typing.Text):
    outp = list()
    api = f"https://api.dictionaryapi.dev/api/v2/entries/en_US/{word}"
    res = requests.get(api)
    if res.status_code != 200:
        return outp

    for r in res.json():
        try:
            meaning = r.get('meanings')
            if meaning:
                for m in meaning:
                    definitions = m.get('definitions')
                    if definitions:
                        for d in definitions:
                            synonyms = d.get('synonyms')
                            if synonyms:
                                outp += synonyms
        except AttributeError:
            pass
    # remove not single token words
    outp = [o for o in outp if len(o.split(" ")) == 1]
    return outp


def replace_antonym(text: typing.Text, search_dict: typing.Dict):
    tag_use = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS"}
    tags = pos_tag(word_tokenize(text))
    antonym_dict = {}
    change_idx = []

    for i, (w, t) in enumerate(tags):
        if t in tag_use:  # if tag is included in tag_use
            # 1. get antonym
            antonym = antonyms_wn(w)
            antonym_cn = extract_from_cn(w, 'Antonym', search_dict)
            antonym = set(list(antonym) + list(antonym_cn))  # add both wordnet and concept net

            # Filter synonyms
            # syns = set(extract_from_cn(w, 'Synonym', search_dict))
            syns = set(get_synonyms_from_google(w))  # use independent synonym dictionary

            intersect = antonym.intersection(syns)
            antonym -= intersect
            # get antonym having same the pos-tag
            ant_tags = [pos_tag(word_tokenize(s))[0] for s in antonym]
            antonym = [a[0] for a in ant_tags if a[1] == t]

            # 2. get synonym and negate
            synonym = list(extract_from_cn(w, 'Synonym', search_dict))

            # get synonym having the same pos-tag
            syn_tags = [pos_tag(word_tokenize(s))[0] for s in synonym]
            synonym = [a[0] for a in syn_tags if a[1] == t]
            # insert word itself as a synonym
            synonym.append(w)
            synonym = list(set(synonym))

            neg_synonym = [f'not {syn_}' for syn_ in synonym] if synonym else []
            if antonym:
                antonym_dict[w] = list(antonym) + neg_synonym
            change_idx.append(i)

    output = []
    if change_idx:
        words = [t[0] for t in tags]
        # replace only one antonym at a time
        for i, w in enumerate(words):
            if i in change_idx:
                value = antonym_dict.get(w)
                if value:
                    for v in value:
                        new_words = copy.deepcopy(words)
                        new_words[i] = v
                        new_text = ' '.join(new_words)
                        new_text = new_text.replace(' .', '.')
                        # post process
                        if ' a not ' in new_text:
                            new_text = new_text.replace(' a not ', ' not a ')

                        if ' an not ' in new_text:
                            new_text = new_text.replace(' an not ', ' not an ')

                        output.append(new_text)
    return list(set(output))


# 3. Last Noun
def replace_last_noun(text: typing.Text, search_dict: typing.Dict):
    engine = inflect.engine()
    outp = []
    tags = pos_tag(word_tokenize(text))
    for i, (w, t) in enumerate(tags):
        if t.startswith('NN'):
            if i + 1 >= len(tags) - 1: # last word (or before .)
                # 1. Extract Antonym/DistinctFrom/HasA relation from dictionary
                ants = list(extract_from_cn(w, 'Antonym', search_dict))
                distincts = list(extract_from_cn(w, 'DistinctFrom', search_dict))
                hasa = list(extract_from_cn(w, 'HasA', search_dict))
                unrel_words = set(ants + distincts + hasa)

                # Filter synonyms
                syns = extract_from_cn(w, 'Synonym', search_dict)
                # add plural and singular forms as synonym
                if engine.plural(w):
                    syns.add(engine.plural(w))
                if engine.singular_noun(w):
                    syns.add(engine.singular_noun(w))

                intersect = unrel_words.intersection(syns)
                unrel_words -= intersect

                # 2. generate new word
                if unrel_words:
                    words_ = [e[0] for e in tags]
                    for word_ in unrel_words:
                        words_[i] = word_
                        new_text = " ".join(words_)
                        new_text = new_text.replace(" .", ".")
                        if new_text != text:
                            outp.append(new_text)
    return list(set(outp))


def split_text(df: pd.DataFrame) -> typing.Sequence[typing.List]:
    expls = [s.split("Explanation:")[1].strip() for s in df['input'].tolist()]
    contexts = [s.split("Explanation:")[0].strip() for s in df['input'].tolist()]
    variables = df['label'].tolist()
    return contexts, variables, expls


def main(args):

    data_dir = args.data_dir
    save_dir = args.save_dir

    if args.dataset == 'esnli':
        test_df = pd.read_csv(os.path.join(data_dir, f'reverse_test.tsv'), sep="\t")
    else:
        test_df = pd.read_csv(os.path.join(data_dir, f'dev.tsv'), sep="\t")

    context_part, variable_part, generated_expl = split_text(test_df)
    pair_id = test_df['pairID'].tolist()

    cn_search_set = load_cn()

    new_pair_id, new_context, inconsist_expl, tags = list(), list(), list(), list()
    for i, e in tqdm(enumerate(generated_expl), total=len(generated_expl)):
        if e.isupper():  # if all characters are uppder case
            e = e.lower()

        # 1. negation
        neg_e = negation(e)
        if neg_e:
            if neg_e.__class__ == str:
                inconsist_expl.append(neg_e)
                new_pair_id.append(pair_id[i])
                new_context.append(context_part[i])
                tags.append('negation')
            elif neg_e.__class__ == list:
                inconsist_expl += neg_e
                new_pair_id += [pair_id[i]] * len(neg_e)
                new_context += [context_part[i]] * len(neg_e)
                tags += ['negation'] * len(neg_e)
            else:
                pass

        # 2. antonym (WordNet, ConceptNet)
        ant_e = replace_antonym(e, cn_search_set)
        if ant_e:
            inconsist_expl += ant_e
            new_pair_id += [pair_id[i]] * len(ant_e)
            new_context += [context_part[i]] * len(ant_e)
            tags += ['antonym'] * len(ant_e)

        # 3. Unrelated noun
        lm_la = replace_last_noun(e, cn_search_set)
        if lm_la:
            inconsist_expl += lm_la
            new_pair_id += [pair_id[i]] * len(lm_la)
            new_context += [context_part[i]] * len(lm_la)
            tags += ['unrel_noun'] * len(lm_la)

    outp = {
        "pair_id": new_pair_id,
        "context": new_context,
        "inconsist_expl": inconsist_expl,
        'tags': tags
    }

    assert len(new_pair_id) == len(inconsist_expl)

    save_file_name = "inconsist-expls-test.json"
    os.makedirs(os.path.join(save_dir), exist_ok=True)
    with open(os.path.join(save_dir, save_file_name), 'w', encoding='utf-8') as saveFile:
        json.dump(outp, saveFile)


if __name__ == '__main__':
    args = return_args()
    main(args)



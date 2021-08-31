# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        Copyright (c) -2021 - Mtumbuka F. M and Jang M.                                         #
#        All rights reserved.                                                                       #
#                                                                                                   #
#        Redistribution and use in source and binary forms, with or without modification, are       #
#        permitted provided that the following conditions are met:                                  #    
#        1. Redistributions of source code must retain the above copyright notice, this list of     #
#           conditions and the following disclaimer.                                                #
#        2. Redistributions in binary form must reproduce the above copyright notice, this list of  #
#           conditions and the following disclaimer in the documentation and/or other materials     #
#           provided with the distribution.                                                         #
#                                                                                                   #
#        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY      #
#        EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    #
#        MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE #
#        COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,   #
#        EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF         #
#        SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     #
#        HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR   #
#        TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         #
#        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                               #
#                                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


__license__ = "BSD-2-Clause"
__version__ = "2021.1"
__date__ = "30 03 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"

import pandas as pd
import os
import typing
import requests
import pickle
import re
import argparse
from nltk.corpus import wordnet as wn
from textblob import TextBlob
from collections import Counter
from tqdm import tqdm

neg_patterns = re.compile(" not |Not | cannot | can't | isn't | doesn't | don't | aren't | no ")


def return_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='esnli', choices=['esnli', 'comve', 'cose1.0'])
    parser.add_argument('--model_type', type=str, default='t5-base')

    return parser.parse_args()


# functions
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


# ----------------------------------------------------------------------------------------
# 1. Both explanations contain the same negation expression
def extract_both_neg_idx(df: pd.DataFrame) -> typing.List:
    """
    This function returns index where both explanations contain the same negation expression
    Args:
        df: pandas dataframe

    Returns: list of index

    """
    outp = list()
    orin_expls = df['original_expl'].tolist()
    reve_expls = df['reverse_expl'].tolist()

    for i, (o_e, r_e) in enumerate(zip(orin_expls, reve_expls)):
        origin_neg = neg_patterns.findall(o_e)
        if origin_neg:
            reverse_neg = neg_patterns.findall(r_e)
            if reverse_neg:
                outp.append(i)
    return outp


# 2. Replaced words are not the last word in LM case
def extract_not_last_word_idx(df: pd.DataFrame) -> typing.List:
    """
    This function returns the index where replaced word is not the last word for LM case
    Args:
        df: pandas dataframe

    Returns: list of index

    """
    outp = list()
    original_expl = df['original_expl'].tolist()
    reverse_expl = df['reverse_expl'].tolist()
    tags = df['tags'].tolist()

    for i, (oe, re, t) in tqdm(enumerate(zip(original_expl, reverse_expl, tags)), total=len(tags)):
        if t != 'lm':
            continue
        oe = oe.split(" ")
        re = re.split(" ")
        diff_idx = [j for j in range(len(oe)) if oe[j] != re[j]][0]

        if diff_idx + 1 != len(oe):
            outp.append(i)
    return outp


# 3. Remove synonym and hypernym words
def extract_related_word_idx(df: pd.DataFrame, word_dict: typing.Dict) -> typing.List:
    """
    Returns index where replaced word is synonym or hypernym of the original word (only for LM)
    Args:
        df: pandas data frame
        word_dict: dictionary of related words

    Returns: list of index

    """
    outp = list()
    original_expl = df['original_expl'].tolist()
    reverse_expl = df['reverse_expl'].tolist()
    tags = df['tags'].tolist()

    for i, (oe, re, t) in tqdm(enumerate(zip(original_expl, reverse_expl, tags)), total=len(tags)):
        if t != 'lm':
            continue
        oe = oe.split(" ")
        re = re.split(" ")
        diff_idx = [j for j in range(len(oe)) if oe[j] != re[j]][0]

        oe_word = oe[diff_idx].replace(".", "").strip()
        re_word = re[diff_idx].replace(".", "").strip()

        blob = TextBlob(oe_word)  # transform to singular form
        oe_word_singular = str(blob.words[0].singularize())

        blob = TextBlob(re_word)  # transform to singular form
        re_word_singular = str(blob.words[0].singularize())

        syn_hyp_dict = word_dict.get(oe_word_singular)
        if syn_hyp_dict.get('synonym') and syn_hyp_dict.get('hypernym'):
            if re_word_singular in syn_hyp_dict['synonym'] or re_word_singular in syn_hyp_dict['hypernym']:
                outp.append(i)
                continue
    return outp


# 4. extract human-filtered pattern index
def extract_pattern_idx(df: pd.DataFrame, pattern_dict: typing.Dict) -> typing.List:
    """
    Returns index where replaced word is synonym or hypernym of the original word
    Args:
        df: pandas data frame
        word_dict: dictionary of related words

    Returns: list of index

    """
    outp = list()
    original_expl = df['original_expl'].tolist()
    reverse_expl = df['reverse_expl'].tolist()
    tags = df['tags'].tolist()

    templates = pattern_dict['template']
    word_pair_dict = pattern_dict['word-pair']

    for i, (o_e, r_e, t) in tqdm(enumerate(zip(original_expl, reverse_expl, tags)), total=len(tags)):
        if t == 'negation':
            continue

        for tem in templates:
            p = re.compile(tem)
            if p.findall(o_e) and p.findall(r_e):
                outp.append(i)
                break

        o_e = o_e.split(" ")
        r_e = r_e.split(" ")

        try:
            diff_idx = [j for j in range(len(o_e)) if o_e[j] != r_e[j]][0]
            oe_word = o_e[diff_idx].replace(".", "").strip()
            re_word = r_e[diff_idx].replace(".", "").strip()

            word_list = word_pair_dict.get(oe_word)
            if word_list and re_word in word_list:
                outp.append(i)
        except IndexError:
            continue
    return list(set(outp))


def calculate_sample_size(df):
    tags = df['tags'].tolist()
    cnt = Counter(tags)
    return [cnt['negation'], cnt['antonym'], cnt['noun_antonym']]


def main(args):
    if args.previous:
        file_name = 'inconsist-expl-extracted-previous.dict'
    else:
        file_name = 'inconsist-expl-extracted.dict'
    args.result_dir = f'result/{args.dataset}'

    with open(os.path.join(args.result_dir, args.model_type, file_name), 'rb') as loadFile:
        data = pickle.load(loadFile)

    # with open(os.path.join('wt5/result/esnli', 't5-base', file_name), 'rb') as loadFile:
    #     data = pickle.load(loadFile)

    df = pd.DataFrame(data)
    # remove duplicates
    df = df.drop_duplicates(ignore_index=True)
    origin_n, origin_a, origin_l = calculate_sample_size(df)
    print(f"Negation: {origin_n} | Antonym: {origin_a} | Last Noun: {origin_l}")

    # 1. remove both negation
    drop_idx = extract_both_neg_idx(df)

    # print removed explanations
    # for idx in drop_idx:
    #     inst = df.iloc[idx]
    #     origin_expl = inst['original_expl']
    #     reverse_expl = inst['reverse_expl']
    #     print(f"{origin_expl} | {reverse_expl}")

    df = df.drop(drop_idx).reset_index(drop=True)
    neg_n, neg_a, neg_l = calculate_sample_size(df)
    neg_diff, ant_diff, lm_diff = origin_n - neg_n, origin_a - neg_a, origin_l - neg_l
    print(f"Removed both negation | Negation: {neg_diff} | Antonym: {ant_diff} | Noun: {lm_diff}")

    # 2. remove not last word
    # drop_idx = extract_not_last_word_idx(df)
    # df = df.drop(drop_idx).reset_index(drop=True)
    # last_n, last_a, last_l = calculate_sample_size(df)
    # neg_diff, ant_diff, lm_diff = neg_n - last_n, neg_a - last_a, neg_l - last_l
    # print(f"Removed not last word | Negation: {neg_diff} | Antonym: {ant_diff} | LM: {lm_diff}")

    # 3. remove synonym words
    # if args.build_dict:
    #     related_word_dict = build_synonym_hypernym_dict(df, filter_tag=[wn.NOUN])
    #     with open('./lm_related_word_dict.yml', 'w') as saveFile:
    #         yaml.dump(related_word_dict, saveFile)
    # else:
    #     with open('./lm_related_word_dict.yml', 'r') as loadFile:
    #         related_word_dict = yaml.load(loadFile, Loader=yaml.SafeLoader)
    #
    # drop_idx = extract_related_word_idx(df, related_word_dict)
    # df = df.drop(drop_idx).reset_index(drop=True)
    # syn_n, syn_a, syn_l = calculate_sample_size(df)
    # neg_diff, ant_diff, lm_diff = last_n - syn_n, last_a - syn_a, last_l - syn_l
    # print(f"Removed synonyms | Negation: {neg_diff} | Antonym: {ant_diff} | LM: {lm_diff}")

    # 4. remove human-filter patterns
    # with open('./remove_pattern.yml', 'r') as loadFile:
    #     rm_pattern = yaml.load(loadFile, Loader=yaml.SafeLoader)
    #
    # drop_idx = extract_pattern_idx(df, rm_pattern)
    # df = df.drop(drop_idx).reset_index(drop=True)
    # hum_n, hum_a, hum_l = calculate_sample_size(df)
    # neg_diff, ant_diff, lm_diff = neg_n - hum_n, neg_a - hum_a, neg_l - hum_l
    # print(f"Removed human-filter | Negation: {neg_diff} | Antonym: {ant_diff} | Last: {lm_diff}")
    if args.previous:
        save_name = 'final_inconsist_expls-previous.tsv'
    else:
        save_name = 'final_inconsist_expls.tsv'

    if os.path.isfile(os.path.join(args.result_dir, args.model_type, save_name)):
        os.remove(os.path.join(args.result_dir, args.model_type, save_name))
    df.to_csv(os.path.join(args.result_dir, args.model_type, save_name), sep='\t', index=False)

    final_n, final_a, final_l = calculate_sample_size(df)
    print(f"Final | Negation: {final_n} | Antonym: {final_a} | Last: {final_l}")


if __name__ == '__main__':
    args = return_args()
    main(args)


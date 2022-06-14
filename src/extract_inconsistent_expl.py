# -*- coding: utf-8 -*-

__license__ = "Apache License"
__version__ = "2021.1"
__date__ = "25 08 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"


import re
import json
import os
import yaml
import argparse
import pandas as pd
import typing
import editdistance

from tqdm import tqdm


neg_patterns = re.compile(" not |Not | cannot | can't | isn't | doesn't | don't | aren't | no ")


def return_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='esnli', choices=['esnli', 'cose1.0', 'esnli_ve', 'esnli_nile',
                                                                           "cose_cage"])

    parser.add_argument('--data_dir', type=str, default='../resources/esnli_sample',
                        help='directory path where inconsistent explanations for step 2 and step 4 are located')

    parser.add_argument('--save_dir', type=str, default='../resources/esnli_sample',
                        help='directory path to save the final results')

    parser.add_argument('--edit_dist_threshold', type=int, default=1,
                        help="thresholds of edit-distrance to match")
    return parser.parse_args()


def construct_search_set(args):

    # load inconsistent explanations
    load_filename = f'inconsist-expls-test.json'

    with open(os.path.join(args.data_dir, load_filename), 'rb') as loadFile:
        generated_expls = json.load(loadFile)

    expl = generated_expls['inconsist_expl']
    tags = generated_expls['tags']
    pair_id = generated_expls['pair_id']

    expl_search_dict = {}
    for idx, e, t in zip(pair_id, expl, tags):
        if expl_search_dict.get(idx):
            expl_search_dict[idx]['expl'].append(e)
            expl_search_dict[idx]['tags'].append(t)
        else:
            expl_search_dict[idx] = {
                "expl": [e],
                "tags": [t]
            }
    return expl_search_dict


def match_candidates(args) -> typing.Dict:
    expl_search_dict = construct_search_set(args)

    # load generated explanations
    load_filename = f'inconsist-final-test.json'
    with open(os.path.join(args.data_dir, load_filename), 'rb') as loadFile:
        predicted_expls = json.load(loadFile)

    pair_id = predicted_expls['pair_id']
    inconsist_expl = predicted_expls['inconsist_expl']
    context_part = predicted_expls['context']
    variable_part = predicted_expls['variable']
    labels = predicted_expls['label']
    assert len(pair_id) == len(inconsist_expl)

    # search
    extracted_idx, tags = [], []
    for i, e in tqdm(enumerate(inconsist_expl), total=len(inconsist_expl)):
        idx_ = pair_id[i]
        expl_set = expl_search_dict.get(idx_)
        edit_dist = [editdistance.eval(e.lower(), e_s.lower()) for e_s in expl_set['expl']]
        if min(edit_dist) <= args.edit_dist_threshold:
            min_edit_idx = edit_dist.index(min(edit_dist))
            extracted_idx.append(i)
            tags.append(expl_set['tags'][min_edit_idx])

    print(f"Total {len(extracted_idx)} inconsistent explanations are confirmed")
    if args.dataset == 'cose1.0':
        reverse_data_df = pd.read_csv(os.path.join(args.data_dir, 'reverse_dev.tsv'), sep='\t')
    else:
        reverse_data_df = pd.read_csv(os.path.join(args.data_dir, 'reverse_test.tsv'), sep='\t')
    original_pair_id = reverse_data_df['pairID'].tolist()
    original_hypo = reverse_data_df['label'].tolist()
    original_expl = [s.split("Explanation:")[1].strip() for s in reverse_data_df['input'].tolist()]

    # Load original label -------------------------------------------------------------------------
    # This part could be changed according to the file format of the original test data
    if 'esnli' in args.dataset:
        original_data_df = pd.read_csv(os.path.join(args.data_dir, 'test.tsv'), sep='\t')
        original_label = original_data_df['label'].tolist()
    elif 'cose' in args.dataset:
        original_data_df = pd.read_csv(os.path.join(args.data_dir, 'dev.csv'), sep=',')
        original_label = original_data_df['label'].tolist()
    else:
        raise NotImplementedError
    # test_ = []
    # with open(os.path.join(args.data_dir, 'gen_test.json'), 'r', encoding='utf-8') as loadFile:
    #     for line in loadFile:
    #         test_.append(json.loads(line))
    # original_label = [d['gt'] for d in test_]
    # ---------------------------------------------------------------------------------------------

    new_dict = {
        "context": [],
        "original_variable": [],
        "original_label": [],
        "original_expl": [],
        "reverse_variable": [],
        "reverse_label": [],
        "reverse_expl": [],
        'tags': []
    }

    for idx, i in enumerate(extracted_idx):
        pairid_ = pair_id[i]
        idx_ = original_pair_id.index(pairid_)

        if inconsist_expl[i] == original_expl[idx_]:
            continue
        else:
            new_dict['context'].append(context_part[i])
            new_dict['reverse_variable'].append(variable_part[i])
            new_dict['reverse_label'].append(labels[i])
            new_dict['reverse_expl'].append(inconsist_expl[i])
            new_dict['tags'].append(tags[idx])

            new_dict['original_variable'].append(original_hypo[idx_])
            new_dict['original_label'].append(original_label[idx_])
            new_dict['original_expl'].append(original_expl[idx_])
    return new_dict


# Both explanations contain the same negation expression
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


# Extract human-filtered pattern index
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

    word_pair_dict = pattern_dict['word-pair']

    for i, (o_e, r_e, t) in tqdm(enumerate(zip(original_expl, reverse_expl, tags)), total=len(tags)):
        if t == 'negation':
            continue

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


def main(args):
    # 1. Extract inconsistent explanations using exact match (with edit distance threshold)
    output = match_candidates(args)

    # 2. Filtering logic (remove if both explanations contain negative expressions, human filter)
    output_df = pd.DataFrame(output)

    drop_idx = extract_both_neg_idx(output_df)
    output_df = output_df.drop(drop_idx).reset_index(drop=True)
    output_df = output_df.drop_duplicates()

    with open('./remove_pattern.yml', 'r') as loadFile:
        rm_pattern = yaml.load(loadFile, Loader=yaml.SafeLoader)

    drop_idx = extract_pattern_idx(output_df, rm_pattern)
    output_df = output_df.drop(drop_idx).reset_index(drop=True)

    save_filename = f"final_output.tsv"
    output_df.to_csv(os.path.join(args.save_dir, save_filename), sep='\t', index=False, encoding='utf-8')
    print(f"Total {len(output_df)} inconsistent explanations are extracted")

    tmp_outp = output_df[["context", "original_variable", "original_label"]]
    tmp_outp = tmp_outp.drop_duplicates()
    print(f"# of unique successful attack data: {len(tmp_outp)}")


if __name__ == '__main__':
    args = return_args()
    main(args)



from argparse import ArgumentParser
from functools import partial

from transformers import BertTokenizer


def read_infile(infile):
    answer = dict()
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            symbol, index = line.split()
            if symbol[0] != "<" or symbol[-1] != ">":
                answer[symbol] = len(answer)
    return answer

def find_splits(token, index, initial_prefix="", other_prefix="##", allow_single_initial=False):
    answer = []
    if initial_prefix != "" and token.startswith(initial_prefix):
        prefix, token = initial_prefix, token[len(initial_prefix):]
    elif other_prefix != "" and token.startswith(other_prefix):
        prefix, token = other_prefix, token[len(other_prefix):]
    else:
        prefix, token = "", token
    for end in range(int(not allow_single_initial), len(token)):
        first, second = prefix+token[:end], other_prefix+token[end:]
        first_code, second_code = subtoken_codes.get(first), subtoken_codes.get(second)
        if first_code is None or second_code is None:
            continue
        answer.append([(first, first_code), (second, second_code)])
    return answer
    
def is_bad_split(first, second, index, prefix_to_strip="##"):
    first, first_code = first
    second, second_code = second
    if len(first.strip(prefix_to_strip)) > 1 and first_code > index:
        return True
    if len(second.strip(prefix_to_strip)) > 1 and second_code > index:
        return True
    return False

def find_split_score(key, prefix_to_strip="##"):
    first, second = key
    scores = []
    if len(first[0].strip(prefix_to_strip)) > 1:
        scores.append(first[1])
    if len(second[0].strip(prefix_to_strip)) > 1:
        scores.append(second[1])
    return min(scores) if len(scores) > 0 else 0

def normalize_bert_bpe(s, initial_prefix="", other_prefix="##", output_initial_prefix="^"):
    if initial_prefix and s.startswith(initial_prefix):
        return output_initial_prefix + s[len(initial_prefix):]
    elif other_prefix and s.startswith(other_prefix):
        return s[len(other_prefix):]
    else:
        return s

argument_parser = ArgumentParser()
argument_parser.add_argument("-i", "--input_file", required=True)
argument_parser.add_argument("-o", "--outfile", required=True)

if __name__ == "__main__":
    args = argument_parser.parse_args()
    subtoken_codes = read_infile(args.input_file)
    subtokens = list(subtoken_codes.keys())
    merges = []
    split_func = partial(find_splits, initial_prefix='▁', other_prefix='', allow_single_initial=True)
    split_score_func = partial(find_split_score, prefix_to_strip="▁")
    normalize_func = partial(normalize_bert_bpe, initial_prefix="▁", other_prefix="")
    bad_split_func = partial(is_bad_split, prefix_to_strip="▁")
    for index, subtoken in enumerate(subtokens, 0):
        if len(subtoken.strip("▁")) >= 2:
            # print(f"{subtoken_codes[subtoken]}:{subtoken}", end="")
            splits = split_func(subtoken, index)
            if len(splits) == 0:
                print(subtoken)
            good_splits, bad_splits = [], []
            for elem in splits:
                dest = bad_splits if bad_split_func(*elem, index) else good_splits
                dest.append(elem)
            if len(good_splits) == 0:
                good_splits = bad_splits
                if len(bad_splits) == 0:
                    continue
            (first, _), (second, _) = min(good_splits, key=split_score_func)
            merges.append((normalize_func(first), normalize_func(second), index))
    with open(args.outfile, "w", encoding="utf8") as fout:
        for elem in merges:
            print(*elem, sep="\t", file=fout)
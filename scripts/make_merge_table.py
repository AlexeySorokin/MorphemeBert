from argparse import ArgumentParser
from transformers import BertTokenizer

def find_splits(token, index):
    answer = []
    if token[:2] == "##":
        prefix, token = "##", token[2:]
    else:
        prefix, token = "", token
    for end in range(1, len(token)):
        first, second = prefix+token[:end], "##"+token[end:]
        first_code, second_code = subtoken_codes.get(first), subtoken_codes.get(second)
        if first_code is None or second_code is None:
            continue
        answer.append([(first, first_code), (second, second_code)])
    return answer
    
def is_bad_split(first, second, index):
    first, first_code = first
    second, second_code = second
    if len(first.strip("#")) > 1 and first_code > index:
        return True
    if len(second.strip("#")) > 1 and second_code > index:
        return True
    return False

def find_split_score(key):
    first, second = key
    scores = []
    if len(first[0].strip("#")) > 1:
        scores.append(first[1])
    if len(second[0].strip("#")) > 1:
        scores.append(second[1])
    return min(scores) if len(scores) > 0 else 0

def normalize_bert_bpe(s):
    return s[2:] if s[:2] == "##" else "^" + s

argument_parser = ArgumentParser()
argument_parser.add_argument("-m", "--model", required=True)
argument_parser.add_argument("-o", "--outfile", required=True)

if __name__ == "__main__":
    args = argument_parser.parse_args()
    subtoken_codes = BertTokenizer.from_pretrained(args.model).vocab
    subtokens = list(subtoken_codes.keys())
    merges = []
    for index, subtoken in enumerate(subtokens[105:], 105):
        if len(subtoken.strip("#")) >= 2:
            # print(f"{subtoken_codes[subtoken]}:{subtoken}", end="")
            splits = find_splits(subtoken, index)
            if len(splits) == 0:
                print(subtoken)
            good_splits, bad_splits = [], []
            for elem in splits:
                dest = bad_splits if is_bad_split(*elem, index) else good_splits
                dest.append(elem)
            if len(good_splits) == 0:
                good_splits = bad_splits
                if len(bad_splits) == 0:
                    continue
            (first, _), (second, _) = min(good_splits, key=find_split_score)
            merges.append((normalize_bert_bpe(first), normalize_bert_bpe(second), index))
    with open(args.outfile, "w", encoding="utf8") as fout:
        for elem in merges:
            print(*elem, sep="\t", file=fout)
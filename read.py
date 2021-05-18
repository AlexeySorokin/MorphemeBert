from collections import defaultdict, Counter
import tqdm
import random

from sklearn.model_selection import train_test_split

def to_BMES(morphemes):
    answer = []
    for morpheme in morphemes:
        if len(morpheme) > 1 and ":" in morpheme:
            morpheme, morph_type = morpheme.split(":")
        else:
            morph_type = None
        if len(morpheme) == 1:
            curr_answer = ["S"]
        else:
            curr_answer = ["B"] + ["M"] * (len(morpheme) - 2) + ["E"]
        if morph_type is not None:
            curr_answer = [label + "-" + morph_type for label in answer]
        answer += curr_answer
    return answer


def to_BIO(morphemes, morph_type=None):
    answer = []
    for morpheme in morphemes:
        if len(morpheme) > 1 and ":" in morpheme:
            morpheme, morph_type = morpheme.split(":")
        else:
            morph_type = None
        curr_answer = ["B"] + ["I"] * (len(morpheme) - 1)
        if morph_type is not None:
            curr_answer = [label + "-" + morph_type for label in answer]
        answer += curr_answer
    return answer


def read_infile(infile, sep="\t", morph_sep="/", variant_sep=",", language=None):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                continue
            word, analysis = line.split(sep)
            analysis = analysis.strip().split(variant_sep)[0]
            if language == "finnish":
                word = word.replace("å", "a").replace("ö", "o").replace("ä", "a")
                analysis = analysis.replace("å", "a").replace("ö", "o").replace("ä", "a")
            morphemes = [x for x in analysis.split(morph_sep) if x != ""]
            line_data = {"word": word, "morphemes": morphemes, "bmes_labels": to_BMES(morphemes),
                         "bio_labels": to_BIO(morphemes)}
            answer.append(line_data)
    return answer


def read_unimorph_infile(infile, sep="\t", n=None, min_label_count=100, language=None):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in tqdm.tqdm(fin):
            line = line.strip()
            if line == "":
                continue
            splitted = line.split(sep)
            if len(splitted) != 3 or any((not x.isalpha() and x != "-") for x in splitted[1]):
                continue
            word = splitted[1]
            if language == "finnish":
                word = word.replace("å", "a").replace("ö", "o").replace("ä", "a")
            answer.append({"word": word, "label": splitted[2]})
    label_counts = Counter([elem["label"] for elem in answer])
    answer = [elem for elem in answer if label_counts[elem["label"]] >= min_label_count]
    if n is not None and n < len(answer):
        _, answer = train_test_split(answer, test_size=n, stratify=[elem["label"] for elem in answer])
    return answer


def read_unimorph_tables(infile, sep="\t", delay=1, n=None):
    answer = defaultdict(list)
    last_lines = dict()
    with open(infile, "r", encoding="utf8") as fin:
        for i, line in enumerate(tqdm.tqdm(fin)):
            line = line.strip()
            if line == "":
                continue
            splitted = line.split(sep)
            if len(splitted) != 3 or any((not x.isalpha() and x != "-") for x in splitted[1]):
                continue
            key = splitted[0] + "_" + splitted[2][0]
            if key not in last_lines and len(answer) == n:
                break
            if key not in last_lines or last_lines[key] >= i - delay:
                last_lines[key] = i
                answer[key].append({"word": splitted[1], "label": splitted[2]})
    answer = [{"lemma": key[:-2], "table": table} for key, table in answer.items()]
    return answer

def read_wordlist(infile, n=None, min_length=5, language=None):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            if " " in line:
                line = line.split()[-1]
            line = line.strip()
            if len(line) >= min_length and all(x.isalpha() or x == "-" for x in line):
                if language == "finnish":
                    line = line.replace("å", "a").replace("ö", "o").replace("ä", "a")
                answer.append(line)
    random.shuffle(answer)
    if n is not None:
        answer = answer[:n]
    answer = [{"word": word} for word in answer]
    return answer


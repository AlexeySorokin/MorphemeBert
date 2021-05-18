from utils import decode_from_labels

def write_output(words, corr_labels, pred_labels, outfile):
    with open(outfile, "w", encoding="utf8") as fout:
        for word, first, second in zip(words, corr_labels, pred_labels):
            print(word, decode_from_labels(word, first), decode_from_labels(word, second), file=fout, end="")
            if first != second:
                print("\tERROR", file=fout, end="")
            print("", file=fout)
from argparse import ArgumentParser

from read import read_infile
from training import measure_quality
from write import write_output

argument_parser = ArgumentParser()
argument_parser.add_argument("-g", "--gold", required=True)
argument_parser.add_argument("-p", "--pred", required=True)
argument_parser.add_argument("-s", "--sep", default="/")
argument_parser.add_argument("-S", "--pred_sep", default=None)
argument_parser.add_argument("-l", "--language", default=None)
argument_parser.add_argument("-o", "--output_file", default=None)

if __name__ == "__main__":
    args = argument_parser.parse_args()
    if args.pred_sep is None:
        args.pred_sep = args.sep
    gold_data = read_infile(args.gold, morph_sep=args.sep, language=args.language)
    pred_data = read_infile(args.pred, morph_sep=args.pred_sep, language=args.language)
    for i, (first, second) in enumerate(zip(gold_data, pred_data)):
        if first["word"] != second["word"]:
            raise ValueError(f"Incorrect input f{second} for instance f{i}, f{first} expected.")
    gold_labels = [elem["bmes_labels"] for elem in gold_data]
    pred_labels = [elem["bmes_labels"] for elem in pred_data]
    print(measure_quality(gold_labels, pred_labels, measure_last=False))
    if args.output_file is not None:
        words = [word_data["word"] for word_data in gold_data]
        write_output(words, gold_labels, pred_labels, args.output_file)
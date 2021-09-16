from argparse import ArgumentParser

import numpy as np
from gensim.models import Word2Vec
import torch
from transformers import BertTokenizer, BertModel

from read import read_infile, read_wordlist
from BPEDropout.bpe import load_merge_table, tokenize_word


class BPEDropoutCorpus:
    
    def __init__(self, wordlist, merges, min_length=5, n_words=None,
                 dropout=0.2, noise=0.5, sentinels=None,
                 bpe_symbol="##", random_state=187,
                 repeat=1, language=None
                 ):
        self.wordlist = [x["word"] for x in read_wordlist(wordlist, min_length=min_length, n=n_words, language=language)]
        self.merges = load_merge_table(merges, normalize=True)
        self.random_generator = np.random.RandomState(seed=random_state)
        if sentinels is None:
            sentinels = ["^", ""]
        self.tokenization_params = {
            "dropout": dropout, "noise": noise, "sentinels": sentinels, "bpe_symbol": bpe_symbol,
            "random_generator": self.random_generator
        }
        self.repeat = repeat
    
    def __iter__(self):
        self.random_generator.shuffle(self.wordlist)
        self.idx = 0
        self.iteration = 0
        return self
    
    def __next__(self):
        if self.idx == len(self.wordlist):
            self.iteration += 1
            if self.iteration >= self.repeat:
                raise StopIteration
            self.idx = 0
        subtokens = tokenize_word(self.merges, self.wordlist[self.idx], **self.tokenization_params)
        self.idx += 1
        return subtokens

argument_parser = ArgumentParser()
argument_parser.add_argument("-t", "--train", required=True)
argument_parser.add_argument("-m", "--merges", required=True)
argument_parser.add_argument("-i", "--initialization", default=None)
argument_parser.add_argument("-T", "--tmp_file", default="data/tmp.bert")
argument_parser.add_argument("-n", "--n-words", default=None, type=int)
argument_parser.add_argument("-r", "--repeat", default=1, type=int)
argument_parser.add_argument("-d", "--dropout", default=0.2, type=float)
argument_parser.add_argument("-v", "--vector-size", default=300, type=int)
argument_parser.add_argument("-o", "--output-file", required=True)
argument_parser.add_argument("-l", "--language", default=None)

if __name__ == "__main__":
    args = argument_parser.parse_args()
    print("Reading...")
    repeat = args.repeat if args.initialization is None else 1
    epochs = 1 if args.initialization is None else args.repeat
    corpus = BPEDropoutCorpus(args.train, args.merges, dropout=args.dropout,
                              n_words=args.n_words, repeat=args.repeat, language=args.language)
    if args.initialization is None:
        model = Word2Vec(sentences=corpus, min_count=1, vector_size=args.vector_size, workers=1, window=2)
        corpus_count = model.corpus_count
    else:
        print("Reading BERT model...")
        vocab = BertTokenizer.from_pretrained(args.initialization).vocab
        with torch.no_grad():
            embeddings = BertModel.from_pretrained(args.initialization).embeddings.word_embeddings.weight.cpu().numpy()
        print("Saving BERT embeddings...")
        with open(args.tmp_file, "w", encoding="utf8") as fout:
            print(*(embeddings.shape), file=fout)
            for key, vector in zip(list(vocab), embeddings):
                print("{} {}".format(key, " ".join(map(str, vector))), file=fout)
        print("Loading BERT embeddings...")
        model = Word2Vec(min_count=1, vector_size=768, workers=4, window=2)
        model.build_vocab_from_freq({key: 1 for key in vocab})
        model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)
        model.wv.intersect_word2vec_format(args.tmp_file, binary=False)
        corpus_count = len(corpus.wordlist)
    print(len(model.wv), corpus_count)
    print("Training word2vec...")
    model.train(corpus, total_examples=corpus_count, epochs=epochs)
    print("Saving...")
    model.wv.save_word2vec_format(args.output_file, binary=False)
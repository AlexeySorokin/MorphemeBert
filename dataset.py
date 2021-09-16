from collections import Counter

import numpy as np
from torch import LongTensor
from torch.utils.data.dataset import Dataset

from read import read_wordlist
from BPEDropout.bpe import tokenize_word, load_merge_table


class BertSegmentDatasetReader(Dataset):
    
    def __init__(self, data, d, vocab=None, embeddings=None,
                 letters=None, labels=None,
                 min_symbol_count=1, max_morpheme_length=None, field="bio_labels"):
        self.data = data
        self.vocab = vocab
        self.embeddings = embeddings
        self.d = d
        if max_morpheme_length is None or max_morpheme_length <= d:
            self.max_morpheme_length = d
        else:
            self.max_morpheme_length = max_morpheme_length
        self.field = field
        # symbols
        if letters is None:
            letters = Counter(x for elem in data for x in elem["word"])
            self.letters_ = ["PAD", "UNK"] + [x for x, count in letters.items() if count >= min_symbol_count]
        else:
            self.letters_ = letters
        self.letter_codes_ = {label: i for i, label in enumerate(self.letters_)}
        if labels is None:
            labels = ["PAD"] + list({x for elem in data for x in elem[self.field]})
        self.labels_ = labels
        self.label_codes_ = {label: i for i, label in enumerate(self.labels_)}
    
    def _word_to_matrix(self, word):
        d = self.d + int(self.max_morpheme_length > self.d)
        subtoken_indexes = np.zeros(shape=(len(word), 2 * d), dtype=int)
        for start in range(len(word)):
            long_morpheme_index, long_morpheme_length = None, None
            for length in range(1, self.max_morpheme_length + 1):
                if start + length > len(word):
                    break
                ngram = word[start:start + length]
                if start > 0:
                    ngram = "##" + ngram
                index = self.vocab.get(ngram)
                if index is not None:
                    if length <= d:
                        subtoken_indexes[start, length - 1] = index
                        if start + length < len(word):
                            subtoken_indexes[start + length, d + length - 1] = index
                    else:
                        long_morpheme_index, long_morpheme_length = index, length
            if long_morpheme_index is not None:
                subtoken_indexes[start, d - 1] = long_morpheme_index
                if start + long_morpheme_length < len(word) and subtoken_indexes[
                    start + long_morpheme_length, 2 * d - 1] == 0:
                    subtoken_indexes[start + long_morpheme_length, 2 * d - 1] = long_morpheme_index
        answer = self.embeddings[subtoken_indexes]
        answer[subtoken_indexes == 0] = 0.0
        return subtoken_indexes, answer
    
    def __getitem__(self, i):
        word = self.data[i]["word"]
        answer = {"letters": [self.letter_codes_.get(letter, 1) for letter in word]}
        if self.vocab is not None and self.embeddings is not None:
            subtoken_indexes, x = self._word_to_matrix(word)
            answer["inputs"] = x
            answer["subtoken_indexes"] = subtoken_indexes
        try:
            answer["y"] = [self.label_codes_[label] for label in self.data[i][self.field]]
        except:
            pass
        mask = self.data[i].get("mask")
        mask = np.array(mask, dtype="float") if mask is not None else np.ones(shape=(len(word),), dtype="float")
        answer["mask"] = mask * self.data[i].get("weight", 1.0)
        return answer
    
    def __len__(self):
        return len(self.data)
    
    @property
    def output_dim(self):
        return 2 * (self.d + int(self.max_morpheme_length > self.d))

class LetterDatasetReader(Dataset):

    def __init__(self, data, letters=None, min_symbol_count=1, field="bio_labels"):
        self.data = data
        self.field = field
        # symbols
        if letters is None:
            letters = Counter(x for elem in data for x in elem["word"])
            self.letters_ = ["PAD", "UNK"] + [x for x, count in letters.items() if count >= min_symbol_count]
        else:
            self.letters_ = letters
        self.letter_codes_ = {label: i for i, label in enumerate(self.letters_)}

    def __getitem__(self, i):
        word = self.data[i]["word"]
        letters = [self.letter_codes_.get(letter, 1) for letter in word]
        next_letters = letters[1:] + [0]
        prev_letters = [0] + letters[:-1]
        return {
            "letters": letters, "next_letter": next_letters, "prev_letter": prev_letters
        }

    def __len__(self):
        return len(self.data)

class BertUnimorphDatasetReader(BertSegmentDatasetReader):
    
    def __init__(self, data, d,
                 vocab=None, embeddings=None,
                 letters=None, labels=None,
                 min_symbol_count=1, max_morpheme_length=None,
                 field="label"):
        self.data = data
        self.vocab = vocab
        self.embeddings = embeddings
        self.d = d
        if max_morpheme_length is None or max_morpheme_length <= d:
            self.max_morpheme_length = d
        else:
            self.max_morpheme_length = max_morpheme_length
        self.field = field
        # symbols
        if letters is None:
            letters = Counter(x for elem in data for x in elem["word"])
            self.letters_ = ["PAD", "UNK"] + [x for x, count in letters.items() if count >= min_symbol_count]
        else:
            self.letters_ = letters
        self.letter_codes_ = {label: i for i, label in enumerate(self.letters_)}
        if labels is None:
            labels = list({elem[self.field] for elem in data})
        self.labels_ = labels
        self.label_codes_ = {label: i for i, label in enumerate(self.labels_)}
    
    def __getitem__(self, i):
        word = self.data[i]["word"]
        answer = {"letters": [self.letter_codes_.get(letter, 1) for letter in word]}
        if self.vocab is not None and self.embeddings is not None:
            subtoken_indexes, x = self._word_to_matrix(word)
            answer["inputs"] = x
        if self.field in self.data[i]:
            answer["y"] = self.label_codes_[self.data[i][self.field]]
        return answer

class FasttextSegmentDatasetReader(Dataset):
    
    def __init__(self, data, embedder, d, letters=None,
                 min_symbol_count=1, field="bio_labels"):
        self.data = data
        self.embedder = embedder
        self.d = d
        self.field = field
        # symbols
        if letters is None:
            letters = Counter(x for elem in data for x in elem["word"])
            self.letters_ = ["PAD", "UNK"] + [x for x, count in letters.items() if count >= min_symbol_count]
        else:
            self.letters_ = letters
        self.letter_codes_ = {label: i for i, label in enumerate(self.letters_)}
        labels = {x for elem in data for x in elem[self.field]}
        self.labels_ = ["PAD"] + list(labels)
        self.label_codes_ = {label: i for i, label in enumerate(self.labels_)}
    
    def _word_to_matrix(self, word):
        answer = np.zeros(shape=(len(word), 2 * self.d, self.embedder.get_dimension()), dtype=float)
        for start in range(len(word)):
            long_morpheme_index, long_morpheme_length = None, None
            for length in range(1, self.d + 1):
                end = start + length
                if end > len(word):
                    break
                ngram = word[start:end]
                if start == 0:
                    ngram = "<" + ngram
                if end == len(word):
                    ngram = ngram + ">"
                if len(ngram) > self.d:
                    continue
                index = self.embedder.get_subword_id(ngram)
                vector = self.embedder.get_input_vector(index)
                answer[start, length - 1] = vector
                if end < len(word):
                    answer[end, self.d + length - 1] = vector
        return answer
    
    def __getitem__(self, i):
        word = self.data[i]["word"]
        x = self._word_to_matrix(word)
        letters = [self.letter_codes_.get(letter, 1) for letter in word]
        y = [self.label_codes_[label] for label in self.data[i][self.field]]
        return {"inputs": x, "letters": letters, "y": y}
    
    def __len__(self):
        return len(self.data)
    
    @property
    def output_dim(self):
        return 2 * self.d


class PartialDatasetReader(Dataset):
    
    def __init__(self, dataset: Dataset, indexes):
        self.dataset = dataset
        self.indexes = indexes
    
    def __getitem__(self, index):
        return self.dataset[self.indexes[index]]
    
    def __len__(self):
        return len(self.indexes)


class BPEDropoutDataset:
    
    def __init__(self, wordlist, merges, vocab,
                 min_length=5, n_words=None, dropout=0.2, noise=0.5,
                 sentinels=["^", ""], bpe_symbol="##", random_state=187,
                 language=None, attach_special_tokens=False):
        self.language = language
        self.attach_special_tokens = attach_special_tokens
        self.wordlist = [self._preprocess(x["word"]) for x in read_wordlist(wordlist, min_length=min_length, n=n_words)]
        self.merges = load_merge_table(merges, normalize=True)
        self.vocab = vocab
        self.unk_token, self.mask_token = vocab["[UNK]"], vocab["[MASK]"]
        if self.attach_special_tokens:
            self.begin_token, self.end_token = vocab["[BEGIN]"], vocab["[END]"]
        self.random_generator = np.random.RandomState(seed=random_state)
        self.tokenization_params = {
            "dropout": dropout, "noise": noise, "sentinels": sentinels, "bpe_symbol": bpe_symbol,
            "random_generator": self.random_generator
        }
    
    def _preprocess(self, word):
        if self.language == "finnish":
            word = word.replace("å", "a").replace("ö", "o").replace("ä", "a")
        return word
    
    def __getitem__(self, index):
        word = self.wordlist[index]
        tokens = tokenize_word(self.merges, word, **self.tokenization_params)
        tokens = [self.vocab.get(token, self.unk_token) for token in tokens]
        token_index = int(self.random_generator.uniform() * len(tokens))
        label = tokens[token_index]
        tokens[token_index] = self.mask_token
        if self.attach_special_tokens:
            tokens = [self.begin_token] + tokens + [self.end_token]
            token_index += 1
        return {"tokens": tokens, "position": token_index, "y": label}
    
    def __len__(self):
        return len(self.wordlist)
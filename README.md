# MorphemeBert

A repository containing code for AIST 2021 submission *Improving morpheme segmentation using BERT embeddings*.

## Installation
```shell
# Make a fresh virtual environment
python3 -m venv env
source env/bin/activate
# Install requirements
pip install requirements.txt
```

## Data

See the examples of training data for Turkish, Finnish and English in `data` directory. Other datasets used in the paper are private.
Format your own data accordingly. You may use other morpheme delimiter, such as space, by passing it as `-s` flag during training.

## Training the models

Model configuration files are available in `config` directory. You may want to change `checkpoint` parameter to save checkpoints. 
All the path are relative to root directory of the project.

### Train the basic model
```shell
# Train a basic model on English
python main.py -c config/basic.json -t data/eng.train -d data/eng.dev -s "/"
# Train a basic model on Finnish
python main.py -c config/basic.json -t data/eng.train -d data/eng.dev -s "/" - l finnish
```

### Train the bert-enhanced model
```shell
# Train a multilingual BERT model on Turkish
python main.py -c config/bert_multi.json -t data/tur.train -d data/tur.dev -s "/"
# Train a Turkish BERT model on Turkish
python main.py -c config/tur/bert.json -t data/tur.train -d data/tur.dev -s "/"
# Train a model with w2v-style subword embeddings on Turkish
python main.py -c config/zulu/w2v.json -t data/tur.train -d data/tur.dev  -s "/"
```

### Other experiments from the paper

1. Random embeddings: pass, for example, `config/eng/bert_random.json` as `-c` arguments.
2. Word2Vec subword training (you can download the wordlist from [Morphochallenge page](http://morpho.aalto.fi/events/morphochallenge2010/datasets.shtml)).
   1. Using pretrained BERT vocabulary
   ```shell
   # Preparing merges file for BPE dropout
   python scripts/make_merge_table.py -m ${BERT_MODEL} -o ${MERGES_PATH}
   # Training word2vec
   python scripts/word2vec.py -t ${WORDLIST_PATH} -m ${MERGES_PATH} -i ${BERT_MODEL} -r 10 -d 0.3 -o ${OUTPUT_FILE}.vec
   ```
   2. With vocabulary training
   ```shell
   scripts/prepare_w2v.sh -i ${WORDLIST_PATH} -l english -r 5
   ```
3. Bert finetuning: TBD.
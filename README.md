# MorphemeBert

A repository containing code for EMNLP 2021 submission *Improving neural morpheme segmentation using pretrained models*.

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
```

### Other experiments from the paper

1. Random embeddings: pass, for example, `config/eng/bert_random.json` as `-c` arguments.
2. Word2Vec subword training: TBD.
3. Bert finetuning: TBD.
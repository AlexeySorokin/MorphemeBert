# MorphemeBert

## Installation
```shell
# Make a fresh virtual environment
python3 -m venv env
source env/bin/activate
# Install requirements
pip install requirements.txt
```

## Data

See the examples of training data in `data` directory. 
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
#!/bin/bash
cd $(dirname $0)/..

MODEL_TYPE=bpe
VOCAB_SIZE=8000
ROUNDS=10

while getopts "i:l:m:v:r:" opt
  do
      case $opt in
      i) INFILE=$OPTARG ;;
      l) LANGUAGE=$OPTARG ;;
      m) MODEL_TYPE=$OPTARG ;;
      v) VOCAB_SIZE=$OPTARG ;;
      r) ROUNDS=$OPTARG ;;
      *) echo "Unknown option $opt"
        exit 1 ;;
      esac
  done

MODEL_PREFIX=data/trained_merges/${LANGUAGE}_${MODEL_TYPE}
spm_train --input ${INFILE} --model_prefix ${MODEL_PREFIX} \
--vocab_size=${VOCAB_SIZE} --character_coverage=1.0 --model_type ${MODEL_TYPE}
python make_merge_table_from_spm.py -i ${MODEL_PREFIX}.vocab -o ${MODEL_PREFIX}.merges
command="-t ${INFILE} -m ${MODEL_PREFIX}.merges -r $ROUNDS -d 0.3 -o data/word2vec/${LANGUAGE}_${VOCAB_SIZE}.vec"
if [ $LANGUAGE = "finnish" ]; then
  command+=" -l finnish"
fi
python word2vec.py $command
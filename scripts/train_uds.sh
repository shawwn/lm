#!/usr/bin/env bash 
set -e

# clean utf8
CLEANTXT_INPUT=data/uds
CLEANTXT_OUTPUT=/tmp/cleantxt
lm cleantxt ${CLEANTXT_INPUT} ${CLEANTXT_OUTPUT} --force

# train encoder
TOKENIZER_INPUT=${CLEANTXT_OUTPUT}
TOKENIZER_OUTPUT=/tmp/tokenizer/
lm_train_tokenizer --vocab_size 1010 --input ${TOKENIZER_INPUT}/ --output ${TOKENIZER_OUTPUT}

# converts to tfrecord 
ENCODE_INPUT=${CLEANTXT_OUTPUT}
ENCODE_OUTPUT=/tmp/tfrecord
lm encode --encoder ${TOKENIZER_OUTPUT} ${ENCODE_INPUT}/\*.\* ${ENCODE_OUTPUT}

# check output
lm_check_dataset ${ENCODE_OUTPUT}/\*.tfrecord --encoder ${TOKENIZER_OUTPUT}
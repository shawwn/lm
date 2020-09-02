#!/usr/bin/env bash 

# clean utf8
CLEANTXT_INPUT=data/uds
CLEANTXT_OUTPUT=/tmp/cleantxt
lm cleantxt ${CLEANTXT_INPUT} ${CLEANTXT_OUTPUT} --force

# converts to tfrecord 
ENCODE_INPUT=${CLEANTXT_OUTPUT}
ENCODE_OUTPUT=/tmp/tfrecord
lm encode --encoder gpt2 /tmp/cleantxt/\*.\* ${ENCODE_OUTPUT}

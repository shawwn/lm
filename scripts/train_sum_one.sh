#!/usr/bin/env bash 
set -e

VOCAB_SIZE=16
MAX_SEQ_LEN=8
TASK_NAME=sum_one

TASK_SPEC=etc/lm/tasks/sum_one.jsonnet

# create synth data
SYNTH_OUTPUT=/tmp/${TASK_NAME}/train
if [ ! -f ${SYNTH_OUTPUT}/dataset.info.json ];
then
    lm synth ${TASK_SPEC} ${SYNTH_OUTPUT} \
        --vocab_size ${VOCAB_SIZE} \
        --max_seq_len ${MAX_SEQ_LEN} \
        --n_samples 10000 \
        --seed 1337 ;
fi

# train task
SAVED_TRAIN_SPEC=var/lm/runs/run_$(date +%s).json
TRAIN_SPEC=etc/experiments/training/add-one_cpu.jsonnet
# task specifies task, example format, infeed
lm train ${TRAIN_SPEC} \
        --task ${TASK_SPEC} \
        --dataset ${SYNTH_OUTPUT} \
        --save-settings ${SAVED_TRAIN_SPEC}

# evaluate by synth new data
SYNTH_TEST_OUTPUT=/tmp/${TASK_NAME}/test
lm --seed 42 \ 
    synth ${TASK_SPEC} ${SYNTH_TEST_OUTPUT} \
    --vocab_size ${VOCAB_SIZE} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --n_samples 1000

lm eval \
    ${SAVED_TRAIN_SPEC} \
    --dataset ${SYNTH_TEST_OUTPUT}\
    --output results.json

# check output
# lm_check_dataset ${ENCODE_OUTPUT}/\*.tfrecord --encoder ${TOKENIZER_OUTPUT}

# interactive
evaluate task
    eval \
    ${RUN_CONFIG} \
    --dataset ${SYNTH_TEST_OUTPUT}\
    --output results.json

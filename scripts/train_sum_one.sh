#!/usr/bin/env bash 
set -e

VOCAB_SIZE=16
MAX_SEQ_LEN=8
TASK_NAME=sum_one

SYNTH_OUTPUT=/tmp/${TASK_NAME}/train

lm synth etc/lm/tasks/sum_one.jsonnet ${SYNTH_OUTPUT} \
    --vocab_size ${VOCAB_SIZE} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --n_samples 10000

# # train encoder
# TOKENIZER_INPUT=${CLEANTXT_OUTPUT}
# TOKENIZER_OUTPUT=/tmp/tokenizer/
# # lm_train_tokenizer --vocab_size 1010 --input ${TOKENIZER_INPUT}/ --output ${TOKENIZER_OUTPUT}
# lm train \
#     /content/GPTNeo/configs/training/seq2seq_add_one_tpu-v3-8.jsonnet \
#     --dataset gs://{GCS_BUCKET}/{GCS_LOCATION} \
#     --tpu grpc://{os.environ['COLAB_TPU_ADDR']} 
#     # --model_path gs://{GCS_BUCKET}/train/neogpt/ \

# # converts to tfrecord 
# ENCODE_INPUT=${CLEANTXT_OUTPUT}
# ENCODE_OUTPUT=/tmp/tfrecord
# lm synth configs/tasks/add_one.jsonnet /tmp/output/eval --vocab_size 16 --ctx_len 8 --n_samples {10_000}

# !python -m main \
#     eval \
#     /content/GPTNeo/run-20203301_213355.json \
#     --dataset /tmp/output/eval \
#     --output results.json

# # check output
# lm_check_dataset ${ENCODE_OUTPUT}/\*.tfrecord --encoder ${TOKENIZER_OUTPUT}
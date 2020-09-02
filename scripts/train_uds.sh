#!/usr/bin/env bash 

# clean utf8
lm cleantxt data/uds/\*.txt /tmp/cleantxt/ --force

# converts to tfrecord 
lm preprocess data/uds/\*.txt /tmp/tfrecord/ --force

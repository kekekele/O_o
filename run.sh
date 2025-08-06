#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

export TRAIN_LOG_PATH=./logs
export TRAIN_TF_EVENTS_PATH=./logs/tf_events
export TRAIN_DATA_PATH=./data/TencentGR_1k
export TRAIN_CKPT_PATH=./result/


# write your code below
cd /root/autodl-tmp/
python -u main.py

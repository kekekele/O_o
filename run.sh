#!/bin/bash
pip install orjson
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
python -u main.py

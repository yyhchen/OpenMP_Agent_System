#!/bin/bash

# BAAI/bge-large-zh-v1.5, BAAI/bge-m3

python -m fastchat.serve.controller --port 21003 &

python -m fastchat.serve.model_worker --model-path /home/yhchen/huggingface_model/BAAI/bge-large-zh-v1.5 --model-names bge-m3 --num-gpus 1 --controller-address http://localhost:21003 &

python -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8200 --controller-address http://localhost:21003
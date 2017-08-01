#!/usr/bin/env bash
python -u fcn.py --model=fcn32s --model-dir="./modelm" --cutoff=800 --gpus=5,6
#python -u fcn.py --model=fcn16s --model-dir="./model" --cutoff=800 --gpu=7
#python -u fcn.py --model=fcn8s --model-dir="./model" --cutoff=800 --gpu=7

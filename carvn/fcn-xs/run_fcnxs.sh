#!/usr/bin/env bash
python -u fcn_xs.py --model=fcn32s --model-dir="./model2" --cutoff=800
python -u fcn_xs.py --model=fcn16s --model-dir="./model2" --cutoff=800
python -u fcn_xs.py --model=fcn8s --model-dir="./model2" --cutoff=800

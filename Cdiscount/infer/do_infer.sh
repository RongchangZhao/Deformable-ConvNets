#!/usr/bin/env bash

#nohup python -u infer.py --model './model/_job-6435_model_scene-resnet-152,52,224' --output-dir '/data/cdiscount/test152' --gpu 0 > 152.log 2>&1 &
#nohup python -u infer.py --model './model/_job-6608_model_discount-resnet-50,48,224' --output-dir '/data/cdiscount/test50' --gpu 1 > 50.log 2>&1 &
#nohup python -u infer.py --model './model/_job-7437_output_discount-dpn92-5k-180,32,160' --output-dir '/data/cdiscount/testdpn92' --gpu 0 > 92.log 2>&1 &
#nohup python -u infer.py --model './model/discount-dpn107-180,25,160' --output-dir '/data/cdiscount/testdpn107' --gpu 0 > 107.log 2>&1 &
#nohup python -u infer.py --model './model/discount-dpn92-5k-180,39,160' --output-dir '/data/cdiscount/testdpn92' --gpu 1 > 92.log 2>&1 &
#nohup python -u infer.py --model './model/discount-inception-resnet-v2,67,160' --expand 8 --output-dir '/data/cdiscount/testirv2' --gpu 0 > irv2.log 2>&1 &
#nohup python -u infer.py --model './model/discount-inception-resnet-v2,67,160' --expand 16 --output-dir '/data/cdiscount/testirv2_16' --gpu 1 > irv2_16.log 2>&1 &
nohup python -u infer.py --model './model/discount-inception-resnet-v2,67,160' --expand 32 --output-dir '/data/cdiscount/testirv2_32' --gpu 2 > irv2_32.log 2>&1 &

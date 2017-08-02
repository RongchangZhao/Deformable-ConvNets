#!/usr/bin/env bash

python submit.py --gpu 0 --parts 1,2 > 1_2.log 2>&1 &
python submit.py --gpu 1 --parts 3,4 > 3_4.log 2>&1 &
python submit.py --gpu 2 --parts 5,6 > 5_6.log 2>&1 &
python submit.py --gpu 3 --parts 7,8 > 7_8.log 2>&1 &
python submit.py --gpu 4 --parts 9,10 > 9_10.log 2>&1 &
python submit.py --gpu 5 --parts 11,12 > 11_12.log 2>&1 &
python submit.py --gpu 6 --parts 13,14 > 13_14.log 2>&1 &
python submit.py --gpu 7 --parts 15,16 > 15_16.log 2>&1 &

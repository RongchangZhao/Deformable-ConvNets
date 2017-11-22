import os
import sys
import numpy as np
import math
import datetime
import json
import pickle
import pandas as pd

output_file = "./irv2_e32.csv"
path_labelmap='./synset.txt'
labelmap = {}
with open(path_labelmap, 'r') as f:
  for line in f:
    vec = line.strip().split()
    labelmap[int(vec[1])] = int(vec[0])
filenames = [
#'/data/cdiscount/test152/pk.11',
#'/data/cdiscount/test50/pk.11',
#'/data/cdiscount/testdpn92/pk.11',
'/data/cdiscount/testirv2_32/pk.11',
]

X = None
XS = []
Y = []

for line in open('raw_ids', 'r'):
  y = int(line.strip())
  Y.append(y)

print(len(Y))

pk_in_list = []
for filename in filenames:
    pk_in = open(filename, 'rb')
    pk_in_list.append(pk_in)

def output(f, _id, score):
    #v = np.array(v)
    v = score
    #av = np.mean(v, axis=0)
    #mv = np.amax(v, axis=0)
    #v = av*0.5+mv*0.5
    #v = av
    _label = np.argmax(v)
    labelid = labelmap[_label]
    f.write("%d,%d\n"%(_id, labelid))

with open(output_file, 'w') as f:
    f.write("_id,category_id\n")
    last = [-1, None]
    for i in xrange(len(Y)):
        score = None
        _id = Y[i]
        for pk_in in pk_in_list:
            score0 = pickle.load(pk_in)
            if score is None:
                score = score0
            else:
                score += score0
        if _id==last[0]:
            last[1]+=score
        else:
            if last[0]>=0:
                output(f, last[0], last[1])
            last[0] = _id
            last[1] = score

    if last[0]>=0:
        output(f, last[0], last[1])

  
for pk_in in pk_in_list:
    pk_in.close()


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os
import time
import copy
import sys
import numpy as np
import importlib
import itertools
import argparse
import math
import json
import random
#from guuker import prt

SEED = 727

DATA_ROOT = '/data1/deepinsight/carvn'

id2images = {}

train_image_dir = os.path.join(DATA_ROOT, 'train_hq')
for image in os.listdir(train_image_dir):
    image = image.strip()
    if not image.endswith('.jpg'):
        continue
    basename = image.split('.')[0]
    image_id, pos = basename.split('_')
    if image_id not in id2images:
        id2images[image_id] = 1
  #id2images[image_id].append(os.path.join(train_image_dir, image))
  #id2images[image_id].append( (image_id, pos) )

id2images = sorted(id2images.items(), key = lambda x: x[0])
random.Random(SEED).shuffle(id2images)
train_f = open('train.lst', 'w')
val_f = open('val.lst', 'w')
val_count = len(id2images)//20
for i in xrange(len(id2images)):
    #f = val_f if i<val_count else train_f
    image_id = id2images[i][0]
    idxs = range(1, 17)
    random.Random(SEED).shuffle(idxs)
    for idx in idxs:
        data_image = os.path.join(DATA_ROOT, 'train_hq', "%s_%02d.jpg" % (image_id, idx))
        mask_image = os.path.join(DATA_ROOT, 'train_masks', "%s_%02d_mask.gif" % (image_id, idx))
        if i < val_count:
            val_f.write("%d\t%s\t%s\n" % (0, data_image, mask_image))
        train_f.write("%d\t%s\t%s\n" % (0, data_image, mask_image))

train_f.close()
val_f.close()



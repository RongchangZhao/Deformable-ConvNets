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

DATA_ROOT = '/raid5data/dplearn/carvn'

id2images = {}

train_image_dir = os.path.join(DATA_ROOT, 'train')
for image in os.listdir(train_image_dir):
  image = image.strip()
  if not image.endswith('.jpg'):
    continue
  basename = image.split('.')[0]
  image_id, pos = basename.split('_')
  if image_id not in id2images:
    id2images[image_id] = []
  id2images[image_id].append(os.path.join(train_image_dir, image))

train_f = open('train.lst', 'w')
val_f = open('val.lst', 'w')
train_num = 0
val_num = 0
for image_id, images in id2images.iteritems():
  images = sorted(images)
  random.Random(SEED).shuffle(images)
  for i, image in enumerate(images):
    mask_image = os.path.join(DATA_ROOT, 'train_masks', "%s_mask.gif" % image.split('/')[-1][0:-4])
    if i<2:
      val_f.write("%d\t%s\t%s\n" % (val_num, image, mask_image))
      val_num+=1
    else:
      train_f.write("%d\t%s\t%s\n" % (train_num, image, mask_image))
      train_num+=1

train_f.close()
val_f.close()


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
from matplotlib import pyplot as plt
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
import cv2 as cv
#from guuker import prt

SEED = 727

DATA_ROOT = '/home/adminpro/DSB2018'

data_list = {}


train_image_dir = os.path.join(DATA_ROOT, 'train')
for img_name in os.listdir(train_image_dir):
    img_name = img_name.strip()
    data_path = os.path.join(train_image_dir, img_name)
    img_path = os.path.join(data_path, "images")
    mask_path = os.path.join(data_path, "masks")
    # Generate masks
    if os.path.isdir(data_path):
        img = cv.imread(os.path.join(img_path, img_name+ '.png'), cv.IMREAD_UNCHANGED)
        # print(img.shape)
        assert((img[:,:,3]==255).all())
        merged_mask = None
        for mask_name in os.listdir(mask_path):
            if mask_name.endswith('.png') and not mask_name.startswith("mask"):
                mask = cv.imread(os.path.join(mask_path,mask_name), cv.IMREAD_UNCHANGED)
                if merged_mask is not None:
                    merged_mask = merged_mask + mask
                    # plt.imshow(mask/255,cmap ='gray')
                    # plt.show()
                else:
                    merged_mask = mask
        assert(merged_mask is not None)
        merged_mask[merged_mask>255] = 255
        cv.imwrite(os.path.join(mask_path,"mask.png"),merged_mask)
        # plt.imshow(merged_mask/255.0,cmap ='gray')
        #
        # plt.show()
        # cv.imshow("Image",merged_mask)
        # print(mask.shape)
        data_list[img_name] = {
            "img_path":os.path.join(img_path, img_name),
            "mask_path":os.path.join(mask_path, "masks.png")
            }
data_list = sorted(data_list.items(), key = lambda x: x[0])
random.Random(SEED).shuffle(data_list)
for i in range(10):
    
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
        if image_id in val_idmap or (len(val_idmap)==0 and i < val_count):
            val_f.write("%d\t%s\t%s\n" % (0, data_image, mask_image))
        else:
            train_f.write("%d\t%s\t%s\n" % (0, data_image, mask_image))

train_f.close()
val_f.close()

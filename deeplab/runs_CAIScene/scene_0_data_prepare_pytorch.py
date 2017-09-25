#coding=utf-8

import os
import codecs
import json

DATA_ROOT='/data1/deepinsight/aichallenger/scene'

os.system('mkdir -p {0}/ai_challenger_scene_trainval/train'.format(DATA_ROOT))
os.system('mkdir -p {0}/ai_challenger_scene_trainval/val'.format(DATA_ROOT))
os.system('mkdir -p {0}/ai_challenger_scene_trainval/trainval'.format(DATA_ROOT))
os.system('mkdir -p {0}/ai_challenger_scene_trainval/trainval/train'.format(DATA_ROOT))
os.system('mkdir -p {0}/ai_challenger_scene_trainval/trainval/val'.format(DATA_ROOT))

scene_cls_roots = ['{0}/ai_challenger_scene_trainval/train'.format(DATA_ROOT),
                   '{0}/ai_challenger_scene_trainval/val'.format(DATA_ROOT),
                   '{0}/ai_challenger_scene_trainval/trainval/train'.format(DATA_ROOT),\
                   '{0}/ai_challenger_scene_trainval/trainval/val'.format(DATA_ROOT)]

for root in scene_cls_roots:
    for i in range(80):
        os.system('mkdir -p {0}/{1:0>2}'.format(root,i))

DATA_TRAIN_ROOT=DATA_ROOT+'/ai_challenger_scene_train_20170904'
DATA_TRAIN_SRC=DATA_TRAIN_ROOT+'/scene_train_images_20170904'
DATA_TRAIN_JSON=DATA_TRAIN_ROOT+'/scene_train_annotations_20170904.json'



DATA_VAL_ROOT=DATA_ROOT+'/ai_challenger_scene_validation_20170908'
DATA_VAL_SRC=DATA_VAL_ROOT+'/scene_validation_images_20170908'
DATA_VAL_JSON=DATA_VAL_ROOT+'/scene_validation_annotations_20170908.json'



with codecs.open(DATA_TRAIN_JSON,encoding='utf-8') as f:
    trainList = json.load(f)
    train_cls_roots = [scene_cls_roots[0],scene_cls_roots[2]]
    for idx,d in enumerate(trainList):
        if idx%100==0:
            print idx
        i = int(d['label_id'])
        for target in train_cls_roots:
            os.system('cp {0}/{1} {2}/{3:0>2}/'.format(DATA_TRAIN_SRC, d['image_id'], target, i))


with codecs.open(DATA_VAL_JSON,encoding='utf-8') as f:
    valList = json.load(f)
    val_cls_roots = [scene_cls_roots[1],scene_cls_roots[2],scene_cls_roots[3]]
    for idx,d in enumerate(valList):
        if idx%100==0:
            print idx
        i = int(d['label_id'])
        for target in val_cls_roots:
            os.system('cp {0}/{1} {2}/{3:0>2}/'.format(DATA_VAL_SRC, d['image_id'], target, i))
            

    
    
    
    
    
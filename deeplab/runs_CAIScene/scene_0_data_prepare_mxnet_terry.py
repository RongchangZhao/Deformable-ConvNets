#coding=utf-8

import os
import codecs
import json
import glob

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
'''
for root in scene_cls_roots:
    for i in range(80):
        os.system('mkdir -p {0}/{1:0>2}'.format(root,i))
'''
DATA_TRAIN_ROOT=DATA_ROOT+'/ai_challenger_scene_train_20170904'
DATA_TRAIN_SRC=DATA_TRAIN_ROOT+'/scene_train_images_20170904'
DATA_TRAIN_JSON=DATA_TRAIN_ROOT+'/scene_train_annotations_20170904.json'



DATA_VAL_ROOT=DATA_ROOT+'/ai_challenger_scene_validation_20170908'
DATA_VAL_SRC=DATA_VAL_ROOT+'/scene_validation_images_20170908'
DATA_VAL_JSON=DATA_VAL_ROOT+'/scene_validation_annotations_20170908.json'


DATA_TEST_A_ROOT=DATA_ROOT+'/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922'


with codecs.open(DATA_TRAIN_JSON,encoding='utf-8') as f:
    trainList = json.load(f)

with codecs.open(DATA_VAL_JSON,encoding='utf-8') as f:
    valList = json.load(f)
    
mappingList = [2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 2, 2, 1, 0, 1, 0, 1, 1,
       0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 2,
       1, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2]

for part in range(3):
    partindex = [k for k,v in zip(range(80),mappingList) if v==part]
    exec('d_{0} = dict(zip(partindex, range(len(partindex))))'.format(part))
    exec('print d_{0}'.format(part))
    

    
for part in range(3):
    with codecs.open('./train_{0}.lst'.format(part),'w',encoding='utf-8') as f:
        counter=0
        for idx,d in enumerate(trainList):
            if idx%100==0:
                print idx
            i = int(d['label_id'])
            if mappingList[i] != part:
                continue
            s = ''
            exec('s = d_{idx}[i]'.format(idx=part))
            f.write('{0}\t{1}\t{2}\n'.format(counter, s , d['image_id']))
            counter+=1

        
    with codecs.open('./val_{0}.lst'.format(part),'w',encoding='utf-8') as f:
        counter=0
        for idx,d in enumerate(valList):
            if idx%100==0:
                print idx
            i = int(d['label_id'])
            if mappingList[i] != part:
                continue
            s = ''
            exec('s = d_{idx}[i]'.format(idx=part))
            f.write('{0}\t{1}\t{2}\n'.format(counter, s , d['image_id']))
            counter+=1
        
        
    with codecs.open('./trainval_{0}.lst'.format(part),'w',encoding='utf-8') as f:
        counter=0
        for idx,d in enumerate(trainList):
            if idx%100==0:
                print idx
            i = int(d['label_id'])
            if mappingList[i] != part:
                continue
            s = ''
            exec('s = d_{idx}[i]'.format(idx=part))
            f.write('{0}\t{1}\t{2}\n'.format(counter, s , d['image_id']))
            counter+=1
        for idx,d in enumerate(valList):
            if idx%100==0:
                print idx
            i = int(d['label_id'])
            if mappingList[i] != part:
                continue
            s = ''
            exec('s = d_{idx}[i]'.format(idx=part))
            f.write('{0}\t{1}\t{2}\n'.format(counter, s , d['image_id']))
            counter+=1
        

with codecs.open('./train_all.lst'.format(part),'w',encoding='utf-8') as f:
    counter=0
    for idx,d in enumerate(trainList):
        if idx%100==0:
            print idx
        i = int(d['label_id'])
        f.write('{0}\t{1}\t{2}\n'.format(counter, mappingList[i], d['image_id']))
        counter+=1

    
with codecs.open('./val_all.lst'.format(part),'w',encoding='utf-8') as f:
    counter=0
    for idx,d in enumerate(valList):
        if idx%100==0:
            print idx
        i = int(d['label_id'])
        f.write('{0}\t{1}\t{2}\n'.format(counter, mappingList[i], d['image_id']))
        counter+=1
    
    
with codecs.open('./trainval_all.lst'.format(part),'w',encoding='utf-8') as f:
    counter=0
    for idx,d in enumerate(trainList):
        if idx%100==0:
            print idx
        i = int(d['label_id'])
        f.write('{0}\t{1}\t{2}\n'.format(counter, mappingList[i], d['image_id']))
        counter+=1
    for idx,d in enumerate(valList):
        if idx%100==0:
            print idx
        i = int(d['label_id'])
        f.write('{0}\t{1}\t{2}\n'.format(counter, mappingList[i], d['image_id']))
        counter+=1
        
'''

with codecs.open('./test_a.lst','w',encoding='utf-8') as f:
    for idx, jpgfilename in enumerate(glob.glob(DATA_TEST_A_ROOT+'/*.jpg')):
        if idx%100==0:
            print idx
        jpgfilename = jpgfilename.split('/')[-1]
        f.write('{0}\t{1}\t{2}\n'.format(idx, 99, jpgfilename))
    
    
    '''
    
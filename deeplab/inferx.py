import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import sys
import argparse
import numpy as np
import cv2
import math
import datetime
import random
import json
import pandas as pd
#import multiprocessing
from Queue import Queue
from threading import Thread
import mxnet as mx
import mxnet.ndarray as nd
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description="",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test-lst', type=str, default='test_a.lst',
    help='')
parser.add_argument('--val-lst', type=str, default='val.lst',
    help='')
parser.add_argument('--val-root-path', type=str, default='/data1/deepinsight/aichallenger/scene/')
parser.add_argument('--test-root-path', type=str, default='/data1/deepinsight/aichallenger/scene/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922/')
parser.add_argument('--gpu', type=int, default=0,help='')
parser.add_argument('--num-classes', type=int, default=80,help='')
parser.add_argument('--batch-size', type=int, default=32,help='')
parser.add_argument('--mode', type=int, default=0, help='')
parser.add_argument('--mean-max', action="store_true", help='')
parser.add_argument('--expand', type=int, default=12, help='')
#parser.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939', help='set to empty if no mean used')
parser.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939', help='set to empty if no mean used')
parser.add_argument('--layer', type=str, default='fullyconnected0', help='') #flatten0
#parser.add_argument('--model', type=str, default='./model/ft448deformsqex0.0001_9682,3|./model/sft320deformsqex_9692,1')
#parser.add_argument('--model', type=str, default='./model/sft320deformsqex_9692,1')
#parser.add_argument('--model', type=str, default='./model/ft224deformsqex0003_9587,20')
#parser.add_argument('--model', type=str, default='./model/a1,8,14')
#parser.add_argument('--model', type=str, default='./model/a1_2,2,6')
#parser.add_argument('--model', type=str, default='./model/a1_6,1')
#parser.add_argument('--model', type=str, default='./model/a1_6,1|./model/a1_4,3|./model/a1_5,6|./model/a1_7,2')
#parser.add_argument('--model', type=str, default='./model/a1_6,6|./model/a1_4,6|./model/a1_5,6|./model/a1_7,6')
#parser.add_argument('--model', type=str, default='./model/ft224nude0003_97,50,224|./model/sft448from32097nude00003_9740,11,448|./model/sft320nude00003_97,19,320')
#parser.add_argument('--model', type=str, default='./model/ft224nude0003_97,50,224|./model/sft448from32097nude00003_9740,11,448')
#parser.add_argument('--model', type=str, default='./model/sft448from32097nude00003_9740,11,448|./model/sft320nude00003_97,19,320')
#parser.add_argument('--model', type=str, default='./model/sft448from32097nude00003_9740,11,448')
#parser.add_argument('--model', type=str, default='./model/ft224nude0003_97,50,224|./model/sft448from32097nude00003_9740,11,448')
#parser.add_argument('--model', type=str, default='./model/sft448from32097nude00003_9740,11,448|./model/ft224nude0003_97,50,224')
#parser.add_argument('--model', type=str, default='./model/ft224nude0003_97,50,224')
parser.add_argument('--model', type=str, default='/data1/deepinsight/CAIScene/irv2models/irv2ft1,18,299')  
#/data1/deepinsight/CAIScene/dpn92_9613,3,320
#parser.add_argument('--model', type=str, default='./model/irv2ft1,40,299')
parser.add_argument('--output', type=str, default='./224e8', help='')
args = parser.parse_args()

def prt(msg):
  ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  print("%s] %s" % (ts, msg))
  sys.stdout.flush()

def ch_dev(arg_params, aux_params, ctx):
  new_args = dict()
  new_auxs = dict()
  for k, v in arg_params.items():
    new_args[k] = v.as_in_context(ctx)
  for k, v in aux_params.items():
    new_auxs[k] = v.as_in_context(ctx)
  return new_args, new_auxs

def read_image(path):
  img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
  img = np.float32(img)
  return img

def image_preprocess2(img, crop_sz, expandid, blockid, cornerid, flipid):
  nd_img = nd.array(img)
  if len(args.rgb_mean)>0:
    rgb_mean = [float(x) for x in args.rgb_mean.split(',')]
    rgb_mean = np.array(rgb_mean, dtype=np.float32).reshape(1,1,3)
    rgb_mean = nd.array(rgb_mean)
    nd_img -= rgb_mean
    nd_img *= 0.0078125
  #expand = 32
  #if crop_sz<300:
  #  expand = 16
  img_sz = crop_sz+args.expand*(expandid+1)
  #img_sz = crop_sz+int(crop_sz/7)*(expandid+1)
  nd_img = mx.image.resize_short(nd_img, img_sz)
  if flipid==1:
    nd_img = nd.flip(nd_img, axis=1)
  img = nd_img.asnumpy()
  h = img.shape[0]
  w = img.shape[1]
  block_size = min(h,w)
  blockh = 0
  blockw = 0
  if h>w:
    if blockid==1:
      _half = int( (h-w)/2 )
      blockh = _half
    elif blockid==2:
      blockh = h-w
  else:
    if blockid==1:
      _half = int( (w-h)/2 )
      blockw = _half
    elif blockid==2:
      blockw = w-h 
  block = img[blockh:(blockh+block_size), blockw:(blockw+block_size), :]
  if cornerid==5:
    img = cv2.resize(block, (crop_sz, crop_sz))
  else:
    if cornerid==0:
      cornerh = int((block_size-crop_sz)/2)
      cornerw = int((block_size-crop_sz)/2)
    elif cornerid==1:
      cornerh = 0
      cornerw = 0
    elif cornerid==2:
      cornerh = 0
      cornerw = block_size-crop_sz
    elif cornerid==3:
      cornerh = block_size-crop_sz
      cornerw = 0
    elif cornerid==4:
      cornerh = block_size-crop_sz
      cornerw = block_size-crop_sz
    img = block[cornerh:(cornerh+crop_sz), cornerw:(cornerw+crop_sz), :]


  img = np.swapaxes(img, 0, 2)
  img = np.swapaxes(img, 1, 2)  # change to CHW
  #print(img.shape)
  return img

def val(X, imgs):
  top1=0
  top3=0
  for ii in range(X.shape[0]):
    score = X[ii]
    gt_label = imgs[ii][1]
    #print("%d sum %f" % (ii, _sum))
    sort_index = np.argsort(score)[::-1]
    for k in xrange(3):
      if sort_index[k]==gt_label:
        top3+=1
        if k==0:
          top1+=1
  print('top3', float(top3)/X.shape[0])
  print('top1', float(top1)/X.shape[0])

if args.mode>0:
  args.root_path = args.test_root_path
  args.lst = args.test_lst
else:
  args.root_path = args.val_root_path
  args.lst = args.val_lst


#ctxs = [mx.gpu(int(i)) for i in args.gpus.split(',')]

nets = []
gpuid = args.gpu
if gpuid>=0:
  ctx = mx.gpu(gpuid)
else:
  ctx = mx.cpu()
for model_str in args.model.split('|'):
  vec = model_str.split(',')
  assert len(vec)>1
  prefix = vec[0]
  epoch = int(vec[1])
  crop_sz = int(vec[2])
  print('loading',prefix, epoch)
  net = edict()
  net.crop_sz = crop_sz
  net.ctx = ctx
  net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_symbols = net.sym.get_internals()
  print all_symbols
  net.sym = all_symbols[args.layer+'_output']
  if args.mean_max:
    print('use mean_max')
    all_symbols = net.sym.get_internals()
    relu1 = all_symbols['relu1_output']
    poola = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='poola')
    poolb = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='max', name='poolb')
    pool1 = poola*0.5+poolb*0.5
    flat = mx.sym.Flatten(data=pool1, name='flatten0')
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=args.num_classes, name='fc1')
    softmax = mx.sym.SoftmaxOutput(data=fc1, name='softmax')
    net.sym = softmax
  net.arg_params, net.aux_params = ch_dev(net.arg_params, net.aux_params, net.ctx)
  nets.append(net)

assert len(nets)==1

imgs = []
i = 0
for line in open(args.lst, 'r'):
  vec = line.strip().split("\t")
  imgs.append( (i, int(vec[1]), os.path.join(args.root_path, vec[2])) )
  i+=1

#models = []
#for net in nets:
#  model = mx.mod.Module(
#      context       = ctxs,
#      symbol        = net.sym,
#  )
#  hw = int(args.size.split(',')[0])
#  model.bind(data_shapes=[('data', (args.batch_size, 3, hw, hw))], label_shapes=[('softmax_label',(args.batch_size,))], for_training=False, grad_req="null")
#  model.set_params(net.arg_params, net.aux_params)
#  models.append(model)


#X = np.zeros( (len(imgs), args.num_classes) , dtype=np.float32 )
X = None

num_batches = int( math.ceil(len(imgs) / args.batch_size) )
print("num_batches %d" % num_batches)

crop_id = 0
for expandid in xrange(0,1):
  for blockid in [0,1,2]:
    for cornerid in xrange(0,6):
      for flipid in xrange(0,2):
        print('start loop', expandid, blockid, cornerid, flipid)
        score_weight = 1.0
        #if blockid==1:
        #  score_weight = 2.0
        batch_head = 0
        batch_num = 0
        while batch_head<len(imgs):
          prt("processing batch %d" % batch_num)
          current_batch_sz = min(args.batch_size, len(imgs)-batch_head)
          #print batch_head
          ids = []
          datas = []
          for index in range(batch_head, batch_head+current_batch_sz):
            img_path = imgs[index][2]
            data = read_image(img_path)
            datas.append(data)
            ids.append(imgs[index][0])

          #assert len(datas)==1
          #_data = datas[0]
          #_hw = min(_data.shape[0], _data.shape[1])
          #if _hw<256:
          #  _nets = [nets[0]]
          #else:
          #  _nets = nets

          #for model in models:
          for net in nets:
            input_blob = np.zeros((current_batch_sz,3,net.crop_sz,net.crop_sz))
            for idx in xrange(len(datas)):
              data = datas[idx]
              img = image_preprocess2(data, net.crop_sz, expandid, blockid, cornerid, flipid)
              #print(img.shape)
              input_blob[idx,:,:,:] = img
            #print(input_blob.shape)
            net.arg_params["data"] = mx.nd.array(input_blob, net.ctx)
            net.arg_params["softmax_label"] = mx.nd.empty((current_batch_sz,), net.ctx)
            exe = net.sym.bind(net.ctx, net.arg_params ,args_grad=None, grad_req="null", aux_states=net.aux_params)
            exe.forward(is_train=False)
            net_out = exe.outputs[0].asnumpy()

            #_data = mx.nd.array(input_blob)
            #_label = nd.ones( (current_batch_sz,) )
            #db = mx.io.DataBatch(data=(_data,), label=(_label,))
            #model.forward(db, is_train=False)
            #net_out = model.get_outputs()[0].asnumpy()
            #print(net_out.shape)

            for bz in xrange(current_batch_sz):
              feature = np.squeeze(net_out[bz,:])
              if X is None:
                X = np.zeros( (len(imgs), 36, len(feature)), dtype=np.float32 )
                print('init X',X.shape)
              #print(score)
              im_id = ids[bz]
              X[im_id, crop_id, :] = feature
              #print('set',im_id, crop_id, feature.shape)

          batch_head += current_batch_sz
          batch_num += 1
        crop_id+=1
        #np.save(args.output, X)

np.save(args.output, X)


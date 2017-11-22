import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import sys
import argparse
import numpy as np
import cv2
import math
import pickle
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
from bson_data import BsonImageIter

parser = argparse.ArgumentParser(description="",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test-lst', type=str, default='/data/20171010/test.bson',
    help='')
parser.add_argument('--val-lst', type=str, default='./data/val.lst',
    help='')
parser.add_argument('--gpu', type=int, default=0,
    help='')
parser.add_argument('--batch-size', type=int, default=256,
    help='')
parser.add_argument('--mode', type=int, default=0,
    help='')
parser.add_argument('--mean-max', action="store_true",
    help='')
parser.add_argument('--expand', type=int, default=32,
    help='')
#parser.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939',help='a tuple of size 3 for the mean rgb')
parser.add_argument('--rgb-mean', type=str, default='',help='a tuple of size 3 for the mean rgb')
parser.add_argument('--model', type=str, default='./model/_job-6435_model_scene-resnet-152,52,224')
#parser.add_argument('--model', type=str, default='./model/_job-6608_model_discount-resnet-50,48,224')
parser.add_argument('--output-dir', type=str, default='/data/cdiscount/test152',
    help='')
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
  #print(img.shape)
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



#ctxs = [mx.gpu(int(i)) for i in args.gpus.split(',')]

nets = []
gpuid = args.gpu
ctx = mx.gpu(gpuid)
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
    #print(all_symbols)
  net.arg_params, net.aux_params = ch_dev(net.arg_params, net.aux_params, net.ctx)
  nets.append(net)


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

it = BsonImageIter(path_bson=args.test_lst, 
    batch_size=args.batch_size, data_shape=(3,180,180), 
    path_labelmap='./synset.txt',
    )
num_samples = it.num_samples()

num_batches = int( math.ceil(num_samples / args.batch_size) )
print("num_batches %d" % num_batches)
if not os.path.exists(args.output_dir):
  os.makedirs(args.output_dir)

#X = None
#Y = np.zeros( (num_samples, ), dtype=np.int)

crop_id = 0
for expandid in xrange(0,1):
  for blockid in [0]:
    for cornerid in xrange(0,6):
      for flipid in xrange(0,2):
        print('start loop', expandid, blockid, cornerid, flipid)
        pk_path = os.path.join(args.output_dir, 'pk.%d'%crop_id)
        pk_in = None
        if crop_id>0:
            pk_in_path = os.path.join(args.output_dir, 'pk.%d'%(crop_id-1))
            pk_in = open(pk_in_path, 'rb')
        pk_out = open(pk_path, 'wb')
        ba = 0
        batch_num = 0
        while True:
          try:
            batch_num+=1
            if batch_num%10==0:
              prt('processing batch num %d' % batch_num)
            batch = it.next()
            for net in nets:
              batch_data = batch.data[0].asnumpy()
              batch_label = batch.label[0].asnumpy()
              #print(batch_label.shape)
              #print(batch_data.shape)
              current_bz = batch_data.shape[0] - batch.pad
              batch_data = np.transpose(batch_data, (0, 2,3,1))
              #print(batch_data.shape)
              bb = ba+current_bz
              input_blob = np.zeros((current_bz,3,net.crop_sz,net.crop_sz))
              for i in xrange(current_bz):
                input_blob[i] = image_preprocess2(batch_data[i], net.crop_sz, expandid, blockid, cornerid, flipid)
              net.arg_params["data"] = mx.nd.array(input_blob, net.ctx)
              net.arg_params["softmax_label"] = mx.nd.empty((current_bz,), net.ctx)
              exe = net.sym.bind(net.ctx, net.arg_params ,args_grad=None, grad_req="null", aux_states=net.aux_params)
              exe.forward(is_train=False)
              net_out = exe.outputs[0].asnumpy()
              for i in xrange(current_bz):
                probs = net_out[i,:]
                score = np.squeeze(probs)
                #if X is None:
                #  X = np.zeros( (num_samples, len(score)), dtype=np.float32 )
                #  print('init X', X.shape)
                #X[ba+i, :] += score
                #Y[ba+i] = batch_label[i]
                score0 = None
                if pk_in is not None:
                    score0 = pickle.load(pk_in)
                    score0 += score
                else:
                    score0 = score
                pickle.dump(score0, pk_out, pickle.HIGHEST_PROTOCOL)
            ba = bb
          except StopIteration:
            it.reset()
            break

        if pk_in is not None:
            pk_in.close()
        pk_out.close()
        if crop_id>0:
            os.remove(pk_in_path)
	crop_id += 1


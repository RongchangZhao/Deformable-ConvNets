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
parser.add_argument('--lst', type=str, default='./data/val.lst',
    help='')
parser.add_argument('--val-root-path', type=str, default='/raid5data/dplearn/aichallenger/scene/val')
parser.add_argument('--test-root-path', type=str, default='/raid5data/dplearn/aichallenger/scene/test_a')
parser.add_argument('--gpu', type=int, default=0,
    help='')
parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7',
    help='')
parser.add_argument('--num-classes', type=int, default=80,
    help='')
parser.add_argument('--batch-size', type=int, default=128,
    help='')
parser.add_argument('--mode', type=int, default=0,
    help='')
parser.add_argument('--size', type=str, default='448,504')
#parser.add_argument('--size', type=str, default='224,256')
parser.add_argument('--step', type=int, default=-40,
    help='if negative, use random crops')
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
parser.add_argument('--model', type=str, default='./model/ft224nude0003_97,50,224|./model/sft448from32097nude00003_9740,11,448')
#parser.add_argument('--model', type=str, default='./model/sft448from32097nude00003_9740,11,448|./model/ft224nude0003_97,50,224')
parser.add_argument('--output-dir', type=str, default='./rt',
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

def image_preprocess(img_full_path):
  _size = args.size.split(",")
  img_sz = int(_size[1])
  crop_sz = int(_size[0])
  #print(img_full_path)
  img = cv2.cvtColor(cv2.imread(img_full_path), cv2.COLOR_BGR2RGB)
  img = np.float32(img)
  ori_shape = img.shape
  assert img.shape[2]==3
  rows, cols = img.shape[:2]

  _high = min(rows, cols)
  _high = min(_high, crop_sz*2)
  _high = max(_high, img_sz)

  _img_sz = img_sz
  if _high>img_sz:
    _img_sz = np.random.randint(low=img_sz, high=_high)
  if cols < rows:
    resize_width = _img_sz
    resize_height = resize_width * rows / cols;
  else:
    resize_height = _img_sz
    resize_width = resize_height * cols / rows;

  img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
  #print(_high,ori_shape,img.shape)

  h, w, _ = img.shape
  #x0 = int((w - crop_sz) / 2)
  #y0 = int((h - crop_sz) / 2)

  x0_max = w-crop_sz
  y0_max = h-crop_sz
  x0 = np.random.randint(low=0, high=x0_max)
  y0 = np.random.randint(low=0, high=y0_max)

  img = img[y0:y0+crop_sz, x0:x0+crop_sz, :]

  #lr flip
  if random.randint(0,1)==1:
    for j in xrange(3):
      img[:,:,j] = np.fliplr(img[:,:,j])
  img = np.swapaxes(img, 0, 2)
  img = np.swapaxes(img, 1, 2)  # change to CHW
  return img

def read_image(path):
  img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
  img = np.float32(img)
  return img

def image_preprocess2(img, crop_sz, blockid, cornerid, flipid):
  nd_img = nd.array(img)
  expand = 32
  #if crop_sz<300:
  #  expand = 16
  img_sz = crop_sz+expand
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
else:
  args.root_path = args.val_root_path
args.crop_size = int(args.size.split(',')[0])
args.resize = int(args.size.split(',')[1])


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
  net.arg_params, net.aux_params = ch_dev(net.arg_params, net.aux_params, net.ctx)
  nets.append(net)
  gpuid+=1

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


X = np.zeros( (len(imgs), args.num_classes) , dtype=np.float32 )

num_batches = int( math.ceil(len(imgs) / args.batch_size) )
print("num_batches %d" % num_batches)
if not os.path.exists(args.output_dir):
  os.makedirs(args.output_dir)

for blockid in [1,0,2]:
  for cornerid in xrange(0,5):
    for flipid in xrange(0,2):
      print('start loop', blockid, cornerid, flipid)
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
            img = image_preprocess2(data, net.crop_sz, blockid, cornerid, flipid)
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
            probs = net_out[bz,:]
            score = np.squeeze(probs)
            score *= score_weight
            #print(score.shape)
            #print(score)
            im_id = ids[bz]
            X[im_id,:] += score

        batch_head += current_batch_sz
        batch_num += 1
      val(X, imgs)
      out_filename = os.path.join(args.output_dir, 'result.hdf')
      print(out_filename)
      if os.path.exists(out_filename):
        print("exists, delete first..")
        os.remove(out_filename)
      _X = X
      print("_X row sum %f" % np.sum(_X[0]))
      df = pd.DataFrame(_X)
      df.to_hdf(out_filename, "result")



top1 = 0
top5 = 0
if args.mode==0:
  val(X, imgs)
else:
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  with open(os.path.join(args.output_dir,'result.json'), 'w') as opfile:
    json_data = []
    for ii in range(X.shape[0]):
      score = X[ii]
      #print("%d sum %f" % (ii, _sum))
      sort_index = np.argsort(score)[::-1]
      top_k = list(sort_index[0:3])
      _data = {'image_id' : imgs[ii][2].split('/')[-1], 'label_id': top_k}
      json_data.append(_data)
    opfile.write(json.dumps(json_data))

  out_filename = os.path.join(args.output_dir, 'result.hdf')
  print(out_filename)
  if os.path.exists(out_filename):
    print("exists, delete first..")
    os.remove(out_filename)
  _X = X
  print("_X row sum %f" % np.sum(_X[0]))
  df = pd.DataFrame(_X)
  df.to_hdf(out_filename, "result")


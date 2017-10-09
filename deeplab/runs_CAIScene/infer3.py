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
parser.add_argument('--lst', type=str, default='val.lst',
    help='')
parser.add_argument('--val-root-path', type=str, default='/data1/deepinsight/aichallenger/scene/ai_challenger_scene_validation_20170908/scene_validation_images_20170908')
parser.add_argument('--test-root-path', type=str, default='/data1/deepinsight/aichallenger/scene/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922')
parser.add_argument('--gpu', type=int, default=0,
    help='')
parser.add_argument('--gpus', type=str, default='0,1,2',
    help='')
parser.add_argument('--num-classes', type=int, default=80,
    help='')
parser.add_argument('--batch-size', type=int, default=120,
    help='')
parser.add_argument('--mode', type=int, default=0,
    help='')
parser.add_argument('--size', type=str, default='448,504')
#parser.add_argument('--size', type=str, default='224,256')
parser.add_argument('--step', type=int, default=-32,
    help='if negative, use random crops')
#parser.add_argument('--model', type=str, default='./model/ft448deformsqex0.0001_9682,3|./model/sft320deformsqex_9692,1')
#parser.add_argument('--model', type=str, default='./model/sft320deformsqex_9692,1')
#parser.add_argument('--model', type=str, default='./model/ft224deformsqex0003_9587,20')
#parser.add_argument('--model', type=str, default='./model/a1,8,14')
#parser.add_argument('--model', type=str, default='./model/a1_2,2,6')
#parser.add_argument('--model', type=str, default='./model/a1_6,1')
#parser.add_argument('--model', type=str, default='./model/a1_6,1|./model/a1_4,3|./model/a1_5,6|./model/a1_7,2')
#parser.add_argument('--model', type=str, default='./model/a1_6,6|./model/a1_4,6|./model/a1_5,6|./model/a1_7,6')
#parser.add_argument('--model', type=str, default='./model/sft448from32097nude00003_9740,11,448')
#parser.add_argument('--model', type=str, default='./model/ft224nude0003_97,50,224|./model/sft448from32097nude00003_9740,11,448|./model/sft320nude00003_97,19,320')
#parser.add_argument('--model', type=str, default='./model/ft224nude0003_97,50,224|./model/sft448from32097nude00003_9740,11,448')
parser.add_argument('--model', type=str, default='ft224nude0003_9685,27,224|sft448from32097nude00003_9740,11,448') #  #ft224nude0003_97,50,224|
#parser.add_argument('--model', type=str, default='sft320nude00003_9736,5,320|sft640from448974nude00002_973,12,640')
parser.add_argument('--output-dir', type=str, default='',
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

  x0_side = int((w-crop_sz)/4)
  y0_side = int((h-crop_sz)/4)

  x0_max = w-crop_sz
  y0_max = h-crop_sz
  x0 = np.random.randint(low=x0_side, high=x0_max-x0_side)
  y0 = np.random.randint(low=y0_side, high=y0_max-y0_side)

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

def image_preprocess2(img, crop_sz):
  nd_img = nd.array(img)
  img_sz = crop_sz+random.randint(8,32)
  if args.step==0:
    img_sz = crop_sz+32
  if img_sz>0:
    nd_img = mx.image.resize_short(nd_img, img_sz)
  #nd_img = mx.image.random_size_crop(nd_img, (crop_sz, crop_sz), 0.08, (3.0/4, 4.0/3))[0]
  if args.step==0:
    nd_img = mx.image.center_crop(nd_img, (crop_sz, crop_sz))[0]
  else:
    nd_img = mx.image.random_crop(nd_img, (int((img_sz+crop_sz)/2),int((img_sz+crop_sz)/2)) )[0]
    nd_img = mx.image.center_crop(nd_img, (crop_sz, crop_sz))[0]
    if random.random()<0.5:
      nd_img = nd.flip(nd_img, axis=1)
  img = nd_img.asnumpy()
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
  #gpuid+=1

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
nrof_loops = args.step*-1
if args.step==0:
  nrof_loops = 1

for loop in xrange(nrof_loops):
  print('start loop', loop)
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

    #for model in models:
    for net in nets:
      input_blob = np.zeros((current_batch_sz,3,net.crop_sz,net.crop_sz))
      for idx in xrange(len(datas)):
        data = datas[idx]
        img = image_preprocess2(data, net.crop_sz)
        input_blob[idx,:,:,:] = img
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
        #print(score.shape)
        #print(score)
        im_id = ids[bz]
        X[im_id,:] += score

    batch_head += current_batch_sz
    batch_num += 1
  val(X, imgs)



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
  _X = X/float(nrof_loops)
  print("_X row sum %f" % np.sum(_X[0]))
  df = pd.DataFrame(_X)
  df.to_hdf(out_filename, "result")


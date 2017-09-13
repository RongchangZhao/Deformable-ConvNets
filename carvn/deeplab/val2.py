# pylint: skip-file
import numpy as np
import mxnet as mx
import argparse
import datetime
import cv2
import sys
import os
from PIL import Image
from easydict import EasyDict as edict
import multiprocessing


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

def getpallete(num_cls):
    # this function is to get the colormap for visualizing the segmentation mask
    n = num_cls
    pallete = [0]*(n*3)
    for j in xrange(0,n):
            lab = j
            pallete[j*3+0] = 0
            pallete[j*3+1] = 0
            pallete[j*3+2] = 0
            i = 0
            while (lab > 0):
                    pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return pallete

pallete = getpallete(256)


image_size = (1280, 1918)

def get_img(img_path, cutoff=None):
  """get the (1, 3, h, w) np.array data for the img_path"""
  #mean = np.array([123.68, 116.779, 103.939])  # (R,G,B)
  img = Image.open(img_path)
  img = np.array(img, dtype=np.float32)
  #print(img.shape)
  if cutoff is not None and cutoff>0:
    img = cv2.resize(img, (cutoff, cutoff))
  #print(img.shape)
  #reshaped_mean = mean.reshape(1, 1, 3)
  #img = img - reshaped_mean
  #print(img.shape)
  #img = cv2.resize(img, (1280, 1280) )
  img = np.swapaxes(img, 0, 2)
  img = np.swapaxes(img, 1, 2)
  #img = np.expand_dims(img, axis=0)
  return img

def task_mask_prob(net):
  data = net.data
  if len(data.shape)==3:
    data = np.expand_dims(data, axis=0)
  net.arg_params["data"] = mx.nd.array(data, net.ctx)
  data_shape = net.arg_params["data"].shape
  #print(data_shape)
  label_shape = (1, data_shape[2], data_shape[3])
  net.arg_params["softmax_label"] = mx.nd.empty(label_shape, net.ctx)
  exector = net.sym.bind(net.ctx, net.arg_params,args_grad=None, grad_req="null", aux_states=net.aux_params)
  exector.forward(is_train=False)
  output = exector.outputs[0]
  prob = output.asnumpy()
  #out_img = np.uint8(np.squeeze(prob.argmax(axis=1)))
  out_prob = np.squeeze(prob)
  return out_prob


def get_mask_prob(data, nets):
  if not isinstance(nets, list):
    nets = [nets]


  if len(data.shape)==3:
    data = np.expand_dims(data, axis=0)
  out_prob = None
  for net in nets:
    net.arg_params["data"] = mx.nd.array(data, net.ctx)
    data_shape = net.arg_params["data"].shape
    #print(data_shape)
    label_shape = (1, data_shape[2], data_shape[3])
    net.arg_params["softmax_label"] = mx.nd.empty(label_shape, net.ctx)
    exector = net.sym.bind(net.ctx, net.arg_params,args_grad=None, grad_req="null", aux_states=net.aux_params)
    exector.forward(is_train=False)
    output = exector.outputs[0]
    prob = output.asnumpy()
    #out_img = np.uint8(np.squeeze(prob.argmax(axis=1)))
    _out_prob = np.squeeze(prob)
    if out_prob is None:
      out_prob = _out_prob
    else:
      out_prob += _out_prob
  #print(prob.shape, out_prob.shape)
  return out_prob

#POOL = multiprocessing.Pool(5)
def _get_mask_prob(data, nets):
  global POOL
  if not isinstance(nets, list):
    nets = [nets]
  for net in nets:
    net.data = data
  results = []
  for net in nets:
    result = POOL.apply_async(task_mask_prob, (net,))
    results.append(result)
  out_prob = None
  for result in results:
    _out_prob = result.get()
    if out_prob is None:
      out_prob = _out_prob
    else:
      out_prob += _out_prob
  return out_prob

def prob_to_out(prob):
  out_img = np.uint8(np.squeeze(prob.argmax(axis=0)))
  #print(out_img.shape)
  out_img = cv2.resize(out_img, (image_size[1], image_size[0]))
  #print(out_img.shape)
  return out_img

def do_flip(data):
  for i in xrange(data.shape[0]):
    data[i,:,:] = np.fliplr(data[i,:,:])

def get_mask(img_path, cutoff, net):
  img = get_img(img_path)

  mask = np.zeros( (2,img.shape[1], img.shape[2]), dtype=np.float32 )
  #dense prediction
  #moves = [2,4] # moves in h,w
  #assert (img.shape[1]-cutoff)%(moves[0]-1)==0
  #assert (img.shape[2]-cutoff)%(moves[1]-1)==0
  #moves = [ (img.shape[1]-cutoff)/(moves[0]-1), (img.shape[2]-cutoff)/(moves[1]-1) ]
  #for h in xrange(0,img.shape[1]-cutoff+1,moves[0]):
  #  for w in xrange(0, img.shape[2]-cutoff+1, moves[1]):
  #    _img = img[:,h:(h+cutoff),w:(w+cutoff)]
  #    #print(h,w)
  #    _mask = get_mask_prob(_img)
  #    mask[:,h:(h+cutoff),w:(w+cutoff)] += _mask
  use_flip = False
  #step = cutoff
  step = 256
  for flip in [0,1]:
    if not use_flip and flip==1:
      continue
    for x in xrange(0, img.shape[1],step):
      xstart = x
      xstop = min(xstart+cutoff[0],img.shape[1])
      xstart = xstop-cutoff[0]
      for y in xrange(0, img.shape[2], step):
        ystart = y
        ystop = min(ystart+cutoff[1],img.shape[2])
        ystart = ystop-cutoff[1]
        #print(xstart,ystart,xstop,ystop)
        _img = img[:,xstart:xstop,ystart:ystop]
        if flip==1:
          do_flip(_img)
        _mask = get_mask_prob(_img, net)
        if flip==1:
          do_flip(_mask)
        mask[:,xstart:xstop,ystart:ystop] += _mask
        #mask[:,xstart:xstop,ystart:ystop] = np.maximum(mask[:, xstart:xstop, ystart:ystop], _mask)
        if ystop>=img.shape[2]:
          break
      if xstop>=img.shape[1]:
        break

  return prob_to_out(mask)

def rle_encode(mask_image):
  pixels = mask_image.flatten()
  # We avoid issues with '1' at the start or end (at the corners of 
  # the original image) by setting those pixels to '0' explicitly.
  # We do not expect these to be non-zero for an accurate mask, 
  # so this should not harm the score.
  pixels[0] = 0
  pixels[-1] = 0
  runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
  runs[1::2] = runs[1::2] - runs[:-1:2]
  return runs


def rle_to_string(runs):
  return ' '.join(str(x) for x in runs)


def main():
  ensembles = [ ('./model/deeplab5', 12), ('./model/deeplab6', 11), ('./model/deeplab7', 9) ]
  ensembles = [ ('./model/deeplab7', 9) ]
  #ensembles = [ ('./model/deeplab-remote/DeeplabV3-ResNeXt-152L64X1D4XP', 67) ]
  #ensembles = [ ('./model/deeplab-remote/_job-5145_output_DeeplabV3-ResNeXt-152L64X1D4XP', 67) ]
  DATA_ROOT = '/raid5data/dplearn/carvn'
  parser = argparse.ArgumentParser(description='carvn submit')
  parser.add_argument('--gpu', type=int, default=7,
      help='gpu for use.')
  parser.add_argument('--cutoff', type=int, default=1200,
      help='cutoff size.')
  parser.add_argument('--parts', default='',
      help='test parts.')
  args = parser.parse_args()
  ctx = mx.gpu(args.gpu)
  nets = []
  for m in ensembles:
    net = edict()
    net.ctx = ctx
    #net.ctx = mx.gpu(args.gpu)
    print('loading', m[0], m[1])
    net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(m[0], m[1])
    net.arg_params, net.aux_params = ch_dev(net.arg_params, net.aux_params, net.ctx)
    nets.append(net)
  #test_data_dir = os.path.join(DATA_ROOT, 'test')
  suffix = '_'.join(args.parts.split(','))
  parts = {}
  for p in args.parts.split(','):
    if len(p)==0:
      continue
    parts[int(p)] = 1
  dice = [0.0, 0.0]
  pp = 0
  cutoff = args.cutoff
  cutoff = [1280, 1918]
  use_hq = True
  car_dice = []
  last_carid = ''
  for line in open('../data/val.lst', 'r'):
    vec = line.strip().split("\t")
    img = vec[1]
    if use_hq:
      img = img.replace('/train/', '/train_hq/')
      #print(img)
    id = img
    pos = int(img.split('/')[-1].split('.')[0].split('_')[1])
    carid = img.split('/')[-1].split('.')[0].split('_')[0]
    if len(parts)>0 and not pos in parts:
      #print("skip %d"%pos)
      continue
    #print(img)
    mask = get_mask(img, cutoff, nets)
    pred_label = mask.flatten()
    #pred_label = cv2.resize(mask, (image_size[1], image_size[0])).flatten()
    gt_mask = Image.open(vec[2])
    label = np.array(gt_mask, dtype=np.uint8).flatten()
    assert pred_label.shape==label.shape
    #print(pred_label.shape, label.shape)
    dice[1] += 1.0
    pred_label_sum = np.sum(pred_label)
    label_sum = np.sum(label)
    intersection = 0.0
    ret = 0.0
    if pred_label_sum==0 and label_sum==0:
      sys.exit(0)
      dice[0] += 1.0
    else:
      intersection = np.sum(pred_label * label)
      ret = (2. * intersection) / (pred_label_sum + label_sum)
      dice[0] += ret
    #print(dice[0]/dice[1])
    if carid!=last_carid and len(car_dice)>0:
      mean = sum(car_dice)/len(car_dice)
      print(last_carid, mean)
    if ret<0.9968:
      print(img, ret)

    car_dice.append(ret)
    last_carid = carid
    pp+=1
    if pp%10==0:
      prt("Processing %d"%pp)
  print('final', dice[0]/dice[1])

if __name__ == "__main__":
    main()


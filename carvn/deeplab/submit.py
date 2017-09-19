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
#img = "./person_bicycle.jpg"
#seg = img.replace("jpg", "png")


def get_img(img_path):
  """get the (1, 3, h, w) np.array data for the img_path"""
  mean = np.array([123.68, 116.779, 103.939])  # (R,G,B)
  reshaped_mean = mean.reshape(1, 1, 3)
  img = Image.open(img_path)
  img = np.array(img, dtype=np.float32)
  img = img - reshaped_mean
  #print(img.shape)
  #img = cv2.resize(img, (1280, 1280) )
  img = np.swapaxes(img, 0, 2)
  img = np.swapaxes(img, 1, 2)
  #img = np.expand_dims(img, axis=0)
  return img


def get_label(label_path):
  img = Image.open(label_path)
  img = np.array(img, dtype=np.uint8)
  img = np.swapaxes(img, 0, 2)
  img = np.swapaxes(img, 1, 2)
  return img

def do_flip(data):
  for i in xrange(data.shape[0]):
    data[i,:,:] = np.fliplr(data[i,:,:])

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
      #print('aaaaaa')
      #print(out_prob[0,300:350,300:350])
      #print('bbbbbb')
      #print(_out_prob[0,300:350,300:350])
      out_prob += _out_prob
  return out_prob

def prob_to_out(prob):
  out_img = np.uint8(np.squeeze(prob.argmax(axis=0)))
  return out_img

def get_mask(img_path, cutoff, nets):
  img = get_img(img_path)
  if cutoff is None or cutoff<=0:
    prob = get_mask_prob(img, nets)
    return prob_to_out(prob)

  mask = np.zeros( (2,img.shape[1], img.shape[2]), dtype=np.float32 )
  step = 256
  step = cutoff
  for flip in [0,1]:
    for x in xrange(0, img.shape[1],step):
      xstart = x
      xstop = min(xstart+cutoff,img.shape[1])
      xstart = xstop-cutoff
      for y in xrange(0, img.shape[2], step):
        ystart = y
        ystop = min(ystart+cutoff,img.shape[2])
        ystart = ystop-cutoff
        #print(xstart,ystart,xstop,ystop)
        _img = img[:,xstart:xstop,ystart:ystop]
        if flip==1:
          do_flip(_img)
        _mask = get_mask_prob(_img, nets)
        if flip==1:
          do_flip(_mask)
          do_flip(_img)
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

def dice_coef(pred_label, label):

  r = 0.0
  pred_label_sum = np.sum(pred_label)
  label_sum = np.sum(label)
  if pred_label_sum==0 and label_sum==0:
    r = 1.0
  else:
    intersection = np.sum(pred_label * label)
    r = (2. * intersection) / (pred_label_sum + label_sum)
  return r

def main():
  #ensembles = [ ('./model/deeplab5', 12), ('./model/deeplab6', 11), ('./model/deeplab7', 9) ]
  ensembles = [ ('./model/deeplab-9-19/deeplab-1024', 96), ('./model/deeplab-9-19/deeplab-1152', 51) ]
  DATA_ROOT = '/raid5data/dplearn/carvn'
  parser = argparse.ArgumentParser(description='carvn submit')
  parser.add_argument('--gpu', type=int, default=7,
      help='gpu for use.')
  parser.add_argument('--cutoff', type=int, default=1280,
      help='cutoff size.')
  parser.add_argument('--parts', default='',
      help='test parts.')
  args = parser.parse_args()
  ctx = mx.gpu(args.gpu)
  nets = []
  for m in ensembles:
    net = edict()
    net.ctx = ctx
    print('loading', m[0], m[1])
    net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(m[0], m[1])
    net.arg_params, net.aux_params = ch_dev(net.arg_params, net.aux_params, net.ctx)
    nets.append(net)
  test_data_dir = os.path.join(DATA_ROOT, 'test_hq')
  suffix = '_'.join(args.parts.split(','))
  parts = {}
  for p in args.parts.split(','):
    if len(p)==0:
      continue
    parts[int(p)] = 1
  out_fn = 'submit.csv'
  if len(suffix)>0:
    out_fn = "submit_%s.csv" % suffix
  outf = open(out_fn, 'w')
  #outf.write("img,rle_mask\n")
  pp = 0
  for img in os.listdir(test_data_dir):
    #id = img.split('.')[0]
    id = img
    pos = int(img.split('.')[0].split('_')[1])
    if len(parts)>0 and not pos in parts:
      #print("skip %d"%pos)
      continue
    img = os.path.join(test_data_dir, img)
    #print(img)
    mask = get_mask(img, args.cutoff, nets)
    #print(mask.shape)
    #mask = Image.fromarray(mask)
    #mask.putpalette(pallete)
    #out_img = os.path.join('./mask_images', img)
    #out_img = out_img.replace("jpg", "png")
    #mask.save(out_img)
    rle = rle_encode(mask)
    rle_str = rle_to_string(rle)
    outf.write("%s,%s\n" % (id, rle_str))
    pp+=1
    if pp%1==0:
      prt("Processing %d"%pp)
  outf.close()

if __name__ == "__main__":
    main()



# pylint: skip-file
import numpy as np
import mxnet as mx
import argparse
import datetime
import cv2
import sys
import os
from PIL import Image

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

ctx = None

fcnxs, fcnxs_args, fcnxs_auxs = None, None, None

def get_img(img_path):
  """get the (1, 3, h, w) np.array data for the img_path"""
  mean = np.array([123.68, 116.779, 103.939])  # (R,G,B)
  img = Image.open(img_path)
  img = np.array(img, dtype=np.float32)
  reshaped_mean = mean.reshape(1, 1, 3)
  img = img - reshaped_mean
  #print(img.shape)
  #img = cv2.resize(img, (1280, 1280) )
  img = np.swapaxes(img, 0, 2)
  img = np.swapaxes(img, 1, 2)
  #img = np.expand_dims(img, axis=0)
  return img


def get_mask_out(data):
  global ctx, fcnxx, fcnxs_args, fcnxs_auxs

  if len(data.shape)==3:
    data = np.expand_dims(data, axis=0)
  fcnxs_args["data"] = mx.nd.array(data, ctx)
  data_shape = fcnxs_args["data"].shape
  #print(data_shape)
  label_shape = (1, data_shape[2]*data_shape[3])
  fcnxs_args["softmax_label"] = mx.nd.empty(label_shape, ctx)
  exector = fcnxs.bind(ctx, fcnxs_args ,args_grad=None, grad_req="null", aux_states=fcnxs_args)
  exector.forward(is_train=False)
  output = exector.outputs[0]
  #output = output.asnumpy()
  out_img = np.uint8(np.squeeze(output.asnumpy().argmax(axis=1)))
  return out_img

def get_mask(img_path, cutoff):
  img = get_img(img_path)
  if cutoff is None or cutoff<=0:
    return get_mask_out(img)

  mask = np.zeros( (img.shape[1], img.shape[2]), dtype=np.uint8 )
  #print(img.shape)
  #for x in [0, img.shape[1]-cutoff]:
  #  for y in [0, img.shape[2]-cutoff]:
  #    _img = img[:,x:(x+cutoff),y:(y+cutoff)]
  #    _mask = get_mask_out(_img)
  #    mask[x:(x+cutoff),y:(y+cutoff)] = _mask
  for x in xrange(0, img.shape[1], cutoff):
    xstart = x
    xstop = min(xstart+cutoff,img.shape[1])
    xstart = xstop-cutoff
    for y in xrange(0, img.shape[2], cutoff):
      ystart = y
      ystop = min(ystart+cutoff,img.shape[2])
      ystart = ystop-cutoff
      print(xstart,ystart,xstop,ystop)
      _img = img[:,xstart:xstop,ystart:ystop]
      _mask = get_mask_out(_img)
      mask[xstart:xstop,ystart:ystop] = _mask
  return mask

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
  global ctx, fcnxs, fcnxs_args, fcnxs_auxs
  DATA_ROOT = '/raid5data/dplearn/carvn'
  parser = argparse.ArgumentParser(description='carvn submit')
  parser.add_argument('--model-dir', default='./model',
      help='directory to save model.')
  parser.add_argument('--epoch', type=int, default=5,
      help='load epoch.')
  parser.add_argument('--gpu', type=int, default=0,
      help='gpu for use.')
  parser.add_argument('--cutoff', type=int, default=959,
      help='cutoff size.')
  parser.add_argument('--parts', default='',
      help='test parts.')
  args = parser.parse_args()
  ctx = mx.gpu(args.gpu)
  #ctx = mx.cpu()
  model_prefix = args.model_dir+"/FCN8s_VGG16"
  fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_prefix, args.epoch)
  fcnxs_args, fcnxs_auxs = ch_dev(fcnxs_args, fcnxs_auxs, ctx)
  test_data_dir = os.path.join(DATA_ROOT, 'test')
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
    out_img = os.path.join('./mask_images', img)
    out_img = out_img.replace("jpg", "png")
    img = os.path.join(test_data_dir, img)
    #print(img)
    mask = get_mask(img, args.cutoff)
    #print(mask.shape)
    #mask = Image.fromarray(mask)
    #mask.putpalette(pallete)
    #mask.save(out_img)
    rle = rle_encode(mask)
    rle_str = rle_to_string(rle)
    outf.write("%s,%s\n" % (id, rle_str))
    pp+=1
    if pp%10==0:
      prt("Processing %d"%pp)
    #count = 0
    #for i in xrange(mask.shape[0]):
    #  for j in xrange(mask.shape[1]):
    #    if mask[i][j]>0:
    #      count+=1
    #print(count)
    #rle = []
    #for w in xrange(mask.shape[1]):
    #  for h in xrange(mask.shape[0]):
    #    pixel = mask.shape[0]*w+h
    #    m = mask[h][w]
    #    if m==0:
    #      continue
    #    if len(rle)==0:
    #      rle.append( [pixel, 1] )
    #    else:
    #      if pixel==sum(rle[-1]):
    #        rle[-1][1]+=1
    #      else:
    #        rle.append( [pixel,1] )
    #line = id+","
    #for _rle in rle:
    #  line += str(_rle[0])+" "+str(_rle[1])+" "
    #line += "\n"
    #outf.write(line)
  outf.close()

if __name__ == "__main__":
    main()


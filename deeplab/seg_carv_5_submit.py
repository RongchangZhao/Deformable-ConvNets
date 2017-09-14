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

deeplab, deeplab_args, deeplab_auxs = None, None, None

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

'''
def get_mask_out(data):
    global ctx, deeplab, deeplab_args, deeplab_auxs

    if len(data.shape)==3:
        data = np.expand_dims(data, axis=0)
    deeplab_args["data"] = mx.nd.array(data, ctx)
    data_shape = deeplab_args["data"].shape
    #print(data_shape)
    label_shape = (1, data_shape[2]*data_shape[3])
    deeplab_args["softmax_label"] = mx.nd.empty(label_shape, ctx)
    exector = deeplab.bind(ctx, deeplab_args ,args_grad=None, grad_req="null", aux_states=deeplab_auxs)
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
'''

def get_mask_prob(data):
    global ctx, deeplab, deeplab_args, deeplab_auxs

    if len(data.shape)==3:
        data = np.expand_dims(data, axis=0)
    deeplab_args["data"] = mx.nd.array(data, ctx)
    data_shape = deeplab_args["data"].shape
    #print(data_shape)
    label_shape = (1, data_shape[2]*data_shape[3])
    deeplab_args["softmax_label"] = mx.nd.empty(label_shape, ctx)
    exector = deeplab.bind(ctx, deeplab_args ,args_grad=None, grad_req="null", aux_states=deeplab_auxs)
    exector.forward(is_train=False)
    output = exector.outputs[0]
    prob = output.asnumpy()
    #out_img = np.uint8(np.squeeze(prob.argmax(axis=1)))
    out_prob = np.squeeze(prob)
    #print(prob.shape, out_prob.shape)
    return out_prob

def prob_to_out(prob):
    out_img = np.uint8(np.squeeze(prob.argmax(axis=0)))
    return out_img

def get_mask(img_path, cutoff, resize, flip):
    img = get_img(img_path)
    if cutoff is None or cutoff<=0:
        prob = get_mask_prob(img)
        return prob_to_out(prob)
    if resize:
        img = cv2.resize(img, (img.shape[0]/2, img.shape[1]/2))

    mask = np.zeros( (2,img.shape[1], img.shape[2]), dtype=np.float32 )
    moves = [2,3] # moves in h,w
    print img.shape[1],img.shape[2],moves[0],moves[1]
    assert (img.shape[1]-cutoff)%(moves[0]-1)==0
    assert (img.shape[2]-cutoff)%(moves[1]-1)==0
    moves = [ (img.shape[1]-cutoff)/(moves[0]-1), (img.shape[2]-cutoff)/(moves[1]-1) ]
    for h in xrange(0,img.shape[1]-cutoff+1,max(moves[0],1)):
        for w in xrange(0, img.shape[2]-cutoff+1, moves[1]):
            _img = img[:,h:(h+cutoff),w:(w+cutoff)]
            #print(h,w)
            _mask = get_mask_prob(_img)
            mask[:,h:(h+cutoff),w:(w+cutoff)] += _mask
            if not flip:
                continue
            # Flip Once
            for idx in range(_img.shape[0]):
                _img[idx,:,:] = np.fliplr(_img[idx,:,:])
            # Gen
            _mask = get_mask_prob(_img)
            # Flip Twice
            for idx in range(_mask.shape[0]):
                _mask[idx,:,:] = np.fliplr(_mask[idx,:,:])
                #_img[idx,:,:] = np.fliplr(_img[idx,:,:])
            # Shared Memory Recovery
            for idx in range(_img.shape[0]):
                _img[idx,:,:] = np.fliplr(_img[idx,:,:])
            mask[:,h:(h+cutoff),w:(w+cutoff)] += _mask
            
    #for x in xrange(0, img.shape[1], cutoff):
    #  xstart = x
    #  xstop = min(xstart+cutoff,img.shape[1])
    #  xstart = xstop-cutoff
    #  for y in xrange(0, img.shape[2], cutoff):
    #    ystart = y
    #    ystop = min(ystart+cutoff,img.shape[2])
    #    ystart = ystop-cutoff
    #    #print(xstart,ystart,xstop,ystop)
    #    _img = img[:,xstart:xstop,ystart:ystop]
    #    _mask = get_mask_prob(_img)
    #    mask[:,xstart:xstop,ystart:ystop] += _mask
    if resize:
        mask = cv2.resize(mask, mask.shape[0]*2, mask.shape[1]*2 )
    
    return prob_to_out(mask)

def do_flip(data):
  for i in xrange(data.shape[0]):
    data[i,:,:] = np.fliplr(data[i,:,:])

def get_mask_step(img_path, cutoff, step, flip):
    img = get_img(img_path)
    if cutoff is None or cutoff<=0:
        prob = get_mask_prob(img)
        return prob_to_out(prob)
    if resize:
        img = cv2.resize(img, (img.shape[0]/2, img.shape[1]/2))

    mask = np.zeros( (2,img.shape[1], img.shape[2]), dtype=np.float32 )

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
        _mask = get_mask_prob(_img)
        mask[:,xstart:xstop,ystart:ystop] += _mask
        if not flip:
          continue
        do_flip(_img)
        _mask = get_mask_prob(_img)
        do_flip(_mask)
        mask[:,xstart:xstop,ystart:ystop] += _mask
        do_flip(_img)
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
    global ctx, deeplab, deeplab_args, deeplab_auxs
    DATA_ROOT = '/data1/deepinsight/carvn'
    parser = argparse.ArgumentParser(description='carvn submit')
    parser.add_argument('--model-dir', default='./',
      help='directory to save model.')
    parser.add_argument('--model', default='DeeplabV3-ResNeXt-152L64X1D4XP_997265',
      help='filename to savemodel.')
    parser.add_argument('--epoch', type=int, default=18,
      help='load epoch.')
    parser.add_argument('--gpu', type=int, default=0,
      help='gpu for use.')
    parser.add_argument('--cutoff', type=int, default=1280,
      help='cutoff size.')
    parser.add_argument('--step', type=int, default=256,
      help='step size.')
    parser.add_argument('--resize', type=int, default=0,
      help='resize size.')
    parser.add_argument('--flip', type=int, default=1,
      help='if flip.')
    parser.add_argument('--parts', default='',
      help='test parts.')
    args = parser.parse_args()
    ctx = mx.gpu(args.gpu)
    #ctx = mx.cpu()
    
    model_prefix = args.model_dir + args.model
    deeplab, deeplab_args, deeplab_auxs = mx.model.load_checkpoint(model_prefix, args.epoch)
    deeplab_args, deeplab_auxs = ch_dev(deeplab_args, deeplab_auxs, ctx)
    
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
        out_img = os.path.join('./mask_images', img)
        out_img = out_img.replace("jpg", "png")
        img = os.path.join(test_data_dir, img)
        #print(img)
        #mask = get_mask(img, args.cutoff, args.resize, args.flip)
        mask = get_mask_step(img, args.cutoff, args.step, args.flip)
        #print(mask.shape)
        #mask = Image.fromarray(mask)
        #mask.putpalette(pallete)
        #mask.save(out_img)
        rle = rle_encode(mask)
        rle_str = rle_to_string(rle)
        #print "%s,%s\n" % (id, rle_str)
        outf.write("%s,%s\n" % (id, rle_str))
        pp += 1
        if pp % 10 == 0:
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



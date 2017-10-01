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
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description="",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lst', type=str, default='./val.lst',
    help='')
parser.add_argument('--val-root-path', type=str, default='/data1/deepinsight/aichallenger/scene/ai_challenger_scene_validation_20170908/scene_validation_images_20170908')
parser.add_argument('--test-root-path', type=str, default='/data1/deepinsight/aichallenger/scene/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922')
parser.add_argument('--gpu', type=int, default=0,
    help='')
parser.add_argument('--num-classes', type=int, default=80,
    help='')
parser.add_argument('--batch-size', type=int, default=128,
    help='')
parser.add_argument('--mode', type=int, default=0,
    help='')
#parser.add_argument('--size', type=str, default='448,512')
parser.add_argument('--size', type=str, default='448,512')
parser.add_argument('--step', type=int, default=-24,
    help='if negative, use random crops')
#parser.add_argument('--model', type=str, default='./model/ft448deformsqex0.0001_9682,3|./model/sft320deformsqex_9692,1')
#parser.add_argument('--model', type=str, default='./model/sft320deformsqex_9692,1')
parser.add_argument('--model', type=str, default='sft448deformsqex00003,1|/data1/deepinsight/CAIScene/ft448deformsqex0.0001_9682,3') #
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

class DataLoader:
    def __init__(self, imgs, im_process):
        self.imgs = imgs
        self.im_process = im_process
        self.path_q = Queue()
        for i in xrange(len(imgs)):
            self.path_q.put(imgs[i])
        self.img_q = Queue(maxsize=4000)
        self.proc_img = [Thread(target=self.write_img) for i in range(5)]
        self.exit_flag = False
    #self.proc_path = [Thread(target=self.write_path) for i in range(1)]
    #self.batch_q = multiprocessing.Queue(maxsize=40)
    #self.proc_batch = [multiprocessing.Process(target=self.write_batch) for i in range(16)]

    def start(self):
        for proc in self.proc_img:
            proc.start()


    def write_img(self):
        while(True):
            try:
                img = self.path_q.get(block=False)
                im_path = img[2]
                im_id = img[0]
                im = self.im_process(im_path)
                self.path_q.task_done()
                self.img_q.put((im_id,img[1],im))
            except Exception, ex:
                print(ex)
                break
        print("writing thread exit..")

    def get(self):
        v = self.img_q.get()
        self.img_q.task_done()
        return v

    def close(self):
        print("left qsize %d" % self.path_q.qsize())
        print("left qimg %d" % self.img_q.qsize())
        self.exit_flag = True
        for proc in self.proc_img:
            proc.join()

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
ctx = mx.gpu(args.gpu)


nets = []
for model_str in args.model.split('|'):
    net = edict()
    prefix, epoch = model_str.split(',')
    epoch = int(epoch)
    print('loading',prefix, epoch)
    net.sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    net.arg_params, net.aux_params = ch_dev(arg_params, aux_params, ctx)
    nets.append(net)

imgs = []
i = 0
for line in open(args.lst, 'r'):
    vec = line.strip().split("\t")
    imgs.append( (i, int(vec[1]), os.path.join(args.root_path, vec[2])) )
    i+=1


X = np.zeros( (len(imgs), args.num_classes) , dtype=np.float32 )

num_batches = int( math.ceil(len(imgs) / args.batch_size) )
print("num_batches %d" % num_batches)
nrof_loops = args.step*-1

for loop in xrange(nrof_loops):
    print("loop %d start" % loop)
    #data_loader = DataLoader(imgs, image_preprocess)
    #data_loader.start()
  
    batch_head = 0
    batch_num = 0
    while batch_head<len(imgs):
        prt("processing batch %d" % batch_num)
        current_batch_sz = min(args.batch_size, len(imgs)-batch_head)
        input_blob = np.zeros((current_batch_sz,3,args.crop_size,args.crop_size))
        #print batch_head
        idx = 0
        ids = []
        for index in range(batch_head, batch_head+current_batch_sz):
            #img_name = imgs[index]
            #im_id = str(im_ids[index])
            #img_full_name = 'data/test2017/' + img_name
            raw_img = imgs[index]
            #img = data_loader.get()
            img = (raw_img[0], raw_img[1], image_preprocess(raw_img[2]))
            im_id = img[0]
            assert(im_id<X.shape[0])
            input_blob[idx,:,:,:] = img[2]
            ids.append(im_id)
            #cnt += 1
            idx += 1
            #print(idx)
  
  
        for net in nets:
            net.arg_params["data"] = mx.nd.array(input_blob, ctx)
            net.arg_params["softmax_label"] = mx.nd.empty((current_batch_sz,), ctx)
            exe = net.sym.bind(ctx, net.arg_params ,args_grad=None, grad_req="null", aux_states=net.aux_params)
            exe.forward(is_train=False)
            net_out = exe.outputs[0].asnumpy()
      
            for bz in xrange(current_batch_sz):
                probs = net_out[bz,:]
                score = np.squeeze(probs)
                im_id = ids[bz]
                X[im_id,:] += score
      
        batch_head += current_batch_sz
        batch_num += 1
  
    val(X, imgs)
    #data_loader.close()
  
  
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
  

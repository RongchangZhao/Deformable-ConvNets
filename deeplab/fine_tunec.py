import os
import sys
import argparse
import math
import random
import logging
logging.basicConfig(level=logging.DEBUG)
#from common import find_mxnet
import common.data_cotrain as data
#from common import fit
import mxnet as mx
import mxnet.optimizer as optimizer
import resnet_dcn
import numpy as np

import os, urllib

LABEL_WIDTH = 1

def get_fine_tune_model(symbol, arg_params, args):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    if args.size1==args.batch_size:
      fc1 = mx.sym.FullyConnected(data=symbol, num_hidden=80, name='fc1')
      out = mx.sym.SoftmaxOutput(data = fc1, name='softmax')
    else:
      args.per_size1 = int(args.size1/args.ctx_num)
      gt_label = mx.symbol.Variable('softmax_label')
      in1 = mx.symbol.slice_axis(symbol, axis=0, begin=0, end=args.per_size1)
      in2 = mx.symbol.slice_axis(symbol, axis=0, begin=args.per_size1, end=args.per_batch_size)
      gt1 = mx.symbol.slice_axis(gt_label, axis=0, begin=0, end=args.per_size1)
      gt2 = mx.symbol.slice_axis(gt_label, axis=0, begin=args.per_size1, end=args.per_batch_size)
      fc1 = mx.sym.FullyConnected(data=in1, num_hidden=80, name='fc1')
      fc2 = mx.sym.FullyConnected(data=in2, num_hidden=365, name='fc2')
      out1 = mx.sym.SoftmaxOutput(data = fc1, label=gt1, name='softmax1')
      out2 = mx.sym.SoftmaxOutput(data = fc2, label=gt2, grad_scale=args.scale2, name='softmax2')
      out = mx.symbol.Group([out1, out2])
    #layer_name = args.layer_before_fullc
    #all_layers = symbol.get_internals()
    #last_before = all_layers[layer_name+'_output']
    #lr_mult = 1
    #feature = last_before
    #fc = mx.symbol.FullyConnected(data=feature, num_hidden=args.num_classes, name='fc', lr_mult=lr_mult)
    #symbol = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    if args.retrain:
      new_args = arg_params
    else:
      new_args = dict({k:arg_params[k] for k in arg_params if (('fc' not in k))})
    return (out, new_args)

class JAccuracy(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(JAccuracy, self).__init__(
        'jaccuracy', axis=self.axis,
        output_names=None, label_names=None)
    self.loss = []

  def update(self, labels, preds):
    for label, pred_label in zip(labels, preds):
        pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        if len(label.shape)>1:
          label = mx.ndarray.argmax(label, axis=self.axis)
        #print(pred_label.shape, label.shape)
        pred_label = pred_label.asnumpy().astype('int32')
        label = label.asnumpy().astype('int32')
        if pred_label.shape[0]<label.shape[0]:
          label = label[0:pred_label.shape[0]]
        assert label.shape[0]==pred_label.shape[0]
        pred_label = pred_label.flatten()
        label = label.flatten()
        stat = [0,0]
        for i in xrange(len(label)):
          stat[1]+=1
          if label[i]==pred_label[i]:
            stat[0]+=1
        #print(label)
        #print(pred_label)
        #print(label.shape)

        #check_label_shapes(label, pred_label)

        #print('eval_stat', stat)
        self.sum_metric += stat[0]
        self.num_inst += stat[1]

class KAccuracy(mx.metric.EvalMetric):
    def __init__(self, top_k=5, name='k_accuracy',
                 output_names=None, label_names=None):
        super(KAccuracy, self).__init__(
            name, top_k=top_k,
            output_names=output_names, label_names=label_names)
        self.top_k = top_k
        assert(self.top_k > 1), 'Please use Accuracy if top_k is no more than 1'
        self.name += '_%d' % self.top_k

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        #check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            assert(len(pred_label.shape) <= 2), 'Predictions should be no more than 2 dims'
            pred_label = np.argsort(pred_label.asnumpy().astype('float32'), axis=1)
            if len(label.shape)>1:
              label = mx.ndarray.argmax(label, axis=1)
            label = label.asnumpy().astype('int32')
            if pred_label.shape[0]<label.shape[0]:
              label = label[0:pred_label.shape[0]]
            #print(label[0])
            #check_label_shapes(label, pred_label)
            num_samples = pred_label.shape[0]
            num_dims = len(pred_label.shape)
            if num_dims == 1:
                self.sum_metric += (pred_label.flat == label.flat).sum()
            elif num_dims == 2:
                num_classes = pred_label.shape[1]
                top_k = min(num_classes, self.top_k)
                for j in range(top_k):
                    self.sum_metric += (pred_label[:, num_classes - 1 - j].flat == label.flat).sum()
            self.num_inst += num_samples

if __name__ == "__main__":
    print('mxnet version', mx.__version__)
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--kv-store', type=str, default='device',
                       help='key-value store type')
    train.add_argument('--num-epochs', type=int, default=100,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.01,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str,
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--mom', type=float, default=0.0,
                       help='momentum for sgd')
    train.add_argument('--wd', type=float, default=0.0,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=256,
                       help='the batch size')
    train.add_argument('--size1', type=int, default=128,
                       help='the batch size')
    train.add_argument('--scale2', type=float, default=1.0,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='model prefix')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    train.add_argument('--load-epoch', type=int,
                       help='load the model on an epoch using the model-load-prefix')
    train.add_argument('--top-k', type=int, default=5,
                       help='report the top-k accuracy. 0 means no report.')
    data.add_data_args(parser)
    aug = data.add_data_aug_args(parser)
    parser.add_argument('--pretrained-model', type=str, default='model/resnet-152',
                        help='the pre-trained model')
    parser.add_argument('--pretrained-epoch', type=int, default=0,
                        help='the pre-trained model epoch to load')
    parser.add_argument('--layer-before-fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')
    parser.add_argument('--no-checkpoint', action="store_true",
                        help='do not save checkpoints')
    parser.add_argument('--retrain', action="store_true",
                        help='retrain')
    # use less augmentations for fine-tune
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(data_dir="/home/deepinsight/frankwang/Deformable-ConvNets/deeplab/runs_CAIScene", top_k=3, kv_store='device', data_nthreads=15)
    #parser.set_defaults(model_prefix="", data_nthreads=15, batch_size=64, num_classes=263, gpus='0,1,2,3')
    #parser.set_defaults(image_shape='3,320,320', num_epochs=32,
    #                    lr=.0001, lr_step_epochs='12,20,24,28', wd=0, mom=0.9, lr_factor=0.5)
    parser.set_defaults(image_shape='3,224,224', wd=0, mom=0)
    parser.set_defaults(image_root='/data1/deepinsight/aichallenger/scene')

    args = parser.parse_args()
    args.label_width = LABEL_WIDTH
    args.gpu_num = len(args.gpus.split(','))
    args.batch_per_gpu = args.batch_size/args.gpu_num
    print('gpu_num', args.gpu_num)
    args.ctx_num = args.gpu_num
    hw = int(args.image_shape.split(',')[1])
    #args.train_lst = os.path.join(args.data_dir, 'train.lst')
    #args.val_lst = os.path.join(args.data_dir, 'val.lst')
    #if args.retrain:
    args.train_lst = os.path.join(args.data_dir, 'trainval9.lst')
    args.train_lst2 = os.path.join(args.data_dir, 'place365_challenge.lst')
    args.val_lst = os.path.join(args.data_dir, 'val1.lst')

    kv = mx.kvstore.create(args.kv_store)
    train, val = data.get_rec_iter(args, kv)
    args.num_samples = train.num_samples()

    print('num_samples', args.num_samples)
    assert args.batch_size % args.ctx_num==0
    assert args.size1 % args.ctx_num==0
    args.per_batch_size = int(args.batch_size/args.ctx_num)

    print(args)

    # load pretrained model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    prefix = args.pretrained_model
    epoch = args.pretrained_epoch
    if args.retrain:
      #sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
      _,flat,_,sym = resnet_dcn.get_cls_symbol(args.num_classes, 152, args.image_shape, use_deformable=0, sqex=0)
      print('loading', prefix, epoch)
      _, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
      (sym, arg_params) = get_fine_tune_model(flat, arg_params, args)
    else:
      _,flat,_,sym = resnet_dcn.get_cls_symbol(args.num_classes, 152, args.image_shape, use_deformable=0, sqex=0)
      print('loading', prefix, epoch)
      _, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
      #resnet_dcn.init_weights(sym, {'data': (args.batch_size, 3, hw, hw), 'softmax_label': (args.batch_size,)}, arg_params, aux_params)
      resnet_dcn.init_weights(sym, {'data': (args.batch_size, 3, hw, hw), 'softmax_label': (args.batch_size, args.num_classes)}, arg_params, aux_params)
      (sym, arg_params) = get_fine_tune_model(flat, arg_params, args)

    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]
    model = mx.mod.Module(
        context       = devs,
        symbol        = sym,
        #label_names   = ['softmax_label', 'softmax2_label']
    )
    #optimizer_params = {
    #        'learning_rate': args.lr,
    #        'momentum' : args.mom,
    #        'wd' : args.wd}
    initializer = mx.init.Xavier(
        rnd_type='gaussian', factor_type="in", magnitude=2)
    metric = JAccuracy()
    eval_metrics = [mx.metric.create(metric)]
    #eval_metrics = ['accuracy']
    if args.top_k>0:
      metric = KAccuracy(top_k=args.top_k)
      eval_metrics.append(mx.metric.create(metric))
      #eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=args.top_k))



    opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd, rescale_grad=1.0/args.batch_size)
    #print(args.lr, args.mom, args.wd)
    #opt = optimizer.AdaGrad(learning_rate=base_lr, wd=base_wd, rescale_grad=1.0)
    _cb = mx.callback.Speedometer(args.batch_size, args.disp_batches)
    epoch_size = int(math.ceil(args.num_samples / args.size1))
    print('epoch_size', epoch_size)
    checkpoint = mx.callback.do_checkpoint(args.model_prefix)
    if args.no_checkpoint:
      checkpoint = None

    T_0 = epoch_size*2
    T_MULTI = 2
    lr_max = args.lr
    lr_min = 0.0

    global_step = [0]
    T_i = [T_0]
    T_curr = [0]
    #lr_pol = random.randint(0,1)
    lr_pol = 0
    lr_step_batches = {}
    for x in args.lr_step_epochs.split(','):
      y = int(x)*epoch_size
      print('lr step', y)
      lr_step_batches[y] = 1
    def _batch_callback(param):
      _cb(param)
      global_step[0]+=1
      mbatch = global_step[0]
      if args.retrain:
        if lr_pol==1:
          T_curr[0]+=1
          if T_curr[0]>T_i[0]:
            T_i[0] *= T_MULTI
            print('change T_i to',T_i[0])
            T_curr[0] = 0
          _lr = lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos(math.pi*T_curr[0]/T_i[0]))
        else:
          _lr = opt.lr
          if mbatch%(epoch_size*10)==0:
            _lr *= 0.3
      else:
        _lr = opt.lr
        if mbatch in lr_step_batches:
          _lr *= args.lr_factor
          print('lr change to', _lr)
      if mbatch%args.disp_batches==0:
        print(mbatch, _lr)
      opt.lr = _lr

    # run
    model.fit(train,
        begin_epoch        = 0,
        num_epoch          = args.num_epochs,
        eval_data          = val,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        optimizer          = opt,
        #optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        batch_end_callback = _batch_callback,
        epoch_end_callback = checkpoint,
        allow_missing      = True,
        monitor            = None)
    


# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zheng Zhang
# --------------------------------------------------------

#import _init_paths

import time
import argparse
import logging
import pprint
import os
import sys
from dice_metric import DiceMetric, AccMetric
#from config.config import config, update_config
from data import BatchFileIter
import shutil
import numpy as np
import mxnet as mx
import mxnet.optimizer as optimizer

#from symbols import resnet_v1_101_deeplab_dcn as network
import mxcommon.resnet_dcn as resnet_dcn
#from core import callback, metric
#from core.loader import TrainDataLoader
#from core.module import MutableModule
#from utils.load_data import load_gt_segdb, merge_segdb
#from utils.load_model import load_param
#from utils.PrefetchingIter import PrefetchingIter
#from utils.create_logger import create_logger
#from utils.lr_scheduler import WarmupMultiFactorScheduler
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_args():
  parser = argparse.ArgumentParser(description='Train deeplab network')
  # general
  parser.add_argument('--prefix', default='./model/deeplab',
      help='directory to save model.')
  parser.add_argument('--cutoff', type=int, default=800,
      help='cutoff size.')
  parser.add_argument('--end-epoch', type=int, default=30,
      help='training epoch size.')
  args = parser.parse_args()
  return args

#curr_path = os.path.abspath(os.path.dirname(__file__))
#sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))


def get_symbol(sym):
  use_dice = True
  croped_score = sym
  label = mx.symbol.Variable(name='softmax_label')
  if use_dice:
    softmax = mx.symbol.SoftmaxOutput(data = croped_score, label = label,  normalization='valid', multi_output = True, name='softmax')
    pred = mx.symbol.argmax(croped_score, axis=1)
    intersection = pred*label
    intersection = mx.symbol.sum(intersection, axis=[1,2])
    dice = intersection*2.0/(mx.symbol.sum(pred, axis=[1,2])+mx.symbol.sum(label,axis=[1,2]))
    dice_loss = mx.symbol.mean(dice)*-1.0
    dice_loss = mx.symbol.MakeLoss(dice_loss, grad_scale=1.0, name='dice_loss')
    out = mx.symbol.Group([softmax, dice_loss])
  else:
    out = mx.symbol.SoftmaxOutput(data = croped_score, label = label,  normalization='valid', multi_output = True, name='softmax')
  return out


def train_net(args):
    ctx = []
    gi = 0
    for i in xrange(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))):
      ctx.append(mx.gpu(gi))
      gi+=1
    #logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    #prefix = os.path.join(final_output_path, prefix)
    #image_shape = '3,800,800'
    #hw = int(image_shape.split(',')[1])
    prefix = args.prefix
    print('gpus:', len(ctx))
    hw = args.cutoff
    end_epoch = args.end_epoch
    pretrained = './model/resnet-152'
    epoch = 0
    image_shape = "%d,%d,%d" % (3, hw, hw)

    #net = network()
    # load symbol
    #shutil.copy2(os.path.join(curr_path, 'symbols', config.symbol + '.py'), final_output_path)
    #sym_instance = eval(config.symbol + '.' + config.symbol)()
    #sym = net.get_symbol(config, is_train=True)
    #sym = eval('get_' + args.network + '_train')(num_classes=config.dataset.NUM_CLASSES)
    sym = resnet_dcn.get_seg_score(2, 152, image_shape)
    sym = get_symbol(sym)

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = batch_size

    # print config
    #pprint.pprint(config)
    #logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # load dataset and prepare imdb for training
    #image_sets = [iset for iset in config.dataset.image_set.split('+')]
    #segdbs = [load_gt_segdb(config.dataset.dataset, image_set, config.dataset.root_path, config.dataset.dataset_path,
    #                        result_path=final_output_path, flip=config.TRAIN.FLIP)
    #          for image_set in image_sets]
    #segdb = merge_segdb(segdbs)

    # load training data
    #train_data = TrainDataLoader(sym, segdb, config, batch_size=input_batch_size, crop_height=config.TRAIN.CROP_HEIGHT, crop_width=config.TRAIN.CROP_WIDTH,
    #                             shuffle=config.TRAIN.SHUFFLE, ctx=ctx)
    train_dataiter = BatchFileIter(
        path_imglist         = "../data/train.lst",
        resize_size          = int(hw*1.125),
        cut_off_size         = hw,
        random_flip          = False,
        #rgb_mean             = (123.68, 116.779, 103.939),
        batch_size           = input_batch_size,
        )
    val_dataiter = BatchFileIter(
        path_imglist         = "../data/val.lst",
        resize_size          = hw,
        random_flip          = False,
        #cut_off_size         = hw,
        #rgb_mean             = (123.68, 116.779, 103.939),
        batch_size           = input_batch_size,
        )

    # infer max shape
    #max_scale = [(config.TRAIN.CROP_HEIGHT, config.TRAIN.CROP_WIDTH)]
    #max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3, max([v[0] for v in max_scale]), max([v[1] for v in max_scale])))]
    #max_label_shape = [('label', (config.TRAIN.BATCH_IMAGES, 1, max([v[0] for v in max_scale]), max([v[1] for v in max_scale])))]
    #max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape, max_label_shape)
    #print 'providing maximum shape', max_data_shape, max_label_shape

    ## infer shape
    #data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)
    #pprint.pprint(data_shape_dict)
    #sym_instance.infer_shape(data_shape_dict)

    # load and initialize params
    print pretrained
    _, arg_params, aux_params = mx.model.load_checkpoint(pretrained, epoch)
    #arg_params, aux_params = load_param(pretrained, epoch, convert=True)
    data_shape_dict = {'data': (input_batch_size, 3,hw, hw), 'softmax_label': (input_batch_size,hw, hw)}
    resnet_dcn.init_weights(sym, data_shape_dict, arg_params, aux_params)

    # check parameter shapes
    #sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

    # create solver
    #fixed_param_prefix = config.network.FIXED_PARAMS
    fixed_param_prefix = ['conv1', 'bn_conv1', 'res2', 'bn2', 'gamma', 'beta']
    #data_names = [k[0] for k in train_data.provide_data_single]
    #label_names = [k[0] for k in train_data.provide_label_single]

    #mod = MutableModule(sym, data_names=data_names, label_names=label_names,
    #                    logger=logger, context=ctx, max_data_shapes=[max_data_shape for _ in xrange(batch_size)],
    #                    max_label_shapes=[max_label_shape for _ in xrange(batch_size)], fixed_param_prefix=fixed_param_prefix)
    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
        #fixed_param_prefix = fixed_param_prefix,
    )

    # decide training params
    # metric
    #fcn_loss_metric = metric.FCNLogLossMetric(config.default.frequent * batch_size)
    #eval_metrics = mx.metric.CompositeEvalMetric()
    _dice = DiceMetric()
    _acc = AccMetric()
    eval_metrics = [mx.metric.create(_dice), mx.metric.create(_acc)]

    # rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric
    #for child_metric in [fcn_loss_metric]:
    #    eval_metrics.add(child_metric)

    # callback
    #batch_end_callback = callback.Speedometer(input_batch_size, frequent=args.frequent)
    #epoch_end_callback = mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True)

    # decide learning rate
    begin_epoch = 0
    base_lr = 0.02
    #lr_step = '10,20,30'
    train_size = 4848
    nrof_batch_in_epoch = int(train_size/input_batch_size)
    print('nrof_batch_in_epoch:', nrof_batch_in_epoch)
    #lr_factor = 0.1
    #lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    #lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    #lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    #lr_iters = [int(epoch * train_size / batch_size) for epoch in lr_epoch_diff]
    #print 'lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters

    #lr_scheduler = MultiFactorScheduler(lr_iters, lr_factor)

    # optimizer
    #optimizer_params = {'momentum': 0.9,
    #                    'wd': 0.0005,
    #                    'learning_rate': base_lr,
    #                    'rescale_grad': 1.0,
    #                    'clip_gradient': None}
    #initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
    opt = optimizer.SGD(learning_rate=base_lr, momentum=0.9, wd=0.0005, rescale_grad=(1.0/batch_size))
    _cb = mx.callback.Speedometer(input_batch_size, 10)


    def _batch_callback(param):
      mbatch = param.nbatch+1
      if mbatch % nrof_batch_in_epoch == 0:
        opt.lr *= 0.94
      #print(param.nbatch, opt.lr)
      _cb(param)
      if param.nbatch%10==0:
        print('lr-batch:',opt.lr,param.nbatch)
      sys.stdout.flush()
      sys.stderr.flush()

    #epoch_cb = mx.callback.do_checkpoint(prefix)
    epoch_cb = None

    #if not isinstance(train_data, PrefetchingIter):
    #    train_data = PrefetchingIter(train_data)

    #model.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
    #        batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
    #        optimizer='sgd', optimizer_params=optimizer_params,
    #        arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)
    model.fit(train_dataiter,
        begin_epoch        = 0,
        num_epoch          = end_epoch,
        eval_data          = val_dataiter,
        eval_metric        = eval_metrics,
        kvstore            = 'device',
        optimizer          = opt,
        #optimizer_params   = optimizer_params,
        #initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        #batch_end_callback = mx.callback.Speedometer(input_batch_size, 10),
        batch_end_callback = _batch_callback,
        epoch_end_callback = epoch_cb )

def main():
    args = parse_args()
    print 'Called with argument:', args
    train_net(args)

if __name__ == '__main__':
    main()


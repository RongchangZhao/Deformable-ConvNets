# pylint: skip-file
import sys, os
import argparse
import mxnet as mx
import numpy as np
import logging
import symbol_fcnxs
import init_fcnxs
from data import FileIter
from solver import Solver
from dice_metric import DiceMetric

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main():
    gpu_list = []
    _gpus = args.gpu.split(',')
    for _gpu in _gpus:
      _gpu = _gpu.strip()
      if len(_gpu)==0:
        continue
      gpu_list.append(int(_gpu))
    assert len(gpu_list)>0
    ctx = mx.gpu(gpu_list[0])
    carvn_root = ''
    num_classes = 2
    cutoff = None if args.cutoff==0 else args.cutoff
    epochs = [74,31,27,19]
    model_prefixes = ['VGG_FC_ILSVRC_16_layers', args.model_dir+"/FCN32s_VGG16", args.model_dir+"/FCN16s_VGG16", args.model_dir+"/FCN8s_VGG16"]
    if args.model == "fcn16s":
      fcnxs = symbol_fcnxs.get_fcn16s_symbol(numclass=num_classes, workspace_default=1536)
      fcnxs_model_prefix = model_prefixes[2]
      load_prefix = model_prefixes[1]
      lr = 1e-5
      run_epochs = epochs[2]
      load_epoch = epochs[1]
    elif args.model == "fcn8s":
      fcnxs = symbol_fcnxs.get_fcn8s_symbol(numclass=num_classes, workspace_default=1536)
      fcnxs_model_prefix = model_prefixes[3]
      load_prefix = model_prefixes[2]
      lr = 1e-6
      run_epochs = epochs[3]
      load_epoch = epochs[2]
    else:
      fcnxs = symbol_fcnxs.get_fcn32s_symbol(numclass=num_classes, workspace_default=1536)
      fcnxs_model_prefix = model_prefixes[1]
      load_prefix = model_prefixes[0]
      lr = 1e-4
      run_epochs = epochs[1]
      load_epoch = epochs[0]
    arg_names = fcnxs.list_arguments()
    print('loading', load_prefix, load_epoch)
    print('lr', lr)
    print('model_prefix', fcnxs_model_prefix)
    print('running epochs', run_epochs)
    _, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(load_prefix, load_epoch)
    if not args.retrain:
      if args.model == "fcn16s" or args.model == "fcn8s":
        fcnxs_args, fcnxs_auxs = init_fcnxs.init_from_fcnxs(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
      else:
        fcnxs_args, fcnxs_auxs = init_fcnxs.init_from_vgg16(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
    train_dataiter = FileIter(
        root_dir             = carvn_root,
        flist_name           = "../data/train.lst",
        cut_off_size         = cutoff,
        rgb_mean             = (123.68, 116.779, 103.939),
        )
    val_dataiter = FileIter(
        root_dir             = carvn_root,
        flist_name           = "../data/val.lst",
        cut_off_size         = cutoff,
        rgb_mean             = (123.68, 116.779, 103.939),
        )
    model = Solver(
        ctx                 = ctx,
        symbol              = fcnxs,
        begin_epoch         = 0,
        num_epoch           = run_epochs,
        arg_params          = fcnxs_args,
        aux_params          = fcnxs_auxs,
        learning_rate       = lr,
        momentum            = 0.99,
        wd                  = 0.0005)
    _metric = DiceMetric()
    model.fit(
        train_data          = train_dataiter,
        eval_data           = val_dataiter,
        eval_metric         = _metric,
        batch_end_callback  = mx.callback.Speedometer(1, 10),
        epoch_end_callback  = mx.callback.do_checkpoint(fcnxs_model_prefix))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert vgg16 model to vgg16fc model.')
    parser.add_argument('--model', default='fcnxs',
        help='The type of fcn-xs model, e.g. fcnxs, fcn16s, fcn8s.')
    parser.add_argument('--model-dir', default='./model',
        help='directory to save model.')
    parser.add_argument('--cutoff', type=int, default=800,
        help='cutoff size.')
    parser.add_argument('--gpu', default='',
        help='gpu for use.')
    parser.add_argument('--retrain', action='store_true', default=False,
        help='true means continue training.')
    args = parser.parse_args()
    logging.info(args)
    main()

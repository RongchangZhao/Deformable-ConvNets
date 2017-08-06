import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx
from symbols.irnext_v2_deeplab_v3_dcn_w_hypers import *

'''
def download_cifar10():
    data_dir="data"
    fnames = (os.path.join(data_dir, "cifar10_train.rec"),
              os.path.join(data_dir, "cifar10_val.rec"))
    download_file('http://data.mxnet.io/data/cifar10/cifar10_val.rec', fnames[1])
    download_file('http://data.mxnet.io/data/cifar10/cifar10_train.rec', fnames[0])
    return fnames
'''

if __name__ == '__main__':
    # download data
    # (train_fname, val_fname) = download_cifar10()

    # parse args
    parser = argparse.ArgumentParser(description="train cifar100",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(
        # network
        network        = 'irnext',
        num_layers       = 29,
        outfeature       = 3072,
        bottle_neck      = 1,
        expansion        = 0.25, 
        num_group        = 96,
        dilpat           = '', 
        irv2             = False, 
        deform           = 0, 
        
        # data
        data_train     = '/data3/dataset/CIFAR100_MXNet/train.rec',
        data_val       = '/data3/dataset/CIFAR100_MXNet/test.rec',
        num_classes    = 100,
        num_examples   = 50000,
        image_shape    = '3,28,28',
        pad_size       = 4,
        lastout        = 8,
        # train
        batch_size     = 128,
        num_epochs     = 400,
        lr             = .05,
        lr_step_epochs = '150,240',
        wd             = 0.0010
    )
    args = parser.parse_args()

    # load network
    #from importlib import import_module
    #net = import_module('symbols.'+args.network)
    #sym = net.get_symbol(**vars(args))

    net = irnext_deeplab_dcn(**vars(args))
    sym = net.get_cls_symbol()


    
    # train
    fit.fit(args, sym, data.get_rec_iter)


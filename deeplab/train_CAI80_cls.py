import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx
from symbols.irnext_v2_deeplab_v3_dcn_w_hypers import *


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train imagenet1k",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    # use a large aug level
    data.set_data_aug_level(parser, 3)
    parser.set_defaults(
        # network
        network          = 'irnext',
        num_layers       = 101,
        outfeature       = 1024,
        bottle_neck      = 1,
        expansion        = 0.5, 
        num_group        = 1,
        dilpat           = '', 
        irv2             = False, 
        deform           = 1,
        sqex             = 1,
        ratt             = 0,
        # data
        num_classes      = 80,
        num_examples     = 1281167,
        image_shape      = '3,224,224',
        lastout          = 7,
        min_random_scale = 1.0 , # if input image has min size k, suggest to use
                              # 256.0/x, e.g. 0.533 for 480
        # train
        num_epochs       = 80,
        lr_step_epochs   = '150,240,400',
        dtype            = 'float32'
    )
    args = parser.parse_args()

    # load network
    #from importlib import import_module
    #net = import_module('symbols.'+args.network)
    # sym = net.get_symbol(**vars(args))
    
    net = irnext_deeplab_dcn(**vars(args))
    sym = net.get_cls_symbol()

    # train
    fit.fit(args, sym, data.get_rec_iter)


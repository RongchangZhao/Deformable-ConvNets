import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

import runs_CAIScene.scene_init_from_cls

from symbols.inceptions import *


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train imagenet1k",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    
    # use a large aug level
    data.set_data_aug_level(parser, 1)
    
    
    
    parser.set_defaults(
        
        # network
        # irv2
        
        basefilter=16,
        num_group=1,
        num_group_11=1,
        scale=1.0,
        units=[10,20,9],
        
        #V4
        #basefilter = 32,
        #num_group = 1,
        #num_group_11 = 1,
        #units = [4,7,3],
        
        #V3
        # basefilter = 16,
        # num_group = 1,
        # num_group_11 = 1,
        
        # data
        num_classes = 80,
        num_examples = 480*365, # 53878
        image_shape = '3,299,299',
        lastout = 8,
        batch_size = 256,
        
        # train
        num_epochs       = 45,
        lr               = 0.1,
        lr_step_epochs   = '24,32,40',
        dtype            = 'float32',
        retrain          = 1,
        
        
        # load , please tune
        load_ft_epoch       = 41,
        model_ft_prefix     = 'irv2_299'

    )
    
    '''
    parser.set_defaults(
        # network
        network          = 'irnext',
        num_layers       = 50,
        outfeature       = 2048,
        bottle_neck      = 1,
        expansion        = 4, 
        num_group        = 1,
        dilpat           = '',#'DEEPLAB.HEAD', 
        irv2             = False, 
        deform           = 0,
        sqex             = 0,
        ratt             = 0,
        block567         = 0,
        lmar             = 0,
        lmarbeta         = 1000,
        lmarbetamin      = 0,
        lmarscale        = 0.9997,
        # data
        num_classes      = 80,
        num_examples     = 53878,
        image_shape      = '3,224,224',
        lastout          = 7,
        min_random_scale = 1.0 , # if input image has min size k, suggest to use
                              # 256.0/x, e.g. 0.533 for 480
        # train
        num_epochs       = 70,
        lr               = 0.003,
        lr_step_epochs   = '25,40',
        dtype            = 'float32',
        
        # load , please tune
        load_ft_epoch       = 21,
        model_ft_prefix     = '/data1/deepinsight/CAIScene/50ft320nude0003_9656'
        
    )
    '''
    args = parser.parse_args()

    
    
    
    # load network
    #from importlib import import_module
    #net = import_module('symbols.'+args.network)
    # sym = net.get_symbol(**vars(args))
    
    sym = get_symbol_irv2(**vars(args))
    # sym = get_symbol_V4(**vars(args))
    # sym = get_symbol_V3(**vars(args))

    # Init Parameters
    ctx = mx.cpu()
    
    if args.retrain == 1: # 1 From Scratch 0 FT
        fit.fit(args, sym, data.get_rec_iter)
    else:
        args.lr_step_epochs = '5,10'
        _ , deeplab_args, deeplab_auxs = mx.model.load_checkpoint(args.model_ft_prefix, args.load_ft_epoch)
        fit.fit(args, sym, data.get_rec_iter, arg_params=deeplab_args,aux_params=deeplab_auxs)
        
        
    # train
    
    #args.arg_params         = deeplab_args
    #args.aux_params         = deeplab_auxs
    



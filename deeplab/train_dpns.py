import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

import runs_CAIScene.scene_init_from_cls

from symbols.dpns import *


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
        # DPN92
        
        # data
        num_classes = 80,
        num_examples = 53878, # 53878
        image_shape = '3,320,320',
        lastout = 10,
        batch_size = 160,
        
        # train
        num_epochs       = 100,
        lr               = 0.1,
        lr_step_epochs   = '24,40,56',
        dtype            = 'float32',
        retrain          = 0,
        
        
        # load , please tune
        load_ft_epoch       = 0,
        model_ft_prefix     = '/data1/deepinsight/CAIScene/dpn92_places365-standard_2017_05_13/dpn92-365std'

    )
    args = parser.parse_args()

    
    
    
    # load network
    #from importlib import import_module
    #net = import_module('symbols.'+args.network)
    # sym = net.get_symbol(**vars(args))
    
    sym = get_symbol(**vars(args))
    # sym = get_symbol_V4(**vars(args))
    # sym = get_symbol_V3(**vars(args))

    # Init Parameters
    ctx = mx.cpu()
    
    if args.retrain == 1: # 1 From Scratch 0 FT
        fit.fit(args, sym, data.get_rec_iter)
    else:
        args.lr_step_epochs = '24,32,40'
        _ , deeplab_args, deeplab_auxs = mx.model.load_checkpoint(args.model_ft_prefix, args.load_ft_epoch)
        data_shape_dict = {'data': (args.batch_size, 3, 320, 320), 
                       'softmax_label': (args.batch_size,)}
        if args.model_ft_prefix[0] == '/':
            deeplab_args, deeplab_auxs = runs_CAIScene.scene_init_from_cls.init_from_irnext_cls(ctx, \
                            sym, deeplab_args, deeplab_auxs, data_shape_dict)
        else:
            args.lr_step_epochs = '20,50'
            
        fit.fit(args, sym, data.get_rec_iter, arg_params=deeplab_args,aux_params=deeplab_auxs)
        
        
    # train
    
    #args.arg_params         = deeplab_args
    #args.aux_params         = deeplab_auxs
    



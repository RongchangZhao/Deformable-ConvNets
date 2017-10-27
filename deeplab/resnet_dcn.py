# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
import numpy as np
import math
from mxnet.base import _Null


SEG_OUTPUT_STRIDE = 8

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False, dilate=(1,1), use_deformable=0, sqex=0):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        if not use_deformable:
          conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), num_group=1, kernel=(3,3), stride=stride, pad=dilate,
                                     dilate = dilate, no_bias=True, workspace=workspace, name=name + '_conv2')
        else:
          conv2_offset_weight = mx.symbol.Variable(name+'_conv2_offset_weight', lr_mult=0.1)
          conv2_offset_bias = mx.symbol.Variable(name+'_conv2_offset_bias', lr_mult=0.2)
          conv2_offset = mx.symbol.Convolution(name=name+'_conv2_offset', data = act2,
                                                        num_filter=18, pad=(1, 1), kernel=(3, 3), stride=stride,
                                                        weight=conv2_offset_weight, bias=conv2_offset_bias)
          conv2 = mx.contrib.symbol.DeformableConvolution(name=name+"_conv2", data=act2, offset=conv2_offset,
                                        num_filter=int(num_filter*0.25), pad=dilate, kernel=(3, 3), num_deformable_group=1,
                                        stride=stride, dilate=dilate, no_bias=True)
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if sqex:
          pool_se = mx.symbol.Pooling(data=conv3, cudnn_off=True, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_conv3_pool1')
          flat = mx.symbol.Flatten(data=pool_se)
          se_fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=int(num_filter/16), name=name + '_conv3_fc1', attr={'lr_mult': '0.25'} )
          #se_fc1_relu = mx.symbol.Activation(data=se_fc1, act_type='prelu', name= name+'_conv3_fc1_relu')
          se_fc1_relu = mx.symbol.LeakyReLU(data=se_fc1, act_type='prelu', name= name+'_conv3_fc1_relu')
          se_fc2 = mx.symbol.FullyConnected(data=se_fc1_relu, num_hidden=int(num_filter), name=name + '_conv3_fc2',attr={'lr_mult': '0.25'} )
          se_act = mx.sym.Activation(se_fc2, act_type="sigmoid", name=name+'_conv3_fc2_act')
          se_reshape = mx.symbol.Reshape(se_act, shape=(-1, num_filter, 1, 1), name=name+"conv3_reshape") 
          conv3 = mx.sym.broadcast_mul(conv3, se_reshape)
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def resnet_conv(data, units, num_stages, filter_list, num_classes, image_shape, bottle_neck=True, bn_mom=0.9, workspace=256, dtype='float32', is_seg = False, **kwargs):
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    dtype : str
        Precision (float32 or float16)
    """
    memonger = False
    num_unit = len(units)
    assert(num_unit == num_stages)
    if dtype == 'float32':
        data = mx.sym.identity(data=data, name='id')
    else:
        if dtype == 'float16':
            data = mx.sym.Cast(data=data, dtype=np.float16)
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    if height <= 32:            # such as cifar10
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    else:                       # often expected to be 224 such as imagenet
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    bodies = []
    use_deformable = kwargs.get('use_deformable', 0)
    for i in range(num_stages):
        if not is_seg:
          _dilate = 1
          stride = 1 if i==0 else 2
          _use_deformable = 0
          if use_deformable and i>=3:
            _use_deformable = use_deformable
            #_dilate = 2
        else:
          ossp = 2 if SEG_OUTPUT_STRIDE==8 else 3
          _use_deformable = 0
          if i<ossp:
            _dilate = 1
            stride = 1 if i==0 else 2
          else:
            _dilate = int(math.pow(2, i-ossp+1))
            if i>2 and use_deformable:
              _use_deformable = use_deformable
            stride = 1

        #_use_deformable = False

        #_dilate= _Null
        #_use_deformable = False
        #stride = 1
        #if i!=0:
        #  stride = 2
        #multi_grid = [1,2,1]
        multi_grid = [1,1,1]
        if not is_seg:
          multi_grid = [1,1,1]
        body = residual_unit(body, filter_list[i+1], (stride, stride), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger, dilate=(_dilate*multi_grid[0],_dilate*multi_grid[0]), use_deformable = _use_deformable, sqex=kwargs.get('sqex', 0))
        for j in range(units[i]-1):
            mg_index = (j+1)%len(multi_grid)
            mg = multi_grid[mg_index]
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger, dilate=(_dilate*mg,_dilate*mg), use_deformable=_use_deformable, sqex=kwargs.get('sqex',0))
        if is_seg and i>=ossp:
          bodies.append(body)
    if not is_seg:
      return body
    else:
      return body
      #return bodies

def get_cls_symbol(num_classes, num_layers, image_shape, conv_workspace=256, dtype='float32', **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    data = mx.sym.Variable(name='data')
    body = resnet_conv(data = data,
                  units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  image_shape = image_shape,
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace,
                  dtype       = dtype,
                  is_seg = False,
                  use_deformable=kwargs.get('use_deformable',0),
                  sqex=kwargs.get('sqex',0))
    bn_mom = 0.9
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    if dtype == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
    return relu1, flat, fc1, mx.sym.SoftmaxOutput(data=fc1, name='softmax')



def get_seg_score(num_classes, num_layers, image_shape, conv_workspace=256, dtype='float32', ignore_label = None, **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 92:
            units = [3, 4, 11, 3, 3, 3, 3]
            num_stages += 3
            filter_list += [2048]*3
        elif num_layers == 52:
            units = [3, 3, 3, 3, 3, 3, 3]
            num_stages  = len(units)
            filter_list += [2048]*3
        elif num_layers == 78:
            #units = [3, 8, 11, 3, 1,1,1]
            #num_stages += 3
            #filter_list += [2048]*3
            units = [3, 8, 11, 3, 3]
            num_stages += 1
            filter_list += [2048]*1
        elif num_layers == 72:
            units = [3, 6, 11, 3, 1,1,1]
            num_stages += 3
            filter_list += [2048]*3
        elif num_layers == 122:
            units = [3, 8, 17, 3, 3, 3, 3]
            num_stages += 3
            filter_list += [2048]*3
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    data = mx.sym.Variable(name='data')
    body = resnet_conv(data = data,
                  units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  image_shape = image_shape,
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace,
                  dtype       = dtype,
                  is_seg = True,
                  use_deformable=kwargs.get('use_deformable',0),
                  sqex=kwargs.get('sqex',0))

    #fc6_pre_bias = mx.symbol.Variable('fc6_pre_bias', lr_mult=2.0)
    #fc6_pre_weight = mx.symbol.Variable('fc6_pre_weight', lr_mult=1.0)
    #fc6 = mx.symbol.Convolution(data=body, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="fc6_pre",
    #                            bias=fc6_pre_bias, weight=fc6_pre_weight, workspace=conv_workspace)
    #relu_fc6 = mx.sym.Activation(data=fc6, act_type='relu', name='relu_fc6')
    if isinstance(body, list):
      relu_fc6 = mx.symbol.concat(*body, dim=1)
    else:
      relu_fc6 = body

    scores = []
    fc6_filters = num_classes
    #fc6_filters = 256
    for i in xrange(1):
      _bias = mx.symbol.Variable("fc6_%d_segscore_bias" % i, lr_mult=2.0)
      _weight = mx.symbol.Variable("fc6_%d_segscore_weight" % i, lr_mult=1.0)
      if i==0:
        _score = mx.symbol.Convolution(data=relu_fc6, kernel=(1, 1), pad=(0, 0), num_filter=fc6_filters, 
            name="fc6_%d_segscore" % i, bias = _bias, weight = _weight, workspace=conv_workspace)
      else:
        pad = (i*6, i*6)
        dilate = pad
        _score = mx.symbol.Convolution(data=relu_fc6, kernel=(3, 3), pad=pad, dilate=dilate, num_filter=fc6_filters, 
            name="fc6_%d_segscore" % i, bias = _bias, weight = _weight, workspace=conv_workspace)
      scores.append(_score)
    #gpool = mx.symbol.Pooling(data=relu_fc6, global_pool=True, kernel=(7, 7), pool_type='avg', name='fc6_global_pool')
    #_bias = mx.symbol.Variable("fc6_gpool_segscore_bias" , lr_mult=2.0)
    #_weight = mx.symbol.Variable("fc6_gpool_segscore_weight" , lr_mult=1.0)
    #gpool = mx.symbol.Convolution(data=gpool, kernel=(1, 1), pad=(0,0), num_filter=fc6_filters, 
    #    name="fc6_gpool_segscore", bias = _bias, weight = _weight, workspace=conv_workspace)
    #de_kernel = 128
    #gpool = mx.symbol.Deconvolution(data=gpool, num_filter=fc6_filters, kernel=(de_kernel, de_kernel), 
    #    stride=(de_kernel, de_kernel),
    #    num_group=fc6_filters, no_bias=True, name='upsampling_gpool',
    #    attr={'lr_mult': '0.0'}, workspace=conv_workspace)
    #scores.append(gpool)


    score = mx.symbol.concat(*scores, dim=1, name='fc6_score')

    upsampling = mx.symbol.Deconvolution(data=score, num_filter=num_classes, kernel=(SEG_OUTPUT_STRIDE*2, SEG_OUTPUT_STRIDE*2), 
        stride=(SEG_OUTPUT_STRIDE, SEG_OUTPUT_STRIDE),
        num_group=num_classes, no_bias=True, name='upsampling',
        attr={'lr_mult': '0.1'}, workspace=conv_workspace)

    croped_score = mx.symbol.Crop(*[upsampling, data], offset=(SEG_OUTPUT_STRIDE/2, SEG_OUTPUT_STRIDE/2), name='croped_score')

    #_bias = mx.symbol.Variable("fc6_data_segscore_bias", lr_mult=2.0)
    #_weight = mx.symbol.Variable("fc6_data_segscore_weight", lr_mult=1.0)
    #data_to_score = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=num_classes, 
    #    name="fc6_data_segscore", bias = _bias, weight = _weight, workspace=conv_workspace)
    #croped_score = croped_score + data_to_score
    #croped_score2 = mx.symbol.concat(croped_score, data_to_score, dim=1)
    #_bias = mx.symbol.Variable("fc6_x_segscore_bias", lr_mult=2.0)
    #_weight = mx.symbol.Variable("fc6_x_segscore_weight", lr_mult=1.0)
    #x_score = mx.symbol.Convolution(data=data, kernel=(1, 1), pad=(0, 0), num_filter=num_classes, 
    #    name="fc6_x_segscore", bias = _bias, weight = _weight, workspace=conv_workspace)
    #croped_score = x_score

    return croped_score

def get_seg_symbol(num_classes, num_layers, image_shape, conv_workspace=256, dtype='float32', ignore_label = None, **kwargs):
    croped_score = get_seg_score(num_classes, num_layers, image_shape, conv_workspace, dtype, ignore_label, **kwargs)
    use_ignore, ignore_label = (True, ignore_label) if ignore_label is not None else (False, -1)
    #data = mx.symbol.Variable(name="data")
    seg_cls_gt = mx.symbol.Variable(name='softmax_label')
    softmax = mx.symbol.SoftmaxOutput(data=croped_score, label=seg_cls_gt, normalization='valid', multi_output=True,
                                      use_ignore=use_ignore, ignore_label=ignore_label, name="softmax")
    return softmax

def init_weights(sym, data_shape_dict, arg_params, aux_params):
    arg_name = sym.list_arguments()
    aux_name = sym.list_auxiliary_states()
    arg_shape, aaa, aux_shape = sym.infer_shape(**data_shape_dict)
    #print(data_shape_dict)
    #print(arg_name)
    #print(arg_shape)
    #print(aaa)
    #print(aux_shape)
    arg_shape_dict = dict(zip(arg_name, arg_shape))
    aux_shape_dict = dict(zip(aux_name, aux_shape))
    #print(aux_shape)
    #print(aux_params)
    #print(arg_shape_dict)
    for k,v in arg_shape_dict.iteritems():
      #print(k,v)
      if k.endswith('offset_weight') or k.endswith('offset_bias'):
        print('initializing',k)
        arg_params[k] = mx.nd.zeros(shape = v)
      elif k.startswith('fc6_'):
        if k.endswith('_weight'):
          print('initializing',k)
          arg_params[k] = mx.random.normal(0, 0.01, shape=v)
        elif k.endswith('_bias'):
          print('initializing',k)
          arg_params[k] = mx.nd.zeros(shape=v)
        #pass
      elif k.startswith('upsampling') and k.endswith("_weight"):
        print('initializing',k)
        arg_params[k] = mx.nd.zeros(shape=v)
        init = mx.init.Initializer()
        init._init_bilinear(k, arg_params[k])
      else:
        if k.startswith('stage'):
          stage_id = int(k[5])
          if stage_id>4:
            rk = "stage4"+k[6:]
            if rk in arg_params:
              print('initializing', k, rk)
              if arg_shape_dict[rk]==v:
                arg_params[k] = arg_params[rk].copy()
              else:
                if k.endswith('_beta'):
                  arg_params[k] = mx.nd.zeros(shape=v)
                elif k.endswith('_gamma'):
                  arg_params[k] = mx.nd.random_uniform(shape=v)
                else:
                  arg_params[k] = mx.random.normal(0, 0.01, shape=v)
    for k,v in aux_shape_dict.iteritems():
        if k.startswith('stage'):
          stage_id = int(k[5])
          if stage_id>4:
            rk = "stage4"+k[6:]
            if rk in aux_params:
              print('initializing aux', k, rk)
              if aux_shape_dict[rk]==v:
                aux_params[k] = aux_params[rk].copy()
              else:
                if k.endswith('_moving_var'):
                  aux_params[k] = mx.nd.zeros(shape=v)
                elif k.endswith('_moving_mean'):
                  aux_params[k] = mx.nd.ones(shape=v)


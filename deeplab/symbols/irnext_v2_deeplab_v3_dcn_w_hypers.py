# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zheng Zhang
# --------------------------------------------------------

# --------------------------------------------------------

# Modified By DeepInsight

#  0. Todo: Make Code Tidier (with exec)
#  1. Todo: ResNeXt v2 : Grouping + Pre-Activation
#  2. Todo: IRB: First Block of Inception-ResNet , with Grouping
#  3. Todo: DeepLab v3, with multiple dilation pattern options
#  4. Todo: Dual Path Network ( DenseNet + ResNeXt Nx4d)

# --------------------------------------------------------


import cPickle
import mxnet as mx
# from utils.symbol import Symbol


###### UNIT LIST #######

# Todo 1,2,4

def irnext_unit(data, num_filter, stride, dim_match, name, bottle_neck=1, expansion=0.5, \
                 num_group=32, dilation=1, irv2 = False, deform=1, sqex=1, ratt=0, bn_mom=0.9, unitbatchnorm=True, workspace=256, memonger=False):
    
    """
    Return Unit symbol for building ResNeXt/simplified Xception block
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    stride : int
        Number of stride, 2 when block-crossing with downsampling or simple downsampling, else 1.
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name: str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    bottle_neck : int = 0,1,2,3
        If 0: Use conventional Conv3,3-Conv3,3
        If 1: Use ResNeXt Conv1,1-Conv3,3-Conv1,1
        If 2: Use IRB use Conv1,1-[Conv3,3;Conv3,3-Conv3,3]-Conv1,1
        If 3: Use Dual-Path-Net
    irv2: Boolean 
        if True: IRB use pre-activation
        if False: IRB do not use pre-activation
        
    expansion : float
        ResNet use 4, ResNeXt use 2, DenseNet use 0.25
    num_group: int
        Feasible Range: 4,8,16,32,64
    dilation: int
        a.k.a Atrous Convolution
    deform: 
        Deformable Conv Net
    """
    
    ## If 0: Use conventional Conv3,3-Conv3,3
    if bottle_neck == 0 :
        
        if unitbatchnorm:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        else:
            act1 = mx.sym.Activation(data=data, act_type='relu', name=name + '_relu1')
            
        
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, 
                                   pad=(dilation,dilation), dilate=(dilation,dilation), 
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        
        if unitbatchnorm:
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        else:
            act2 = mx.sym.Activation(data=conv1, act_type='relu', name=name + '_relu2')
            
        
        if deform == 0:
            conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1),
                                   pad=(dilation,dilation), dilate=(dilation,dilation),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        else:
            
            offsetweight = mx.symbol.Variable(name+'_offset_weight', lr_mult=1.0)
            offsetbias = mx.symbol.Variable(name+'_offset_bias', lr_mult=2.0)
            offset = mx.symbol.Convolution(name=name+'_offset', data = act2,
                                                      num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                                      weight=offsetweight, bias=offsetbias)
            
            
            conv2 = mx.contrib.symbol.DeformableConvolution(data=act2, offset=offset,
                     num_filter=num_filter, pad=(dilation,dilation), kernel=(3, 3), num_deformable_group=1,
                     stride=(1, 1), dilate=(dilation,dilation), no_bias=True)
        

        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,  workspace=workspace, name=name+'_sc')
            
            
        if memonger:
            shortcut._set_attr(mirror_stage='True')
            
        return conv2 + shortcut
        
    
    # If 1: Use ResNeXt Conv1,1-Conv3,3-Conv1,1 
    elif bottle_neck == 1:
        
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter/expansion), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        
        '''
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter/expansion), 
                                   num_group=num_group, kernel=(3,3), stride=stride, 
                                   pad=(dilation,dilation), dilate=(dilation,dilation),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        '''
        
        if deform == 0:
            conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter/expansion),
                                       num_group=num_group, kernel=(3,3), stride=stride,
                                       pad=(dilation,dilation), dilate=(dilation,dilation),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        else:
            
            offsetweight = mx.symbol.Variable(name+'_offset_weight', lr_mult=1.0)
            offsetbias = mx.symbol.Variable(name+'_offset_bias', lr_mult=2.0)
            offset = mx.symbol.Convolution(name=name+'_offset', data = act2,
                                                      num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                                      weight=offsetweight, bias=offsetbias)
            
            
            conv2 = mx.contrib.symbol.DeformableConvolution(name=name+'_deform', data=act2, offset=offset,
                     num_filter=int(num_filter/expansion), 
                     pad=(dilation,dilation), kernel=(3, 3), num_deformable_group=num_group,
                     stride=stride, dilate=(dilation,dilation), no_bias=True)
        
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,workspace=workspace, name=name + '_conv3')
        
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True, workspace=workspace, name=name+'_sc')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
        # out =  conv3 + shortcut
    
    
        if sqex == 0:
            out = conv3
        else:
            pool_se = mx.symbol.Pooling(data=conv3, cudnn_off=True, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_se_pool')
            flat = mx.symbol.Flatten(data=pool_se)
            se_fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=(num_filter/expansion/4), name=name + '_se_fc1',
                                             attr={'lr_mult': '0.25'} ) #, lr_mult=0.25)
            # se_relu = mx.sym.Activation(data=se_fc1, act_type='relu')
            se_relu = se_fc1
            se_fc2 = mx.symbol.FullyConnected(data=se_relu, num_hidden=num_filter, name=name + '_se_fc2',
                                             attr={'lr_mult': '0.25'} ) #, lr_mult=0.25)
            se_act = mx.sym.Activation(se_fc2, act_type="sigmoid")
            se_reshape = mx.symbol.Reshape(se_act, shape=(-1, num_filter, 1, 1), name="se_reshape")
            se_scale = mx.sym.broadcast_mul(conv3, se_reshape)
            out = se_scale
        
        if ratt == 0:
            pass
        else:
            ratt1 = mx.symbol.Convolution(data=out, num_filter=num_filter, kernel=(1,1), stride=(1,1), no_bias=True, \
                                             workspace=workspace, name=name+'_poolra1')
            ratt2 = mx.symbol.Convolution(data=ratt1, num_filter=1, kernel=(1,1), stride=(1,1), no_bias=True, \
                                              workspace=workspace, name=name+'_poolra2')
            ratt = mx.symbol.Activation(ratt2, act_type="sigmoid")
            ratt_scale = mx.sym.broadcast_mul(out, ratt)
            out = ratt_scale
            
        return out + shortcut
            
            
            
    
    
    
    
    elif bottle_neck == 2:
        # TODOOOOOOOOOOOOOOOOOOO
        raise Exception("bottle_neck error: Unimplemented Bottleneck Unit: Non-Preactivated IRB")
        # Left Branch
        conv11 = mx.sym.Convolution(data=data, num_filter=int(num_filter/expansion), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv11')
        bn11 = mx.sym.BatchNorm(data=conv11, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn11')
        act11 = mx.sym.Activation(data=bn11, act_type='relu', name=name + '_relu11')

        
        conv12 = mx.sym.Convolution(data=act11, num_filter=int(num_filter/expansion), 
                                   num_group=num_group, kernel=(3,3), stride=stride, 
                                   pad=(dilation,dilation), dilate=(dilation,dilation),
                                   no_bias=True, workspace=workspace, name=name + '_conv12')
        
        # Right Branch
        conv21 = mx.sym.Convolution(data=data, num_filter=int(num_filter/expansion/2), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv21')
        bn21 = mx.sym.BatchNorm(data=conv21, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn21')
        act21 = mx.sym.Activation(data=bn21, act_type='relu', name=name + '_relu21')

        
        conv22 = mx.sym.Convolution(data=act21, num_filter=int(num_filter/expansion/2), 
                                   num_group=num_group, kernel=(3,3), stride=stride, 
                                   pad=(dilation,dilation), dilate=(dilation,dilation),
                                   no_bias=True, workspace=workspace, name=name + '_conv22')
        
        bn22 = mx.sym.BatchNorm(data=conv22, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn22')
        act22 = mx.sym.Activation(data=bn22, act_type='relu', name=name + '_relu22')
        
        # Consecutive Conv(3,3) Use  stride=(1,1) instead of stride=(3,3)
        conv23 = mx.sym.Convolution(data=act22, num_filter=int(num_filter/expansion/2), 
                                   num_group=num_group, kernel=(3,3), stride=(1,1), 
                                   pad=(dilation,dilation), dilate=(dilation,dilation),
                                   no_bias=True, workspace=workspace, name=name + '_conv23')
        
        conv2 = mx.symbol.Concat(*[conv12, conv23])
        
        bn30 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn30')
        act30 = mx.sym.Activation(data=bn30, act_type='relu', name=name + '_relu30')
        
        conv31 = mx.sym.Convolution(data=act30, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv31')
        
        bn31 = mx.sym.BatchNorm(data=conv31, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn31')
        # Original Paper: With act31
        act31 = mx.sym.Activation(data=bn31, act_type='relu', name=name + '_relu31')
        
        
        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
        # Original Paper : act31+shortcut, else bn31 + shortcut
        if irv2: 
            eltwise =  bn31 + shortcut
        else :
            eltwise =  act31 + shortcut
        
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')
    
    
    elif bottle_neck == 3:
         # TODOOOOOOOOOOOOOOOOOO
        raise Exception("bottle_neck error: Unimplemented Bottleneck Unit: Dual Path Net.")
         
    else:
        raise Exception("bottle_neck error: Unrecognized Bottleneck params.")


        
def irnext(inputdata, units, num_stages, filter_list, num_classes, num_group, bottle_neck=1, \
               lastout = 7, expansion = 0.5, dilpat = '', irv2 = False,  deform = 0, sqex=0, ratt=0, block567=0,
           taskmode='CLS',
           seg_stride_list = [1,2,2,1], decoder=False,
           bn_mom=0.9, workspace=256, dtype='float32', memonger=False):
    
    """Return ResNeXt symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Number of Classes, 1k/5k/11k/22k/etc
    num_groups: int
        Same as irnext unit
    bottle_neck: int=0,1,2,3
        Same as irnext unit
    lastout: int
        Size of last Conv
        Original Image Size Should Be: 3,(32*lastout),(32*lastout)
    expansion: float
        Same as irnext unit
    dilpat: str
        Best Practice: DEEPLAB.SHUTTLE
        '': (1,1,1)
        DEEPLAB.SHUTTLE: (1,2,1)
        DEEPLAB.HOURGLASS: (2,1,2)
        DEEPLAB.LIN: (1,2,3)
        DEEPLAB.REVLIN: (3,2,1)
        DEEPLAB.DOUBLE: (2,2,2)
        DEEPLAB.EXP: (1,2,4)
        DEEPLAB.REVEXP: (4,2,1)
    deform: int
        Use DCN
    taskmode: str
        'CLS': Classification
        'Seg': Segmentation
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    dtype : str
        Precision (float32 or float16)
    """

    num_unit = len(units)
    assert(num_unit == num_stages)
    
    # Fix Name Alias
    data = inputdata
    
    if dtype == 'float32':
        data = mx.sym.identity(data=data, name='id')
    else:
        if dtype == 'float16':
            data = mx.sym.Cast(data=data, dtype=np.float16)
    
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    
    if num_classes in [10,100]:
        (nchannel, height, width) = (3, 4* lastout, 4*lastout)
    else:
        (nchannel, height, width) = (3, 32* lastout, 32*lastout)
    
    if height <= 32:            # such as cifar10/cifar100
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    else:                       # often expected to be 224 such as imagenet
        
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        '''
        body1 = mx.sym.Convolution(data=data, num_filter=filter_list[0]/2, kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv07", workspace=workspace)
        body1 = mx.sym.BatchNorm(data=body1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn07')
        body1 = mx.sym.Activation(data=body1, act_type='relu', name='relu07')
        body2 = mx.sym.Convolution(data=data, num_filter=filter_list[0]/4, kernel=(5, 5), stride=(2,2), pad=(2, 2),
                                  no_bias=True, name="conv05", workspace=workspace)
        body2 = mx.sym.BatchNorm(data=body2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn05')
        body2 = mx.sym.Activation(data=body2, act_type='relu', name='relu05')
        
        body3 = mx.sym.Convolution(data=data, num_filter=filter_list[0]/4, kernel=(3, 3), stride=(2,2), pad=(1, 1),
                                  no_bias=True, name="conv03", workspace=workspace)
        body3 = mx.sym.BatchNorm(data=body3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn03')
        body3 = mx.sym.Activation(data=body3, act_type='relu', name='relu03')
        
        body = mx.sym.Concat(*[body1, body2, body3])
        '''
        body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

        # To avoid mean_rgb, use another BN-Relu
    # Unit Params List:
    # data, num_filter, stride, dim_match, name, bottle_neck=1, expansion=0.5, \
    # num_group=32, dilation=1, irv2 = False, deform=0, 
    
    dilation_dict = {'DEEPLAB.SHUTTLE':[1,1,2,1],
                    'DEEPLAB.HOURGLASS':[1,2,1,2],
                    'DEEPLAB.EXP':[1,1,2,4],
                    'DEEPLAB.PLATEAU':[1,1,2,2],
                    'DEEPLAB.REVEXP':[1,4,2,1],
                    'DEEPLAB.LIN':[1,1,2,3],
                    'DEEPLAB.REVLIN':[1,3,2,1],
                    'DEEPLAB.DOUBLE':[1,2,2,2]}
    
    
    
    if taskmode == 'CLS':
        stride_plan = [1,2,2,2]
        dilation_plan = [1,1,1,1] if dilpat not in dilation_dict else dilation_dict[dilpat]
        
        
        for i in range(num_stages):
            
            current_deform = 0 if i!=num_stages else deform
            
            body = irnext_unit(body, filter_list[i+1], (stride_plan[i], stride_plan[i]), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, 
                             expansion = expansion, num_group=num_group, dilation = dilation_plan[i],
                             irv2 = irv2, deform = current_deform, sqex = sqex, ratt=ratt, 
                             bn_mom=bn_mom, workspace=workspace, memonger=memonger)
            for j in range(units[i]-1):
                body = irnext_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, expansion = expansion, num_group=num_group, 
                                 dilation = dilation_plan[i], irv2 = irv2, deform = current_deform , sqex = sqex, ratt=ratt,
                                 bn_mom=bn_mom, workspace=workspace, memonger=memonger)
        
        bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(lastout, lastout), pool_type='avg', name='pool1')
        flat = mx.sym.Flatten(data=pool1)
        fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
        if dtype == 'float16':
            fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
        return mx.sym.SoftmaxOutput(data=fc1, name='softmax')
    
    
    elif taskmode == 'SEG':
        
        # Deeplab Without Deform
        # Deeplab With Deform
        # Deeplab v1 Use Stride_List = [1,2,2,1] So a 16x Deconv Needed
        # Deeplab v2/v3 Use Stride_List = [1,2,1,1] So 1/8 gt and 1/8 img compute loss
        # Pytorch-Deeplab Use 1x+0.707x+0.5x Multi-Scale Shared Params Trick
        stride_plan = seg_stride_list
        
        dilation_plan = [1,1,1,1] if dilpat not in dilation_dict else dilation_dict[dilpat]
        
        if block567 == 1:
            dilation_plan = dilation_plan + [8,16,32]
        
        if decoder:
            #conv_basic_out = body
            #imagepyramid = [conv_basic_out]
            imagepyramid = []
        
        for i in range(num_stages):
            
            current_deform = 0 if i<3 else deform
            
            body = irnext_unit(body, filter_list[i+1], (stride_plan[i], stride_plan[i]), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, 
                             expansion = expansion, num_group=num_group, dilation = dilation_plan[i],
                             irv2 = irv2, deform = current_deform, sqex = sqex , ratt=ratt, 
                             bn_mom=bn_mom, workspace=workspace, memonger=memonger)
            for j in range(units[i]-1):
                body = irnext_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, expansion = expansion, num_group=num_group, 
                                 dilation = dilation_plan[i], irv2 = irv2, deform = current_deform , sqex = sqex, ratt =ratt, 
                                 bn_mom=bn_mom, workspace=workspace, memonger=memonger)
            if decoder and i>0:
                exec('conv_{idx}_out = body'.format(idx=i))
                exec('imagepyramid.append(conv_{idx}_out)'.format(idx=i))
        
        bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        
        if decoder:
            return imagepyramid + [body]
        else:
            return body
        

def get_conv(data, num_classes, num_layers, outfeature, bottle_neck=1, expansion=0.5,
               num_group=32, lastout=7, dilpat='', irv2=False, deform=0, sqex = 0, ratt=0, block567=0,  conv_workspace=256,
               taskmode='CLS', decoder=False, seg_stride_mode='', dtype='float32', **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    
    # Model Params List:
    # num_classes, num_layers, bottle_neck=1, expansion=0.5, \
    # num_group=32, dilation=1, irv2 = False, deform=0, taskmode, seg_stride_mode
    
    if num_classes in [10,100]:
        (nchannel, height, width) = (3, 4* lastout, 4*lastout)
    else:
        (nchannel, height, width) = (3, 32* lastout, 32*lastout)
    
    
    if height <= 32: # CIFAR10/CIFAR100
        num_stages = 3
        if num_layers == 29:
            per_unit = [3]
            filter_list = [16, int(outfeature/4), int(outfeature/2), outfeature]
            use_bottle_neck = bottle_neck
        elif (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, int(outfeature/4), int(outfeature/2), outfeature]
            use_bottle_neck = bottle_neck
            
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, int(outfeature/4), int(outfeature/2), outfeature]
            use_bottle_neck = 0
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
        
        units = per_unit * num_stages
        
    else:
        
        if num_layers >= 26:
            filter_list = [64, int(outfeature/8) , int(outfeature/4), int(outfeature/2), outfeature ]
            use_bottle_neck = bottle_neck
        else:
            filter_list = [64, int(outfeature/8) , int(outfeature/4), int(outfeature/2), outfeature ]
            use_bottle_neck = 0
        
        if block567 == 1:
            filter_list = filter_list + [outfeature,outfeature,outfeature]
        
        num_stages = 4 if block567==0 else 7
        if num_layers == 18:
            units = [2, 2, 2, 2]
        #elif num_layers == 34:
        #    units = [3, 4, 6, 3]
        elif num_layers == 23:
            units = [2, 2, 2, 1]
        elif num_layers == 26:
            units = [2, 2, 2, 2]
        elif num_layers == 29:
            units = [2, 2, 2, 3]
        elif num_layers == 38:
            units = [3, 3, 3, 3]
        elif num_layers == 41:
            units = [3, 3, 4, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 59:
            units = [3, 4, 9, 3]
        elif num_layers == 62:
            units = [3, 4, 10, 3]
        elif num_layers == 65:
            units = [3, 5, 10, 3]
        elif num_layers == 74:
            units = [3, 6, 12, 3]
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
            
        if block567 == 1:
            units = units + [3,3,3]

    if seg_stride_mode == '4x':
        seg_stride_list = [1,1,1,1]
    elif seg_stride_mode == '8x':
        seg_stride_list = [1,2,1,1]
    elif seg_stride_mode == '16x':
        seg_stride_list = [1,2,2,1]
    else:
        seg_stride_list = [1,2,2,1]
    
    if block567 == 1:
        seg_stride_list = seg_stride_list + [1,1,1]
        
    
    return irnext(data, 
                  units,
                  num_stages,
                  filter_list,
                  num_classes,
                  num_group, 
                  bottle_neck = use_bottle_neck,
                  lastout     = lastout,
                  expansion   = expansion,
                  dilpat      = dilpat, 
                  irv2        = irv2,
                  deform      = deform, 
                  sqex        = sqex, 
                  ratt        = ratt,
                  block567    = block567,
                  taskmode    = taskmode,
                  seg_stride_list = seg_stride_list,
                  decoder     = decoder,
                  workspace   = conv_workspace,
                  dtype       = dtype)
        
#### Original Deeplab DCN
        
        
        
# Todo 0 & 3 .

# Symbol
class irnext_deeplab_dcn():
    
    
    def __init__(self, num_classes , num_layers , outfeature, bottle_neck=1, expansion=0.5,\
                num_group=32, lastout=7, dilpat='', irv2=False, deform=0, sqex = 0, ratt = 0, block567=0 , 
                 aspp = 0, 
                 conv_workspace=256,
                taskmode='CLS', seg_stride_mode='', deeplabversion=2 , dtype='float32', **kwargs):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 4096
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.outfeature = outfeature
        self.bottle_neck = bottle_neck
        self.expansion = expansion
        self.num_group = num_group
        self.lastout = lastout
        self.dilpat = dilpat
        self.irv2 = irv2
        self.deform = deform
        self.sqex = sqex
        self.ratt = ratt
        self.block567 = block567
        self.taskmode = taskmode
        self.seg_stride_mode = seg_stride_mode
        self.deeplabversion = deeplabversion
        self.atrouslist = [] if aspp==0 else [3,6,12,18,24] #6,12,18
        # (3, 4, 23, 3) # use for 101
        # filter_list = [256, 512, 1024, 2048]
        

    def get_cls_symbol(self, **kwargs):
        
        data = mx.symbol.Variable(name="data")
        
        return get_conv(  data,
                          self.num_classes,
                          self.num_layers,
                          self.outfeature,
                          bottle_neck=self.bottle_neck,
                          expansion=self.expansion, 
                          num_group=self.num_group, 
                          lastout=self.lastout,
                          dilpat=self.dilpat, 
                          irv2=self.irv2, 
                          deform=self.deform, 
                          sqex=self.sqex,
                          ratt=self.ratt,
                          conv_workspace=256,
                          taskmode='CLS', 
                          seg_stride_mode='', dtype='float32', **kwargs)
        
    def get_seg_symbol(self, **kwargs):
        
        data = mx.symbol.Variable(name="data")
        seg_cls_gt = mx.symbol.Variable(name="softmax_label")
        
        if self.deeplabversion == 3 :
            self.decoder = True
        else:
            self.decoder = False
        
        conv_feat = get_conv(data,
                            self.num_classes,
                            self.num_layers,
                            self.outfeature,
                            bottle_neck=self.bottle_neck,
                            expansion=self.expansion, 
                            num_group=self.num_group, 
                            lastout=self.lastout,
                            dilpat=self.dilpat, 
                            irv2=self.irv2, 
                            deform=self.deform,
                            decoder=self.decoder,
                            sqex=self.sqex,
                            ratt=self.ratt,
                            block567=self.block567,
                            conv_workspace=256,
                            taskmode='SEG',
                            seg_stride_mode=self.seg_stride_mode,
                            dtype='float32',
                            **kwargs)
        
        
        
        if self.decoder:
            conv_feat_list = conv_feat
            conv_feat = conv_feat[-1]
        
        fc6_bias = mx.symbol.Variable('fc6_bias', lr_mult=2.0)
        fc6_weight = mx.symbol.Variable('fc6_weight', lr_mult=1.0)
        fc6 = mx.symbol.Convolution(data=conv_feat, kernel=(1, 1), pad=(0, 0), num_filter=self.outfeature, name="fc6",
                                    bias=fc6_bias, weight=fc6_weight, workspace=self.workspace)
        relu_fc6 = mx.sym.Activation(data=fc6, act_type='relu', name='relu_fc6')
        

        if self.seg_stride_mode == '4x':
            upstride = 4
        elif self.seg_stride_mode == '8x':
            upstride = 8
        elif self.seg_stride_mode == '16x':
            upstride = 16
        else:
            upstride = 16
        
        
        if self.deeplabversion == 1:
            
            score_bias = mx.symbol.Variable('score_bias', lr_mult=2.0)
            score_weight = mx.symbol.Variable('score_weight', lr_mult=1.0)
            score = mx.symbol.Convolution(data=relu_fc6, kernel=(1, 1), pad=(0, 0), num_filter=self.num_classes, name="score",
                                      bias=score_bias, weight=score_weight, workspace=self.workspace)
            upsampling = mx.symbol.Deconvolution(data=score, num_filter=self.num_classes, kernel=(upstride*2, upstride*2), 
                                             stride=(upstride, upstride),
                                             num_group=self.num_classes, no_bias=True, name='upsampling',
                                             attr={'lr_mult': '0.0'}, workspace=self.workspace)
        
            croped_score = mx.symbol.Crop(*[upsampling, data], offset=(upstride/2, upstride/2), name='croped_score')
            softmax = mx.symbol.SoftmaxOutput(data=croped_score, label=seg_cls_gt, normalization='valid', multi_output=True,
                                          use_ignore=True, ignore_label=255, name="softmax")
            

            return softmax
        elif self.deeplabversion == 2 :
            atrouslistlen = len(self.atrouslist)
            
            # V3
            score_basic_bias = mx.symbol.Variable('score_basic_bias', lr_mult=2.0)
            score_basic_weight = mx.symbol.Variable('score_basic_weight',lr_mult=1.0)
            score_basic = mx.symbol.Convolution(data=relu_fc6, kernel=(1, 1),\
                        num_filter=self.num_classes, \
                        name="score_basic",bias=score_basic_bias, weight=score_basic_weight, \
                        workspace=self.workspace)
            
            atrouslistsymbol = [score_basic]
            
            for i in range(atrouslistlen):
                thisatrous = self.atrouslist[i]
                exec('score_{ind}_bias = mx.symbol.Variable(\'score_{ind}_bias\', lr_mult=2.0)'.format(ind=i))
                exec('score_{ind}_weight = mx.symbol.Variable(\'score_{ind}_weight\', lr_mult=1.0)'.format(ind=i))
                exec('score_{ind} = mx.symbol.Convolution(data=relu_fc6, kernel=(3, 3), pad=(thisatrous, thisatrous),\
                        dilate=(thisatrous, thisatrous) ,num_filter=self.num_classes, \
                        name="score_{ind}",bias=score_{ind}_bias, weight=score_{ind}_weight, \
                        workspace=self.workspace)'.format(ind=i))
                
                exec('atrouslistsymbol.append(score_{ind})'.format(ind=i))
                
                '''
                if i==0:
                    score = score_0
                else:
                    exec('score = score + score_{ind}'.format(ind=i))
                '''
            score = mx.symbol.Concat(*atrouslistsymbol)
            '''
            if self.sqex:
                score_pool_se = mx.symbol.Pooling(data=score, cudnn_off=True, global_pool=True, \
                                            kernel=(7, 7), pool_type='avg', name='final_score_se_pool')
                score_flat = mx.symbol.Flatten(data=score_pool_se)
                score_se_fc1 = mx.symbol.FullyConnected(data=score_flat, num_hidden=12,\
                                                  name='score_se_fc1') #, lr_mult=0.25)
                score_se_relu = mx.sym.Activation(data=score_se_fc1, act_type='relu')
                score_se_fc2 = mx.symbol.FullyConnected(data=score_se_relu, num_hidden=12, name= 'score_se_fc2') #, lr_mult=0.25)
                score_se_act = mx.sym.Activation(score_se_fc2, act_type="sigmoid")
                score_se_reshape = mx.symbol.Reshape(score_se_act, shape=(-1, 12, 1, 1), name="score_se_reshape")
                score_se_scale = mx.sym.broadcast_mul(score, score_se_reshape)
                score = score_se_scale
            '''
            
            
            upsampling = mx.symbol.Deconvolution(data=score, num_filter=self.num_classes, kernel=(upstride*2, upstride*2), 
                                             stride=(upstride, upstride),
                                             num_group=self.num_classes, no_bias=True, name='upsampling',
                                             attr={'lr_mult': '0.1'}, workspace=self.workspace)
        
            croped_score = mx.symbol.Crop(*[upsampling, data], offset=(upstride/2, upstride/2), name='croped_score')
            softmax = mx.symbol.SoftmaxOutput(data=croped_score, label=seg_cls_gt, normalization='valid', multi_output=True,
                                          attr={'lr_mult': '1.0'},use_ignore=True, ignore_label=255, name="softmax")

            return softmax
        elif self.deeplabversion == 3:
            
            # With Decoder, Not Image-Level Pooling, Hard Coded.
            # conv_basic_out, conv_0_out, conv_1_out, conv_2_out, conv_3_out = conv_feat_list[:-1]
            conv_1_out, conv_2_out, conv_3_out = conv_feat_list[:-1]
            
            if self.seg_stride_mode == '8x':
                #conv_basic_out = mx.sym.BatchNorm(data=conv_basic_out, fix_gamma=False, eps=2e-5, momentum=0.9, name='bn_basic_out')
                #conv_basic_out = mx.sym.Pooling(data=conv_basic_out, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='avg')
                #conv_0_out = mx.sym.BatchNorm(data=conv_0_out, fix_gamma=False, eps=2e-5, momentum=0.9, name='bn_0_out')
                #conv_0_out = mx.sym.Pooling(data=conv_0_out, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='avg')
                
                #conv_1_out = mx.sym.BatchNorm(data=conv_1_out, fix_gamma=False, eps=2e-5, momentum=0.9, name='bn_1_out')
                #conv_2_out = mx.sym.BatchNorm(data=conv_2_out, fix_gamma=False, eps=2e-5, momentum=0.9, name='bn_2_out')
                #conv_3_out = mx.sym.BatchNorm(data=conv_3_out, fix_gamma=False, eps=2e-5, momentum=0.9, name='bn_3_out')
                
                #relu_fc6 = mx.symbol.Concat(*[conv_basic_out,conv_0_out,conv_1_out,conv_2_out,conv_3_out,relu_fc6])
                relu_fc6 = mx.symbol.Concat(*[conv_1_out,conv_2_out,conv_3_out,relu_fc6])
                
            
            atrouslistlen = len(self.atrouslist)
            
            # V3
            score_basic_bias = mx.symbol.Variable('score_basic_bias', lr_mult=2.0)
            score_basic_weight = mx.symbol.Variable('score_basic_weight',lr_mult=1.0)
            score_basic = mx.symbol.Convolution(data=relu_fc6, kernel=(1, 1),\
                        num_filter=self.num_classes, \
                        name="score_basic",bias=score_basic_bias, weight=score_basic_weight, \
                        workspace=self.workspace)
            
            atrouslistsymbol = [score_basic]
            
            for i in range(atrouslistlen):
                thisatrous = self.atrouslist[i]
                exec('score_{ind}_bias = mx.symbol.Variable(\'score_{ind}_bias\', lr_mult=2.0)'.format(ind=i))
                exec('score_{ind}_weight = mx.symbol.Variable(\'score_{ind}_weight\', lr_mult=1.0)'.format(ind=i))
                exec('score_{ind} = mx.symbol.Convolution(data=relu_fc6, kernel=(3, 3), pad=(thisatrous, thisatrous),\
                        dilate=(thisatrous, thisatrous) ,num_filter=self.num_classes, \
                        name="score_{ind}",bias=score_{ind}_bias, weight=score_{ind}_weight, \
                        workspace=self.workspace)'.format(ind=i))
                
                exec('atrouslistsymbol.append(score_{ind})'.format(ind=i))
                
                
                #if i==0:
                #    score = score_0
                #else:
                #    exec('score = score + score_{ind}'.format(ind=i))
                
            score = mx.symbol.Concat(*atrouslistsymbol)

            
            
            upsampling = mx.symbol.Deconvolution(data=score, num_filter=self.num_classes, kernel=(upstride*2, upstride*2), 
                                             stride=(upstride, upstride),
                                             num_group=self.num_classes, no_bias=True, name='upsampling',
                                             attr={'lr_mult': '0.0'}, workspace=self.workspace)
        
            croped_score = mx.symbol.Crop(*[upsampling, data], offset=(upstride/2, upstride/2), name='croped_score')
            softmax = mx.symbol.SoftmaxOutput(data=croped_score, label=seg_cls_gt, normalization='valid', multi_output=True,
                                          attr={'lr_mult': '1.0'},use_ignore=True, ignore_label=255, name="softmax")

            return softmax
            
    
    '''
    def init_weights(self, cfg, arg_params, aux_params):
        arg_params['res5a_branch2b_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5a_branch2b_offset_weight'])
        arg_params['res5a_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5a_branch2b_offset_bias'])
        arg_params['res5b_branch2b_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5b_branch2b_offset_weight'])
        arg_params['res5b_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5b_branch2b_offset_bias'])
        arg_params['res5c_branch2b_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5c_branch2b_offset_weight'])
        arg_params['res5c_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5c_branch2b_offset_bias'])
        arg_params['fc6_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc6_weight'])
        arg_params['fc6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc6_bias'])
        arg_params['score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['score_weight'])
        arg_params['score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['score_bias'])
        arg_params['upsampling_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['upsampling_weight'])

        init = mx.init.Initializer()
        init._init_bilinear('upsample_weight', arg_params['upsampling_weight'])

    '''
# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zheng Zhang SL test
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
from op.lsoftmax import LSoftmaxOp
# from utils.symbol import Symbol


###### UNIT LIST #######

# Todo 1,2,4

def irnext_unit(data, num_filter, stride, dim_match, name, bottle_neck=1, expansion=0.5, \
                 num_group=32, dilation=1, irv2 = False, deform=1, sqex=1, ratt=0, scale=1.0,
                 bn_mom=0.9, unitbatchnorm=True, workspace=256, memonger=False):
    
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
            
        return conv2 + shortcut * scale
        
    
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
            
            offsetweight = mx.symbol.Variable(name+'_offset_weight', lr_mult=3.0)
            offsetbias = mx.symbol.Variable(name+'_offset_bias', lr_mult=6.0)
            offset = mx.symbol.Convolution(name=name+'_offset', data = act2,
                                                      num_filter=18, pad=(dilation, dilation), kernel=(3, 3), stride=stride,
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
            se_fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=int(num_filter/expansion/4), name=name + '_se_fc1',
                                             attr={'lr_mult': '0.25'} ) #, lr_mult=0.25)
            # se_relu = mx.sym.Activation(data=se_fc1, act_type='relu')
            se_relu = se_fc1
            se_fc2 = mx.symbol.FullyConnected(data=se_relu, num_hidden=int(num_filter), name=name + '_se_fc2',
                                             attr={'lr_mult': '0.25'} ) #, lr_mult=0.25)
            se_act = mx.sym.Activation(se_fc2, act_type="sigmoid")
            se_reshape = mx.symbol.Reshape(se_act, shape=(-1, num_filter, 1, 1), name="se_reshape")
            se_scale = mx.sym.broadcast_mul(conv3, se_reshape)
            out = se_scale
        
        if ratt == 0:
            pass
        else:
            ratt1 = mx.symbol.Convolution(data=out, num_filter=num_filter, kernel=(1,1), stride=(1,1), no_bias=True, \
                                             workspace=workspace, name=name+'_poolra1',
                                             attr={'lr_mult': '0.25'})
            ratt2 = mx.symbol.Convolution(data=ratt1, num_filter=1, kernel=(1,1), stride=(1,1), no_bias=True, \
                                              workspace=workspace, name=name+'_poolra2',
                                             attr={'lr_mult': '0.25'})
            ratt = mx.symbol.Activation(ratt2, act_type="sigmoid")
            ratt_scale = mx.sym.broadcast_mul(out, ratt)
            out = ratt_scale
            
        return out + shortcut * scale
            
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
               lastout = 7, expansion = 0.5, dilpat = '', irv2 = False,  deform = 0, sqex=0, ratt=0, scale=1.0, usemaxavg=0,
               lmar = 0, lmarbeta=1, lmarbetamin=0, lmarscale=1,
           block567='',
           taskmode='CLS',
           seg_stride_list = [1,2,2,1], decoder=False,
           bn_mom=0.9, workspace=512, dtype='float32', memonger=False):
    
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
        'SEG': Segmentation
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
        if taskmode != "KEY":
            body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
        else:
            body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

        # To avoid mean_rgb, use another BN-Relu
    # Unit Params List:
    # data, num_filter, stride, dim_match, name, bottle_neck=1, expansion=0.5, \
    # num_group=32, dilation=1, irv2 = False, deform=0, 
    
    dilation_dict = {'DEEPLAB.SHUTTLE':[1,1,2,1],
                    'DEEPLAB.HEAD':[1,2,1,1],
                    'DEEPLAB.TAIL':[1,1,1,2],
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
            
            current_deform = 0 if i!=(num_stages-1) else deform
            
            body = irnext_unit(body, filter_list[i+1], (stride_plan[i], stride_plan[i]), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, 
                             expansion = expansion, num_group=num_group, dilation = dilation_plan[i],
                             irv2 = irv2, deform = current_deform, sqex = sqex, ratt=ratt, scale=scale, 
                             bn_mom=bn_mom, workspace=workspace, memonger=memonger)
            for j in range(units[i]-1):
                body = irnext_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, expansion = expansion, num_group=num_group, 
                                 dilation = dilation_plan[i], irv2 = irv2, deform = current_deform , sqex = sqex, ratt=ratt,
                                   scale=scale,
                                   bn_mom=bn_mom, workspace=workspace, memonger=memonger)
        
        bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        
        if usemaxavg == 0:
            pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(lastout, lastout), pool_type='avg', name='pool1')
            flat = mx.sym.Flatten(data=pool1)
        else:
            pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(lastout, lastout), pool_type='avg', name='pool1')
            pool2 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(lastout, lastout), pool_type='max', name='pool2')
            flat = mx.sym.Flatten(data=pool1*0.5+pool2*0.5)
            
        fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
        
        return fc1
    
    
    elif taskmode == 'SEG':
        
        # Deeplab Without Deform
        # Deeplab With Deform
        # Deeplab v1 Use Stride_List = [1,2,2,1] So a 16x Deconv Needed
        # Deeplab v2/v3 Use Stride_List = [1,2,1,1] So 1/8 gt and 1/8 img compute loss
        # Pytorch-Deeplab Use 1x+0.707x+0.5x Multi-Scale Shared Params Trick
        stride_plan = seg_stride_list
        
        dilation_plan = [1,1,1,1] if dilpat not in dilation_dict else dilation_dict[dilpat]
        
        if block567 != '' :
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
        
    elif taskmode[:3] == "DET":
        
        stride_plan = [1,2,2,1]
        dilation_plan = [1,1,1,2] if dilpat not in dilation_dict else dilation_dict[dilpat]
        
        # Expect 3 or 4. So, We Expect taskmode to be "DET3" or "DET4" to specify how much stages it use.
        num_stages = int(taskmode[3])
        
        
        returnList = []
        
        for i in range(num_stages):
            
            current_deform = 0 if i!=(num_stages-1) else deform
            
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
            
            if i>=2:
                exec('body_{0} = body'.format(i))
                exec('returnList.append(body_{0})'.format(i))
                
            
        #bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        #relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        
        return returnList
        
    elif taskmode == "KEY":
        ## TODO: CPM KEYPOINT HUMAN BODY MODEL
        stride_plan = [1,2,1,1]
        dilation_plan = [1,1,2,4] if dilpat not in dilation_dict else dilation_dict[dilpat]
        
        num_stages=4
        
        for i in range(num_stages):
            
            current_deform = 0 if i!=(num_stages-1) else deform
            
            body = irnext_unit(body, filter_list[i+1], (stride_plan[i], stride_plan[i]), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, 
                             expansion = expansion, num_group=num_group, dilation = dilation_plan[i],
                             irv2 = irv2, deform = current_deform, sqex = sqex, ratt=ratt, scale=scale, 
                             bn_mom=bn_mom, workspace=workspace, memonger=memonger)
            for j in range(units[i]-1):
                body = irnext_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, expansion = expansion, num_group=num_group, 
                                 dilation = dilation_plan[i], irv2 = irv2, deform = current_deform , sqex = sqex, ratt=ratt,
                                   scale=scale,
                                   bn_mom=bn_mom, workspace=workspace, memonger=memonger)
        
        bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        
        return relu1 # KEY BACKBONE
   
        
def get_conv(data, num_classes, num_layers, outfeature, bottle_neck=1, expansion=0.5,
               num_group=32, lastout=7, dilpat='', irv2=False, deform=0, sqex = 0, ratt=0, block567='', scale=1.0, usemaxavg=0,
               lmar = 0, lmarbeta=1, lmarbetamin=0, lmarscale=1,
               conv_workspace=512,
               taskmode='CLS', decoder=False, seg_stride_mode='', dtype='float32', **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    
    This Functions wraps irnext with Different Problem Settings (input resolution, network depth hypers etc.)
    
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
        
        if block567 != '' :
            filter_list = filter_list + [outfeature,outfeature,outfeature]
        
        num_stages = 4 if block567=='' else 4+len(block567.split(','))
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
        elif num_layers == 71:
            units = [3, 5, 12, 3]
        elif num_layers == 74:
            units = [3, 6, 12, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 104:
            units = [3, 6, 22, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
            
        if block567 != '' :
            units = units + [int(i) for i in block567.split(',')]

    if seg_stride_mode == '4x':
        seg_stride_list = [1,1,1,1]
    elif seg_stride_mode == '8x':
        seg_stride_list = [1,2,1,1]
    elif seg_stride_mode == '16x':
        seg_stride_list = [1,2,2,1]
    else:
        seg_stride_list = [1,2,2,1]
    
    if block567 != '':
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
                  usemaxavg   = usemaxavg,
                  scale       = scale,
                  lmar        = lmar, 
                  lmarbeta    = lmarbeta,
                  lmarbetamin = lmarbetamin,
                  lmarscale   = lmarscale,
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
                num_group=32, lastout=7, dilpat='', irv2=False, deform=0, sqex = 0, ratt = 0, usemaxavg=0, scale=1.0, 
                 lmar = 0, lmarbeta=1, lmarbetamin=0, lmarscale=1,
                 block567='' , 
                 aspp = 0, usemax =0,
                 conv_workspace=512,
                taskmode='CLS', seg_stride_mode='', deeplabversion=2 , dtype='float32', **kwargs):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = conv_workspace
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
        self.usemaxavg = usemaxavg
        self.scale = scale
        self.lmar = lmar
        self.lmarbeta = lmarbeta
        self.lmarbetamin = lmarbetamin
        self.lmarscale = lmarscale
        self.block567 = block567
        self.usemax = usemax
        self.taskmode = taskmode
        self.seg_stride_mode = seg_stride_mode
        self.deeplabversion = deeplabversion
        self.atrouslist = [] if aspp==0 else [3,6,12,18,24] #6,12,18
        self.dtype=dtype
        # (3, 4, 23, 3) # use for 101
        # filter_list = [256, 512, 1024, 2048]
        

    def get_cls_symbol(self, **kwargs):
        
        data = mx.symbol.Variable(name="data")
        label = mx.symbol.Variable(name="softmax_label")
        
        fc1 =  get_conv(  data,
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
                          usemaxavg=self.usemaxavg,
                          scale=self.scale,
                          lmar=self.lmar,
                          lmarbeta=self.lmarbeta,
                          lmarbetamin=self.lmarbetamin,
                          lmarscale=self.lmarscale,
                          conv_workspace=self.workspace,
                          taskmode='CLS', 
                          seg_stride_mode='', dtype='float32', **kwargs)
    
    
        
        if self.lmar>0:
            fc4 = mx.sym.LSoftmax(data=fc1, label=label, num_hidden=self.num_classes, #
                                  beta=self.lmarbeta, margin=self.lmar, scale=self.lmarscale,
                                  beta_min=self.lmarbetamin, verbose=True)
            
            return mx.sym.SoftmaxOutput(data=fc4, label=label)
            '''
            fc1 = mx.sym.Custom(data=fc1, label=label,
                                num_hidden=self.num_classes,
                                beta=self.lmarbeta, margin=self.lmar, scale=self.lmarscale,
                                beta_min=self.lmarbetamin, op_type='LSoftmax')
                                '''
        
        if self.dtype == 'float16':
            fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
        return mx.sym.SoftmaxOutput(data=fc1, label=label, name='softmax') #
        
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
                
                if self.usemax:
                    if i==0:
                        score = score_0
                    else:
                        exec('score = mx.symbol.broadcast_maximum(score,score_{ind})'.format(ind=i))
                else:
                    exec('atrouslistsymbol.append(score_{ind})'.format(ind=i))
                
                '''
                if i==0:
                    score = score_0
                else:
                    exec('score = score + score_{ind}'.format(ind=i))
                '''
            if self.usemax:
                pass
                #score = mx.sym.BatchNorm(data=score, fix_gamma=False, momentum=0.9, eps=2e-5, name='maxbn')
                #score = mx.sym.Activation(data=score, act_type='relu')
            else:
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
                      
                
                if self.usemax:
                    if i==0:
                        score = score_0
                    else:
                        exec('score = mx.symbol.broadcast_maximum(score,score_{ind})'.format(ind=i))
                else:
                    exec('atrouslistsymbol.append(score_{ind})'.format(ind=i))
                #if i==0:
                #    score = score_0
                #else:
                #    exec('score = score + score_{ind}'.format(ind=i))
            if self.usemax:
                score = mx.sym.BatchNorm(data=score, fix_gamma=False, momentum=0.9, eps=2e-5, name='maxbn')
            else:
                score = mx.symbol.Concat(*atrouslistsymbol)

            
            
            upsampling = mx.symbol.Deconvolution(data=score, num_filter=self.num_classes, kernel=(upstride*2, upstride*2), 
                                             stride=(upstride, upstride),
                                             num_group=self.num_classes, no_bias=True, name='upsampling',
                                             attr={'lr_mult': '0.0'}, workspace=self.workspace)
        
            croped_score = mx.symbol.Crop(*[upsampling, data], offset=(upstride/2, upstride/2), name='croped_score')
            softmax = mx.symbol.SoftmaxOutput(data=croped_score, label=seg_cls_gt, normalization='valid', multi_output=True,
                                          attr={'lr_mult': '1.0'},use_ignore=True, ignore_label=255, name="softmax")

            return softmax
            
         
    ##### SO MANY TODOOOOOOOOOOOS
    
    
    # A Universal Function for RPN in Detectors

    def get_det_rpn(self, conv_feat, num_anchors):
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
        return rpn_cls_score, rpn_bbox_pred


    def get_det_symbol(self, cfg, is_train=True\
                      ## SO MANY KWARGS\
                      ):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

        
        
        conv_feat4, conv_feat5 = get_conv(  data,
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
                          lmar=self.lmar,
                          lmarbeta=self.lmarbeta,
                          lmarbetamin=self.lmarbetamin,
                          lmarscale=self.lmarscale,
                          conv_workspace=self.workspace,
                          taskmode='DET4', 
                          seg_stride_mode='', dtype='float32', **kwargs)
                       
                       
            
        # shared convolutional layers
        # BN-Relu conv_feat4
                       
        conv_feat4_bn = mx.sym.BatchNorm(data=conv_feat4, fix_gamma=False, eps=2e-5, \
                                         momentum=bn_mom, name=name + '_conv_feat4_bn')
        conv_feat = mx.sym.Activation(data=conv_feat4_bn, act_type='relu', name=name + 'conv_feat4_relu')
        
        # res5, BN-Relu conv_feat5
        
        conv_feat5_bn = mx.sym.BatchNorm(data=conv_feat5, fix_gamma=False, eps=2e-5, \
                                         momentum=bn_mom, name=name + '_conv_feat5_bn')
        relu1 = mx.sym.Activation(data=conv_feat5_bn, act_type='relu', name=name + 'conv_feat5_relu')
        

        
        rpn_cls_score, rpn_bbox_pred = self.get_det_rpn(conv_feat, num_anchors)

        if is_train:
            # prepare rpn data
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            # ROI proposal
            rpn_cls_act = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
            rpn_cls_act_reshape = mx.sym.Reshape(
                data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
            if cfg.TRAIN.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            # ROI proposal target
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
            rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)
        else:
            # ROI Proposal
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
            if cfg.TEST.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)



        # conv_new_1
        conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=1024, name="conv_new_1", lr_mult=3.0)
        relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu1')

        # rfcn_cls/rfcn_bbox
        rfcn_cls = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        # trans_cls / trans_cls
        rfcn_cls_offset_t = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=2 * 7 * 7 * num_classes, name="rfcn_cls_offset_t")
        rfcn_bbox_offset_t = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7 * 7 * 2, name="rfcn_bbox_offset_t")

        rfcn_cls_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_cls_offset', data=rfcn_cls_offset_t, rois=rois, group_size=7, pooled_size=7,
                                                                sample_per_part=4, no_trans=True, part_size=7, output_dim=2 * num_classes, spatial_scale=0.0625)
        rfcn_bbox_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_bbox_offset', data=rfcn_bbox_offset_t, rois=rois, group_size=7, pooled_size=7,
                                                                 sample_per_part=4, no_trans=True, part_size=7, output_dim=2, spatial_scale=0.0625)

        psroipooled_cls_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, trans=rfcn_cls_offset,
                                                                     group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1,
                                                                     output_dim=num_classes, spatial_scale=0.0625, part_size=7)
        psroipooled_loc_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, trans=rfcn_bbox_offset,
                                                                     group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1,
                                                                     output_dim=8, spatial_scale=0.0625, part_size=7)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_reshape')
            group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred])

        self.sym = group
        return group

    def get_det_symbol_rpn(self, cfg, is_train=True):
        # config alias for convenient
        num_anchors = cfg.network.NUM_ANCHORS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        
        conv_feat4 = get_conv(  data,
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
                          lmar=self.lmar,
                          lmarbeta=self.lmarbeta,
                          lmarbetamin=self.lmarbetamin,
                          lmarscale=self.lmarscale,
                          conv_workspace=self.workspace,
                          taskmode='DET3', 
                          seg_stride_mode='', dtype='float32', **kwargs) [0]
        
        
        
        # BN-Relu conv_feat4
                       
        conv_feat4_bn = mx.sym.BatchNorm(data=conv_feat4, fix_gamma=False, eps=2e-5, \
                                         momentum=bn_mom, name=name + '_conv_feat4_bn')
        conv_feat = mx.sym.Activation(data=conv_feat4_bn, act_type='relu', name=name + 'conv_feat4_relu')
        
        rpn_cls_score, rpn_bbox_pred = self.get_det_rpn(conv_feat, num_anchors)
        
        if is_train:
            # prepare rpn data
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob",
                                                grad_scale=1.0)
            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)
            group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss])
        else:
            # ROI Proposal
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
            if cfg.TEST.CXX_PROPOSAL:
                rois, score = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            else:
                rois, score = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
                group = mx.symbol.Group([rois, score])
        self.sym = group
        return group

    def get_det_symbol_rfcn(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        # input init
        if is_train:
            data = mx.symbol.Variable(name="data")
            rois = mx.symbol.Variable(name='rois')
            label = mx.symbol.Variable(name='label')
            bbox_target = mx.symbol.Variable(name='bbox_target')
            bbox_weight = mx.symbol.Variable(name='bbox_weight')
            # reshape input
            rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')
            label = mx.symbol.Reshape(data=label, shape=(-1,), name='label_reshape')
            bbox_target = mx.symbol.Reshape(data=bbox_target, shape=(-1, 4 * num_reg_classes), name='bbox_target_reshape')
            bbox_weight = mx.symbol.Reshape(data=bbox_weight, shape=(-1, 4 * num_reg_classes), name='bbox_weight_reshape')
        else:
            data = mx.sym.Variable(name="data")
            rois = mx.symbol.Variable(name='rois')
            # reshape input
            rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')

        # shared convolutional layers
        
        conv_feat4, conv_feat5 = get_conv(  data,
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
                          lmar=self.lmar,
                          lmarbeta=self.lmarbeta,
                          lmarbetamin=self.lmarbetamin,
                          lmarscale=self.lmarscale,
                          conv_workspace=self.workspace,
                          taskmode='DET4', 
                          seg_stride_mode='', dtype='float32', **kwargs)
                       
                       
            
        # shared convolutional layers
        # BN-Relu conv_feat4
                       
        conv_feat4_bn = mx.sym.BatchNorm(data=conv_feat4, fix_gamma=False, eps=2e-5, \
                                         momentum=bn_mom, name=name + '_conv_feat4_bn')
        conv_feat = mx.sym.Activation(data=conv_feat4_bn, act_type='relu', name=name + 'conv_feat4_relu')
        
        # res5, BN-Relu conv_feat5
        
        conv_feat5_bn = mx.sym.BatchNorm(data=conv_feat5, fix_gamma=False, eps=2e-5, \
                                         momentum=bn_mom, name=name + '_conv_feat5_bn')
        relu1 = mx.sym.Activation(data=conv_feat5_bn, act_type='relu', name=name + 'conv_feat5_relu')
        
        

        # conv_new_1
        conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=1024, name="conv_new_1", lr_mult=3.0)
        relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu1')

        # rfcn_cls/rfcn_bbox
        rfcn_cls = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        # trans_cls / trans_cls
        rfcn_cls_offset_t = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=2 * 7 * 7 * num_classes, name="rfcn_cls_offset_t")
        rfcn_bbox_offset_t = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7 * 7 * 2, name="rfcn_bbox_offset_t")

        rfcn_cls_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_cls_offset', data=rfcn_cls_offset_t, rois=rois, group_size=7, pooled_size=7,
                                                                sample_per_part=4, no_trans=True, part_size=7, output_dim=2 * num_classes, spatial_scale=0.0625)
        rfcn_bbox_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_bbox_offset', data=rfcn_bbox_offset_t, rois=rois, group_size=7, pooled_size=7,
                                                                 sample_per_part=4, no_trans=True, part_size=7, output_dim=2, spatial_scale=0.0625)

        psroipooled_cls_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, trans=rfcn_cls_offset,
                                                                     group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1,
                                                                     output_dim=num_classes, spatial_scale=0.0625, part_size=7)
        psroipooled_loc_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, trans=rfcn_bbox_offset,
                                                                     group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1,
                                                                     output_dim=8, spatial_scale=0.0625, part_size=7)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1, grad_scale=1.0)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid', grad_scale=1.0)
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)

            # reshape output
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_reshape')
            group = mx.sym.Group([cls_prob, bbox_loss, mx.sym.BlockGrad(label)]) if cfg.TRAIN.ENABLE_OHEM else mx.sym.Group([cls_prob, bbox_loss])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([cls_prob, bbox_pred])

        self.sym = group
        return group
    
    
    
    def get_key_symbol():
        
        
        numofparts = 15
        numoflinks = 13
        
        
        data = mx.symbol.Variable(name='data')
        ## heat map of human parts
        heatmaplabel = mx.sym.Variable("heatmaplabel")
        ## part affinity graph
        partaffinityglabel = mx.sym.Variable('partaffinityglabel')

        heatweight = mx.sym.Variable('heatweight')
    
        vecweight = mx.sym.Variable('vecweight')
        
        
        relu4_2 = get_conv(data,
                           0,
                           self.num_layers,
                           self.outfeature,
                           taskmode='KEY'
                           )
        
        # TODO: Clean Up The XX Code.
        
        conv4_3_CPM = mx.symbol.Convolution(name='conv4_3_CPM', data=relu4_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        relu4_3_CPM = mx.symbol.Activation(name='relu4_3_CPM', data=conv4_3_CPM , act_type='relu')
        conv4_4_CPM = mx.symbol.Convolution(name='conv4_4_CPM', data=relu4_3_CPM , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        relu4_4_CPM = mx.symbol.Activation(name='relu4_4_CPM', data=conv4_4_CPM , act_type='relu')
        conv5_1_CPM_L1 = mx.symbol.Convolution(name='conv5_1_CPM_L1', data=relu4_4_CPM , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        relu5_1_CPM_L1 = mx.symbol.Activation(name='relu5_1_CPM_L1', data=conv5_1_CPM_L1 , act_type='relu')
        conv5_1_CPM_L2 = mx.symbol.Convolution(name='conv5_1_CPM_L2', data=relu4_4_CPM , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        relu5_1_CPM_L2 = mx.symbol.Activation(name='relu5_1_CPM_L2', data=conv5_1_CPM_L2 , act_type='relu')
        conv5_2_CPM_L1 = mx.symbol.Convolution(name='conv5_2_CPM_L1', data=relu5_1_CPM_L1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        relu5_2_CPM_L1 = mx.symbol.Activation(name='relu5_2_CPM_L1', data=conv5_2_CPM_L1 , act_type='relu')
        conv5_2_CPM_L2 = mx.symbol.Convolution(name='conv5_2_CPM_L2', data=relu5_1_CPM_L2 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        relu5_2_CPM_L2 = mx.symbol.Activation(name='relu5_2_CPM_L2', data=conv5_2_CPM_L2 , act_type='relu')
        conv5_3_CPM_L1 = mx.symbol.Convolution(name='conv5_3_CPM_L1', data=relu5_2_CPM_L1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        relu5_3_CPM_L1 = mx.symbol.Activation(name='relu5_3_CPM_L1', data=conv5_3_CPM_L1 , act_type='relu')
        conv5_3_CPM_L2 = mx.symbol.Convolution(name='conv5_3_CPM_L2', data=relu5_2_CPM_L2 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        relu5_3_CPM_L2 = mx.symbol.Activation(name='relu5_3_CPM_L2', data=conv5_3_CPM_L2 , act_type='relu')
        conv5_4_CPM_L1 = mx.symbol.Convolution(name='conv5_4_CPM_L1', data=relu5_3_CPM_L1 , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        relu5_4_CPM_L1 = mx.symbol.Activation(name='relu5_4_CPM_L1', data=conv5_4_CPM_L1 , act_type='relu')
        conv5_4_CPM_L2 = mx.symbol.Convolution(name='conv5_4_CPM_L2', data=relu5_3_CPM_L2 , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        relu5_4_CPM_L2 = mx.symbol.Activation(name='relu5_4_CPM_L2', data=conv5_4_CPM_L2 , act_type='relu')
        conv5_5_CPM_L1 = mx.symbol.Convolution(name='conv5_5_CPM_L1', data=relu5_4_CPM_L1 , num_filter=numoflinks*2, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        conv5_5_CPM_L2 = mx.symbol.Convolution(name='conv5_5_CPM_L2', data=relu5_4_CPM_L2 , num_filter=numofparts, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        concat_stage2 = mx.symbol.Concat(name='concat_stage2', *[conv5_5_CPM_L1,conv5_5_CPM_L2,relu4_4_CPM] )
        Mconv1_stage2_L1 = mx.symbol.Convolution(name='Mconv1_stage2_L1', data=concat_stage2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu1_stage2_L1 = mx.symbol.Activation(name='Mrelu1_stage2_L1', data=Mconv1_stage2_L1 , act_type='relu')
        Mconv1_stage2_L2 = mx.symbol.Convolution(name='Mconv1_stage2_L2', data=concat_stage2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu1_stage2_L2 = mx.symbol.Activation(name='Mrelu1_stage2_L2', data=Mconv1_stage2_L2 , act_type='relu')
        Mconv2_stage2_L1 = mx.symbol.Convolution(name='Mconv2_stage2_L1', data=Mrelu1_stage2_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu2_stage2_L1 = mx.symbol.Activation(name='Mrelu2_stage2_L1', data=Mconv2_stage2_L1 , act_type='relu')
        Mconv2_stage2_L2 = mx.symbol.Convolution(name='Mconv2_stage2_L2', data=Mrelu1_stage2_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu2_stage2_L2 = mx.symbol.Activation(name='Mrelu2_stage2_L2', data=Mconv2_stage2_L2 , act_type='relu')
        Mconv3_stage2_L1 = mx.symbol.Convolution(name='Mconv3_stage2_L1', data=Mrelu2_stage2_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu3_stage2_L1 = mx.symbol.Activation(name='Mrelu3_stage2_L1', data=Mconv3_stage2_L1 , act_type='relu')
        Mconv3_stage2_L2 = mx.symbol.Convolution(name='Mconv3_stage2_L2', data=Mrelu2_stage2_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu3_stage2_L2 = mx.symbol.Activation(name='Mrelu3_stage2_L2', data=Mconv3_stage2_L2 , act_type='relu')
        Mconv4_stage2_L1 = mx.symbol.Convolution(name='Mconv4_stage2_L1', data=Mrelu3_stage2_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu4_stage2_L1 = mx.symbol.Activation(name='Mrelu4_stage2_L1', data=Mconv4_stage2_L1 , act_type='relu')
        Mconv4_stage2_L2 = mx.symbol.Convolution(name='Mconv4_stage2_L2', data=Mrelu3_stage2_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu4_stage2_L2 = mx.symbol.Activation(name='Mrelu4_stage2_L2', data=Mconv4_stage2_L2 , act_type='relu')
        Mconv5_stage2_L1 = mx.symbol.Convolution(name='Mconv5_stage2_L1', data=Mrelu4_stage2_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu5_stage2_L1 = mx.symbol.Activation(name='Mrelu5_stage2_L1', data=Mconv5_stage2_L1 , act_type='relu')
        Mconv5_stage2_L2 = mx.symbol.Convolution(name='Mconv5_stage2_L2', data=Mrelu4_stage2_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu5_stage2_L2 = mx.symbol.Activation(name='Mrelu5_stage2_L2', data=Mconv5_stage2_L2 , act_type='relu')
        Mconv6_stage2_L1 = mx.symbol.Convolution(name='Mconv6_stage2_L1', data=Mrelu5_stage2_L1 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mrelu6_stage2_L1 = mx.symbol.Activation(name='Mrelu6_stage2_L1', data=Mconv6_stage2_L1 , act_type='relu')
        Mconv6_stage2_L2 = mx.symbol.Convolution(name='Mconv6_stage2_L2', data=Mrelu5_stage2_L2 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mrelu6_stage2_L2 = mx.symbol.Activation(name='Mrelu6_stage2_L2', data=Mconv6_stage2_L2 , act_type='relu')
        Mconv7_stage2_L1 = mx.symbol.Convolution(name='Mconv7_stage2_L1', data=Mrelu6_stage2_L1 , num_filter=numoflinks*2, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mconv7_stage2_L2 = mx.symbol.Convolution(name='Mconv7_stage2_L2', data=Mrelu6_stage2_L2 , num_filter=numofparts, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        concat_stage3 = mx.symbol.Concat(name='concat_stage3', *[Mconv7_stage2_L1,Mconv7_stage2_L2,relu4_4_CPM] )
        Mconv1_stage3_L1 = mx.symbol.Convolution(name='Mconv1_stage3_L1', data=concat_stage3 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu1_stage3_L1 = mx.symbol.Activation(name='Mrelu1_stage3_L1', data=Mconv1_stage3_L1 , act_type='relu')
        Mconv1_stage3_L2 = mx.symbol.Convolution(name='Mconv1_stage3_L2', data=concat_stage3 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu1_stage3_L2 = mx.symbol.Activation(name='Mrelu1_stage3_L2', data=Mconv1_stage3_L2 , act_type='relu')
        Mconv2_stage3_L1 = mx.symbol.Convolution(name='Mconv2_stage3_L1', data=Mrelu1_stage3_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu2_stage3_L1 = mx.symbol.Activation(name='Mrelu2_stage3_L1', data=Mconv2_stage3_L1 , act_type='relu')
        Mconv2_stage3_L2 = mx.symbol.Convolution(name='Mconv2_stage3_L2', data=Mrelu1_stage3_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu2_stage3_L2 = mx.symbol.Activation(name='Mrelu2_stage3_L2', data=Mconv2_stage3_L2 , act_type='relu')
        Mconv3_stage3_L1 = mx.symbol.Convolution(name='Mconv3_stage3_L1', data=Mrelu2_stage3_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu3_stage3_L1 = mx.symbol.Activation(name='Mrelu3_stage3_L1', data=Mconv3_stage3_L1 , act_type='relu')
        Mconv3_stage3_L2 = mx.symbol.Convolution(name='Mconv3_stage3_L2', data=Mrelu2_stage3_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu3_stage3_L2 = mx.symbol.Activation(name='Mrelu3_stage3_L2', data=Mconv3_stage3_L2 , act_type='relu')
        Mconv4_stage3_L1 = mx.symbol.Convolution(name='Mconv4_stage3_L1', data=Mrelu3_stage3_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu4_stage3_L1 = mx.symbol.Activation(name='Mrelu4_stage3_L1', data=Mconv4_stage3_L1 , act_type='relu')
        Mconv4_stage3_L2 = mx.symbol.Convolution(name='Mconv4_stage3_L2', data=Mrelu3_stage3_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu4_stage3_L2 = mx.symbol.Activation(name='Mrelu4_stage3_L2', data=Mconv4_stage3_L2 , act_type='relu')
        Mconv5_stage3_L1 = mx.symbol.Convolution(name='Mconv5_stage3_L1', data=Mrelu4_stage3_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu5_stage3_L1 = mx.symbol.Activation(name='Mrelu5_stage3_L1', data=Mconv5_stage3_L1 , act_type='relu')
        Mconv5_stage3_L2 = mx.symbol.Convolution(name='Mconv5_stage3_L2', data=Mrelu4_stage3_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu5_stage3_L2 = mx.symbol.Activation(name='Mrelu5_stage3_L2', data=Mconv5_stage3_L2 , act_type='relu')
        Mconv6_stage3_L1 = mx.symbol.Convolution(name='Mconv6_stage3_L1', data=Mrelu5_stage3_L1 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mrelu6_stage3_L1 = mx.symbol.Activation(name='Mrelu6_stage3_L1', data=Mconv6_stage3_L1 , act_type='relu')
        Mconv6_stage3_L2 = mx.symbol.Convolution(name='Mconv6_stage3_L2', data=Mrelu5_stage3_L2 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mrelu6_stage3_L2 = mx.symbol.Activation(name='Mrelu6_stage3_L2', data=Mconv6_stage3_L2 , act_type='relu')
        Mconv7_stage3_L1 = mx.symbol.Convolution(name='Mconv7_stage3_L1', data=Mrelu6_stage3_L1 , num_filter=numoflinks*2, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mconv7_stage3_L2 = mx.symbol.Convolution(name='Mconv7_stage3_L2', data=Mrelu6_stage3_L2 , num_filter=numofparts, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        concat_stage4 = mx.symbol.Concat(name='concat_stage4', *[Mconv7_stage3_L1,Mconv7_stage3_L2,relu4_4_CPM] )
        Mconv1_stage4_L1 = mx.symbol.Convolution(name='Mconv1_stage4_L1', data=concat_stage4 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu1_stage4_L1 = mx.symbol.Activation(name='Mrelu1_stage4_L1', data=Mconv1_stage4_L1 , act_type='relu')
        Mconv1_stage4_L2 = mx.symbol.Convolution(name='Mconv1_stage4_L2', data=concat_stage4 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu1_stage4_L2 = mx.symbol.Activation(name='Mrelu1_stage4_L2', data=Mconv1_stage4_L2 , act_type='relu')
        Mconv2_stage4_L1 = mx.symbol.Convolution(name='Mconv2_stage4_L1', data=Mrelu1_stage4_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu2_stage4_L1 = mx.symbol.Activation(name='Mrelu2_stage4_L1', data=Mconv2_stage4_L1 , act_type='relu')
        Mconv2_stage4_L2 = mx.symbol.Convolution(name='Mconv2_stage4_L2', data=Mrelu1_stage4_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu2_stage4_L2 = mx.symbol.Activation(name='Mrelu2_stage4_L2', data=Mconv2_stage4_L2 , act_type='relu')
        Mconv3_stage4_L1 = mx.symbol.Convolution(name='Mconv3_stage4_L1', data=Mrelu2_stage4_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu3_stage4_L1 = mx.symbol.Activation(name='Mrelu3_stage4_L1', data=Mconv3_stage4_L1 , act_type='relu')
        Mconv3_stage4_L2 = mx.symbol.Convolution(name='Mconv3_stage4_L2', data=Mrelu2_stage4_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu3_stage4_L2 = mx.symbol.Activation(name='Mrelu3_stage4_L2', data=Mconv3_stage4_L2 , act_type='relu')
        Mconv4_stage4_L1 = mx.symbol.Convolution(name='Mconv4_stage4_L1', data=Mrelu3_stage4_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu4_stage4_L1 = mx.symbol.Activation(name='Mrelu4_stage4_L1', data=Mconv4_stage4_L1 , act_type='relu')
        Mconv4_stage4_L2 = mx.symbol.Convolution(name='Mconv4_stage4_L2', data=Mrelu3_stage4_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu4_stage4_L2 = mx.symbol.Activation(name='Mrelu4_stage4_L2', data=Mconv4_stage4_L2 , act_type='relu')
        Mconv5_stage4_L1 = mx.symbol.Convolution(name='Mconv5_stage4_L1', data=Mrelu4_stage4_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu5_stage4_L1 = mx.symbol.Activation(name='Mrelu5_stage4_L1', data=Mconv5_stage4_L1 , act_type='relu')
        Mconv5_stage4_L2 = mx.symbol.Convolution(name='Mconv5_stage4_L2', data=Mrelu4_stage4_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu5_stage4_L2 = mx.symbol.Activation(name='Mrelu5_stage4_L2', data=Mconv5_stage4_L2 , act_type='relu')
        Mconv6_stage4_L1 = mx.symbol.Convolution(name='Mconv6_stage4_L1', data=Mrelu5_stage4_L1 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mrelu6_stage4_L1 = mx.symbol.Activation(name='Mrelu6_stage4_L1', data=Mconv6_stage4_L1 , act_type='relu')
        Mconv6_stage4_L2 = mx.symbol.Convolution(name='Mconv6_stage4_L2', data=Mrelu5_stage4_L2 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mrelu6_stage4_L2 = mx.symbol.Activation(name='Mrelu6_stage4_L2', data=Mconv6_stage4_L2 , act_type='relu')
        Mconv7_stage4_L1 = mx.symbol.Convolution(name='Mconv7_stage4_L1', data=Mrelu6_stage4_L1 , num_filter=numoflinks*2, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mconv7_stage4_L2 = mx.symbol.Convolution(name='Mconv7_stage4_L2', data=Mrelu6_stage4_L2 , num_filter=numofparts, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        concat_stage5 = mx.symbol.Concat(name='concat_stage5', *[Mconv7_stage4_L1,Mconv7_stage4_L2,relu4_4_CPM] )
        Mconv1_stage5_L1 = mx.symbol.Convolution(name='Mconv1_stage5_L1', data=concat_stage5 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu1_stage5_L1 = mx.symbol.Activation(name='Mrelu1_stage5_L1', data=Mconv1_stage5_L1 , act_type='relu')
        Mconv1_stage5_L2 = mx.symbol.Convolution(name='Mconv1_stage5_L2', data=concat_stage5 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu1_stage5_L2 = mx.symbol.Activation(name='Mrelu1_stage5_L2', data=Mconv1_stage5_L2 , act_type='relu')
        Mconv2_stage5_L1 = mx.symbol.Convolution(name='Mconv2_stage5_L1', data=Mrelu1_stage5_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu2_stage5_L1 = mx.symbol.Activation(name='Mrelu2_stage5_L1', data=Mconv2_stage5_L1 , act_type='relu')
        Mconv2_stage5_L2 = mx.symbol.Convolution(name='Mconv2_stage5_L2', data=Mrelu1_stage5_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu2_stage5_L2 = mx.symbol.Activation(name='Mrelu2_stage5_L2', data=Mconv2_stage5_L2 , act_type='relu')
        Mconv3_stage5_L1 = mx.symbol.Convolution(name='Mconv3_stage5_L1', data=Mrelu2_stage5_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu3_stage5_L1 = mx.symbol.Activation(name='Mrelu3_stage5_L1', data=Mconv3_stage5_L1 , act_type='relu')
        Mconv3_stage5_L2 = mx.symbol.Convolution(name='Mconv3_stage5_L2', data=Mrelu2_stage5_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu3_stage5_L2 = mx.symbol.Activation(name='Mrelu3_stage5_L2', data=Mconv3_stage5_L2 , act_type='relu')
        Mconv4_stage5_L1 = mx.symbol.Convolution(name='Mconv4_stage5_L1', data=Mrelu3_stage5_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu4_stage5_L1 = mx.symbol.Activation(name='Mrelu4_stage5_L1', data=Mconv4_stage5_L1 , act_type='relu')
        Mconv4_stage5_L2 = mx.symbol.Convolution(name='Mconv4_stage5_L2', data=Mrelu3_stage5_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu4_stage5_L2 = mx.symbol.Activation(name='Mrelu4_stage5_L2', data=Mconv4_stage5_L2 , act_type='relu')
        Mconv5_stage5_L1 = mx.symbol.Convolution(name='Mconv5_stage5_L1', data=Mrelu4_stage5_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu5_stage5_L1 = mx.symbol.Activation(name='Mrelu5_stage5_L1', data=Mconv5_stage5_L1 , act_type='relu')
        Mconv5_stage5_L2 = mx.symbol.Convolution(name='Mconv5_stage5_L2', data=Mrelu4_stage5_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu5_stage5_L2 = mx.symbol.Activation(name='Mrelu5_stage5_L2', data=Mconv5_stage5_L2 , act_type='relu')
        Mconv6_stage5_L1 = mx.symbol.Convolution(name='Mconv6_stage5_L1', data=Mrelu5_stage5_L1 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mrelu6_stage5_L1 = mx.symbol.Activation(name='Mrelu6_stage5_L1', data=Mconv6_stage5_L1 , act_type='relu')
        Mconv6_stage5_L2 = mx.symbol.Convolution(name='Mconv6_stage5_L2', data=Mrelu5_stage5_L2 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mrelu6_stage5_L2 = mx.symbol.Activation(name='Mrelu6_stage5_L2', data=Mconv6_stage5_L2 , act_type='relu')
        Mconv7_stage5_L1 = mx.symbol.Convolution(name='Mconv7_stage5_L1', data=Mrelu6_stage5_L1 , num_filter=numoflinks*2, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mconv7_stage5_L2 = mx.symbol.Convolution(name='Mconv7_stage5_L2', data=Mrelu6_stage5_L2 , num_filter=numofparts, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        concat_stage6 = mx.symbol.Concat(name='concat_stage6', *[Mconv7_stage5_L1,Mconv7_stage5_L2,relu4_4_CPM] )
        Mconv1_stage6_L1 = mx.symbol.Convolution(name='Mconv1_stage6_L1', data=concat_stage6 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu1_stage6_L1 = mx.symbol.Activation(name='Mrelu1_stage6_L1', data=Mconv1_stage6_L1 , act_type='relu')
        Mconv1_stage6_L2 = mx.symbol.Convolution(name='Mconv1_stage6_L2', data=concat_stage6 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu1_stage6_L2 = mx.symbol.Activation(name='Mrelu1_stage6_L2', data=Mconv1_stage6_L2 , act_type='relu')
        Mconv2_stage6_L1 = mx.symbol.Convolution(name='Mconv2_stage6_L1', data=Mrelu1_stage6_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu2_stage6_L1 = mx.symbol.Activation(name='Mrelu2_stage6_L1', data=Mconv2_stage6_L1 , act_type='relu')
        Mconv2_stage6_L2 = mx.symbol.Convolution(name='Mconv2_stage6_L2', data=Mrelu1_stage6_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu2_stage6_L2 = mx.symbol.Activation(name='Mrelu2_stage6_L2', data=Mconv2_stage6_L2 , act_type='relu')
        Mconv3_stage6_L1 = mx.symbol.Convolution(name='Mconv3_stage6_L1', data=Mrelu2_stage6_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu3_stage6_L1 = mx.symbol.Activation(name='Mrelu3_stage6_L1', data=Mconv3_stage6_L1 , act_type='relu')
        Mconv3_stage6_L2 = mx.symbol.Convolution(name='Mconv3_stage6_L2', data=Mrelu2_stage6_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu3_stage6_L2 = mx.symbol.Activation(name='Mrelu3_stage6_L2', data=Mconv3_stage6_L2 , act_type='relu')
        Mconv4_stage6_L1 = mx.symbol.Convolution(name='Mconv4_stage6_L1', data=Mrelu3_stage6_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu4_stage6_L1 = mx.symbol.Activation(name='Mrelu4_stage6_L1', data=Mconv4_stage6_L1 , act_type='relu')
        Mconv4_stage6_L2 = mx.symbol.Convolution(name='Mconv4_stage6_L2', data=Mrelu3_stage6_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu4_stage6_L2 = mx.symbol.Activation(name='Mrelu4_stage6_L2', data=Mconv4_stage6_L2 , act_type='relu')
        Mconv5_stage6_L1 = mx.symbol.Convolution(name='Mconv5_stage6_L1', data=Mrelu4_stage6_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu5_stage6_L1 = mx.symbol.Activation(name='Mrelu5_stage6_L1', data=Mconv5_stage6_L1 , act_type='relu')
        Mconv5_stage6_L2 = mx.symbol.Convolution(name='Mconv5_stage6_L2', data=Mrelu4_stage6_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
        Mrelu5_stage6_L2 = mx.symbol.Activation(name='Mrelu5_stage6_L2', data=Mconv5_stage6_L2 , act_type='relu')
        Mconv6_stage6_L1 = mx.symbol.Convolution(name='Mconv6_stage6_L1', data=Mrelu5_stage6_L1 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mrelu6_stage6_L1 = mx.symbol.Activation(name='Mrelu6_stage6_L1', data=Mconv6_stage6_L1 , act_type='relu')
        Mconv6_stage6_L2 = mx.symbol.Convolution(name='Mconv6_stage6_L2', data=Mrelu5_stage6_L2 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mrelu6_stage6_L2 = mx.symbol.Activation(name='Mrelu6_stage6_L2', data=Mconv6_stage6_L2 , act_type='relu')
        Mconv7_stage6_L1 = mx.symbol.Convolution(name='Mconv7_stage6_L1', data=Mrelu6_stage6_L1 , num_filter=numoflinks*2, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
        Mconv7_stage6_L2 = mx.symbol.Convolution(name='Mconv7_stage6_L2', data=Mrelu6_stage6_L2 , num_filter=numofparts, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    
    
        conv5_5_CPM_L1r = mx.symbol.Reshape(data=conv5_5_CPM_L1, shape=(-1,), name='conv5_5_CPM_L1r')
        partaffinityglabelr = mx.symbol.Reshape(data=partaffinityglabel, shape=(-1, ), name='partaffinityglabelr')
        stage1_loss_L1s = mx.symbol.square(conv5_5_CPM_L1r-partaffinityglabelr)
        vecweightw = mx.symbol.Reshape(data=vecweight, shape=(-1,), name='conv5_5_CPM_L1w')
        stage1_loss_L1w = stage1_loss_L1s*vecweightw
        stage1_loss_L1  = mx.symbol.MakeLoss(stage1_loss_L1w)
        
        conv5_5_CPM_L2r = mx.symbol.Reshape(data=conv5_5_CPM_L2, shape=(-1,), name='conv5_5_CPM_L2r')
        heatmaplabelr = mx.symbol.Reshape(data=heatmaplabel, shape=(-1, ), name='heatmaplabelr')
        stage1_loss_L2s = mx.symbol.square(conv5_5_CPM_L2r-heatmaplabelr)
        heatweightw = mx.symbol.Reshape(data=heatweight, shape=(-1,), name='conv5_5_CPM_L2w')
        stage1_loss_L2w = stage1_loss_L2s*heatweightw
        stage1_loss_L2  = mx.symbol.MakeLoss(stage1_loss_L2w)
            
        Mconv7_stage2_L1r = mx.symbol.Reshape(data=Mconv7_stage2_L1, shape=(-1,), name='Mconv7_stage2_L1')
        #partaffinityglabelr = mx.symbol.Reshape(data=partaffinityglabel, shape=(-1, ), name='partaffinityglabelr')
        stage2_loss_L1s = mx.symbol.square(Mconv7_stage2_L1r - partaffinityglabelr)
        #vecweightw = mx.symbol.Reshape(data=vecweight, shape=(-1,), name='Mconv7_stage2_L1r')
        stage2_loss_L1w = stage2_loss_L1s*vecweightw
        stage2_loss_L1  = mx.symbol.MakeLoss(stage2_loss_L1w)
        
        Mconv7_stage2_L2r = mx.symbol.Reshape(data=Mconv7_stage2_L2, shape=(-1,), name='Mconv7_stage2_L2')
        #heatmaplabelr = mx.symbol.Reshape(data=heatmaplabel, shape=(-1, ), name='heatmaplabelr')
        stage2_loss_L2s = mx.symbol.square(Mconv7_stage2_L2r-heatmaplabelr)
        #heatweightw = mx.symbol.Reshape(data=heatweight, shape=(-1,), name='conv5_5_CPM_L1r')
        stage2_loss_L2w = stage1_loss_L2s*heatweightw
        stage2_loss_L2  = mx.symbol.MakeLoss(stage2_loss_L2w)
        
        
        Mconv7_stage3_L1r = mx.symbol.Reshape(data=Mconv7_stage3_L1, shape=(-1,), name='Mconv7_stage3_L1')
        #partaffinityglabelr = mx.symbol.Reshape(data=partaffinityglabel, shape=(-1, ), name='partaffinityglabelr')
        stage3_loss_L1s = mx.symbol.square(Mconv7_stage3_L1r - partaffinityglabelr)
        #vecweightw = mx.symbol.Reshape(data=vecweight, shape=(-1,), name='Mconv7_stage2_L1r')
        stage3_loss_L1w = stage3_loss_L1s*vecweightw
        stage3_loss_L1  = mx.symbol.MakeLoss(stage3_loss_L1w)
        
        Mconv7_stage3_L2r = mx.symbol.Reshape(data=Mconv7_stage3_L2, shape=(-1,), name='Mconv7_stage3_L2')
        #heatmaplabelr = mx.symbol.Reshape(data=heatmaplabel, shape=(-1, ), name='heatmaplabelr')
        stage3_loss_L2s = mx.symbol.square(Mconv7_stage3_L2r-heatmaplabelr)
        #heatweightw = mx.symbol.Reshape(data=heatweight, shape=(-1,), name='conv5_5_CPM_L1r')
        stage3_loss_L2w = stage3_loss_L2s*heatweightw
        stage3_loss_L2  = mx.symbol.MakeLoss(stage3_loss_L2w)
        
        Mconv7_stage4_L1r = mx.symbol.Reshape(data=Mconv7_stage4_L1, shape=(-1,), name='Mconv7_stage4_L1')
        #partaffinityglabelr = mx.symbol.Reshape(data=partaffinityglabel, shape=(-1, ), name='partaffinityglabelr')
        stage4_loss_L1s = mx.symbol.square(Mconv7_stage4_L1r - partaffinityglabelr)
        #vecweightw = mx.symbol.Reshape(data=vecweight, shape=(-1,), name='Mconv7_stage2_L1r')
        stage4_loss_L1w = stage4_loss_L1s*vecweightw
        stage4_loss_L1  = mx.symbol.MakeLoss(stage4_loss_L1w)
        
        Mconv7_stage4_L2r = mx.symbol.Reshape(data=Mconv7_stage4_L2, shape=(-1,), name='Mconv7_stage4_L2')
        #heatmaplabelr = mx.symbol.Reshape(data=heatmaplabel, shape=(-1, ), name='heatmaplabelr')
        stage4_loss_L2s = mx.symbol.square(Mconv7_stage4_L2r-heatmaplabelr)
        #heatweightw = mx.symbol.Reshape(data=heatweight, shape=(-1,), name='conv5_5_CPM_L1r')
        stage4_loss_L2w = stage1_loss_L2s*heatweightw
        stage4_loss_L2  = mx.symbol.MakeLoss(stage4_loss_L2w)
        
        Mconv7_stage5_L1r = mx.symbol.Reshape(data=Mconv7_stage5_L1, shape=(-1,), name='Mconv7_stage5_L1')
        #partaffinityglabelr = mx.symbol.Reshape(data=partaffinityglabel, shape=(-1, ), name='partaffinityglabelr')
        stage5_loss_L1s = mx.symbol.square(Mconv7_stage5_L1r - partaffinityglabelr)
        #vecweightw = mx.symbol.Reshape(data=vecweight, shape=(-1,), name='Mconv7_stage2_L1r')
        stage5_loss_L1w = stage5_loss_L1s*vecweightw
        stage5_loss_L1  = mx.symbol.MakeLoss(stage5_loss_L1w)
        
        Mconv7_stage5_L2r = mx.symbol.Reshape(data=Mconv7_stage5_L2, shape=(-1,), name='Mconv7_stage5_L2')
        #heatmaplabelr = mx.symbol.Reshape(data=heatmaplabel, shape=(-1, ), name='heatmaplabelr')
        stage5_loss_L2s = mx.symbol.square(Mconv7_stage5_L2r-heatmaplabelr)
        #heatweightw = mx.symbol.Reshape(data=heatweight, shape=(-1,), name='conv5_5_CPM_L1r')
        stage5_loss_L2w = stage5_loss_L2s*heatweightw
        stage5_loss_L2  = mx.symbol.MakeLoss(stage5_loss_L2w)
        
        
        Mconv7_stage6_L1r = mx.symbol.Reshape(data=Mconv7_stage6_L1, shape=(-1,), name='Mconv7_stage3_L1')
        #partaffinityglabelr = mx.symbol.Reshape(data=partaffinityglabel, shape=(-1, ), name='partaffinityglabelr')
        stage6_loss_L1s = mx.symbol.square(Mconv7_stage6_L1r - partaffinityglabelr)
        #vecweightw = mx.symbol.Reshape(data=vecweight, shape=(-1,), name='Mconv7_stage2_L1r')
        stage6_loss_L1w = stage6_loss_L1s*vecweightw
        stage6_loss_L1  = mx.symbol.MakeLoss(stage6_loss_L1w)
        
        Mconv7_stage6_L2r = mx.symbol.Reshape(data=Mconv7_stage6_L2, shape=(-1,), name='Mconv7_stage3_L2')
        #heatmaplabelr = mx.symbol.Reshape(data=heatmaplabel, shape=(-1, ), name='heatmaplabelr')
        stage6_loss_L2s = mx.symbol.square(Mconv7_stage6_L2r-heatmaplabelr)
        #heatweightw = mx.symbol.Reshape(data=heatweight, shape=(-1,), name='conv5_5_CPM_L1r')
        stage6_loss_L2w = stage6_loss_L2s*heatweightw
        stage6_loss_L2  = mx.symbol.MakeLoss(stage6_loss_L2w)
        
        group = mx.symbol.Group([stage1_loss_L1, stage1_loss_L2,
                                 stage2_loss_L1, stage2_loss_L2,
                                 stage3_loss_L1, stage3_loss_L2,
                                 stage4_loss_L1, stage4_loss_L2,
                                 stage5_loss_L1, stage5_loss_L2,
                                 stage6_loss_L1, stage6_loss_L2])
        return group
            
        
    ######## To be debugged
            
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

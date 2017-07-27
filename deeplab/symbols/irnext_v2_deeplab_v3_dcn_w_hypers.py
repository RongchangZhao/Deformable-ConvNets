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
from utils.symbol import Symbol


###### UNIT LIST #######

# Todo 1,2,4

def irnext_unit(data, num_filter, stride, dim_match, name, bottle_neck=1, expansion=0.5, \
                 num_group=32, dilation=1, irv2 = False, deform=0, bn_mom=0.9, workspace=256, memonger=False):
    
    """
    Return ResNet Unit symbol for building ResNeXt
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
        
        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), stride=stride, 
                                   pad=(dilation,dilation), dilate=(dilation,dilation), 
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

        
        conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1),
                                   pad=(dilation,dilation), dilate=(dilation,dilation)
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')

        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
            
        eltwise = bn2 + shortcut
        
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')
    
    # If 1: Use ResNeXt Conv1,1-Conv3,3-Conv1,1 
    elif bottle_neck == 1:
        
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        
        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter/expansion), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

        
        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter/expansion), 
                                   num_group=num_group, kernel=(3,3), stride=stride, 
                                   pad=(dilation,dilation), dilate=(dilation,dilation),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')

        
        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
        eltwise =  bn3 + shortcut
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')
    
    
    elif bottle_neck == 2:
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
         # TODO
        raise Exception("bottle_neck error: Unimplemented Bottleneck Unit: Dual Path Net.")
         
    else:
        raise Exception("bottle_neck error: Unrecognized Bottleneck params.")


# Todo 0 & 3 .
    


class resnet_v1_101_deeplab_dcn(Symbol):
    
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 4096
        self.units = (3, 4, 23, 3) # use for 101
        self.filter_list = [256, 512, 1024, 2048]

    def get_resnet_conv(self, data):
        conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
                                      no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=True, fix_gamma=False, eps = self.eps)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                                  stride=(2, 2), pool_type='max')
        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1, num_filter=256, pad=(0, 0),
                                              kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps = self.eps)
        scale2a_branch1 = bn2a_branch1
        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale2a_branch2a = bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a, act_type='relu')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu, num_filter=64,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale2a_branch2b = bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b, act_type='relu')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu, num_filter=256,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale2a_branch2c = bn2a_branch2c
        res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1, scale2a_branch2c])
        res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a, act_type='relu')
        res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale2b_branch2a = bn2b_branch2a
        res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a, act_type='relu')
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu, num_filter=64,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale2b_branch2b = bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b, act_type='relu')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu, num_filter=256,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale2b_branch2c = bn2b_branch2c
        res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu, scale2b_branch2c])
        res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b, act_type='relu')
        res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale2c_branch2a = bn2c_branch2a
        res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a, act_type='relu')
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu, num_filter=64,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale2c_branch2b = bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b, act_type='relu')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu, num_filter=256,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale2c_branch2c = bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu, scale2c_branch2c])
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c, act_type='relu')
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu, num_filter=512, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps = self.eps)
        scale3a_branch1 = bn3a_branch1
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale3a_branch2a = bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a, act_type='relu')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu, num_filter=128,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale3a_branch2b = bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b, act_type='relu')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu, num_filter=512,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale3a_branch2c = bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1, scale3a_branch2c])
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a, act_type='relu')
        res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2a = mx.symbol.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale3b1_branch2a = bn3b1_branch2a
        res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=scale3b1_branch2a,
                                                    act_type='relu')
        res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b1_branch2b = mx.symbol.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale3b1_branch2b = bn3b1_branch2b
        res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=scale3b1_branch2b,
                                                    act_type='relu')
        res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2c = mx.symbol.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale3b1_branch2c = bn3b1_branch2c
        res3b1 = mx.symbol.broadcast_add(name='res3b1', *[res3a_relu, scale3b1_branch2c])
        res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1, act_type='relu')
        res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2a = mx.symbol.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale3b2_branch2a = bn3b2_branch2a
        res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=scale3b2_branch2a,
                                                    act_type='relu')
        res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b2_branch2b = mx.symbol.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale3b2_branch2b = bn3b2_branch2b
        res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=scale3b2_branch2b,
                                                    act_type='relu')
        res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2c = mx.symbol.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale3b2_branch2c = bn3b2_branch2c
        res3b2 = mx.symbol.broadcast_add(name='res3b2', *[res3b1_relu, scale3b2_branch2c])
        res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2, act_type='relu')
        res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2a = mx.symbol.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale3b3_branch2a = bn3b3_branch2a
        res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=scale3b3_branch2a,
                                                    act_type='relu')
        res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b3_branch2b = mx.symbol.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale3b3_branch2b = bn3b3_branch2b
        res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=scale3b3_branch2b,
                                                    act_type='relu')
        res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2c = mx.symbol.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale3b3_branch2c = bn3b3_branch2c
        res3b3 = mx.symbol.broadcast_add(name='res3b3', *[res3b2_relu, scale3b3_branch2c])
        res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3, act_type='relu')
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3b3_relu, num_filter=1024, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps = self.eps)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3b3_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu, num_filter=256,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps = self.eps)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1, scale4a_branch2c])
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a, act_type='relu')
        res4b1_branch2a = mx.symbol.Convolution(name='res4b1_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2a = mx.symbol.BatchNorm(name='bn4b1_branch2a', data=res4b1_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b1_branch2a = bn4b1_branch2a
        res4b1_branch2a_relu = mx.symbol.Activation(name='res4b1_branch2a_relu', data=scale4b1_branch2a,
                                                    act_type='relu')
        res4b1_branch2b = mx.symbol.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b1_branch2b = mx.symbol.BatchNorm(name='bn4b1_branch2b', data=res4b1_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b1_branch2b = bn4b1_branch2b
        res4b1_branch2b_relu = mx.symbol.Activation(name='res4b1_branch2b_relu', data=scale4b1_branch2b,
                                                    act_type='relu')
        res4b1_branch2c = mx.symbol.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2c = mx.symbol.BatchNorm(name='bn4b1_branch2c', data=res4b1_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b1_branch2c = bn4b1_branch2c
        res4b1 = mx.symbol.broadcast_add(name='res4b1', *[res4a_relu, scale4b1_branch2c])
        res4b1_relu = mx.symbol.Activation(name='res4b1_relu', data=res4b1, act_type='relu')
        res4b2_branch2a = mx.symbol.Convolution(name='res4b2_branch2a', data=res4b1_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2a = mx.symbol.BatchNorm(name='bn4b2_branch2a', data=res4b2_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b2_branch2a = bn4b2_branch2a
        res4b2_branch2a_relu = mx.symbol.Activation(name='res4b2_branch2a_relu', data=scale4b2_branch2a,
                                                    act_type='relu')
        res4b2_branch2b = mx.symbol.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b2_branch2b = mx.symbol.BatchNorm(name='bn4b2_branch2b', data=res4b2_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b2_branch2b = bn4b2_branch2b
        res4b2_branch2b_relu = mx.symbol.Activation(name='res4b2_branch2b_relu', data=scale4b2_branch2b,
                                                    act_type='relu')
        res4b2_branch2c = mx.symbol.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2c = mx.symbol.BatchNorm(name='bn4b2_branch2c', data=res4b2_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b2_branch2c = bn4b2_branch2c
        res4b2 = mx.symbol.broadcast_add(name='res4b2', *[res4b1_relu, scale4b2_branch2c])
        res4b2_relu = mx.symbol.Activation(name='res4b2_relu', data=res4b2, act_type='relu')
        res4b3_branch2a = mx.symbol.Convolution(name='res4b3_branch2a', data=res4b2_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2a = mx.symbol.BatchNorm(name='bn4b3_branch2a', data=res4b3_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b3_branch2a = bn4b3_branch2a
        res4b3_branch2a_relu = mx.symbol.Activation(name='res4b3_branch2a_relu', data=scale4b3_branch2a,
                                                    act_type='relu')
        res4b3_branch2b = mx.symbol.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b3_branch2b = mx.symbol.BatchNorm(name='bn4b3_branch2b', data=res4b3_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b3_branch2b = bn4b3_branch2b
        res4b3_branch2b_relu = mx.symbol.Activation(name='res4b3_branch2b_relu', data=scale4b3_branch2b,
                                                    act_type='relu')
        res4b3_branch2c = mx.symbol.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2c = mx.symbol.BatchNorm(name='bn4b3_branch2c', data=res4b3_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b3_branch2c = bn4b3_branch2c
        res4b3 = mx.symbol.broadcast_add(name='res4b3', *[res4b2_relu, scale4b3_branch2c])
        res4b3_relu = mx.symbol.Activation(name='res4b3_relu', data=res4b3, act_type='relu')
        res4b4_branch2a = mx.symbol.Convolution(name='res4b4_branch2a', data=res4b3_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2a = mx.symbol.BatchNorm(name='bn4b4_branch2a', data=res4b4_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b4_branch2a = bn4b4_branch2a
        res4b4_branch2a_relu = mx.symbol.Activation(name='res4b4_branch2a_relu', data=scale4b4_branch2a,
                                                    act_type='relu')
        res4b4_branch2b = mx.symbol.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b4_branch2b = mx.symbol.BatchNorm(name='bn4b4_branch2b', data=res4b4_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b4_branch2b = bn4b4_branch2b
        res4b4_branch2b_relu = mx.symbol.Activation(name='res4b4_branch2b_relu', data=scale4b4_branch2b,
                                                    act_type='relu')
        res4b4_branch2c = mx.symbol.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2c = mx.symbol.BatchNorm(name='bn4b4_branch2c', data=res4b4_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b4_branch2c = bn4b4_branch2c
        res4b4 = mx.symbol.broadcast_add(name='res4b4', *[res4b3_relu, scale4b4_branch2c])
        res4b4_relu = mx.symbol.Activation(name='res4b4_relu', data=res4b4, act_type='relu')
        res4b5_branch2a = mx.symbol.Convolution(name='res4b5_branch2a', data=res4b4_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2a = mx.symbol.BatchNorm(name='bn4b5_branch2a', data=res4b5_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b5_branch2a = bn4b5_branch2a
        res4b5_branch2a_relu = mx.symbol.Activation(name='res4b5_branch2a_relu', data=scale4b5_branch2a,
                                                    act_type='relu')
        res4b5_branch2b = mx.symbol.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b5_branch2b = mx.symbol.BatchNorm(name='bn4b5_branch2b', data=res4b5_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b5_branch2b = bn4b5_branch2b
        res4b5_branch2b_relu = mx.symbol.Activation(name='res4b5_branch2b_relu', data=scale4b5_branch2b,
                                                    act_type='relu')
        res4b5_branch2c = mx.symbol.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2c = mx.symbol.BatchNorm(name='bn4b5_branch2c', data=res4b5_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b5_branch2c = bn4b5_branch2c
        res4b5 = mx.symbol.broadcast_add(name='res4b5', *[res4b4_relu, scale4b5_branch2c])
        res4b5_relu = mx.symbol.Activation(name='res4b5_relu', data=res4b5, act_type='relu')
        res4b6_branch2a = mx.symbol.Convolution(name='res4b6_branch2a', data=res4b5_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2a = mx.symbol.BatchNorm(name='bn4b6_branch2a', data=res4b6_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b6_branch2a = bn4b6_branch2a
        res4b6_branch2a_relu = mx.symbol.Activation(name='res4b6_branch2a_relu', data=scale4b6_branch2a,
                                                    act_type='relu')
        res4b6_branch2b = mx.symbol.Convolution(name='res4b6_branch2b', data=res4b6_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b6_branch2b = mx.symbol.BatchNorm(name='bn4b6_branch2b', data=res4b6_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b6_branch2b = bn4b6_branch2b
        res4b6_branch2b_relu = mx.symbol.Activation(name='res4b6_branch2b_relu', data=scale4b6_branch2b,
                                                    act_type='relu')
        res4b6_branch2c = mx.symbol.Convolution(name='res4b6_branch2c', data=res4b6_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2c = mx.symbol.BatchNorm(name='bn4b6_branch2c', data=res4b6_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b6_branch2c = bn4b6_branch2c
        res4b6 = mx.symbol.broadcast_add(name='res4b6', *[res4b5_relu, scale4b6_branch2c])
        res4b6_relu = mx.symbol.Activation(name='res4b6_relu', data=res4b6, act_type='relu')
        res4b7_branch2a = mx.symbol.Convolution(name='res4b7_branch2a', data=res4b6_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2a = mx.symbol.BatchNorm(name='bn4b7_branch2a', data=res4b7_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b7_branch2a = bn4b7_branch2a
        res4b7_branch2a_relu = mx.symbol.Activation(name='res4b7_branch2a_relu', data=scale4b7_branch2a,
                                                    act_type='relu')
        res4b7_branch2b = mx.symbol.Convolution(name='res4b7_branch2b', data=res4b7_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b7_branch2b = mx.symbol.BatchNorm(name='bn4b7_branch2b', data=res4b7_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b7_branch2b = bn4b7_branch2b
        res4b7_branch2b_relu = mx.symbol.Activation(name='res4b7_branch2b_relu', data=scale4b7_branch2b,
                                                    act_type='relu')
        res4b7_branch2c = mx.symbol.Convolution(name='res4b7_branch2c', data=res4b7_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2c = mx.symbol.BatchNorm(name='bn4b7_branch2c', data=res4b7_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b7_branch2c = bn4b7_branch2c
        res4b7 = mx.symbol.broadcast_add(name='res4b7', *[res4b6_relu, scale4b7_branch2c])
        res4b7_relu = mx.symbol.Activation(name='res4b7_relu', data=res4b7, act_type='relu')
        res4b8_branch2a = mx.symbol.Convolution(name='res4b8_branch2a', data=res4b7_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2a = mx.symbol.BatchNorm(name='bn4b8_branch2a', data=res4b8_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b8_branch2a = bn4b8_branch2a
        res4b8_branch2a_relu = mx.symbol.Activation(name='res4b8_branch2a_relu', data=scale4b8_branch2a,
                                                    act_type='relu')
        res4b8_branch2b = mx.symbol.Convolution(name='res4b8_branch2b', data=res4b8_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b8_branch2b = mx.symbol.BatchNorm(name='bn4b8_branch2b', data=res4b8_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b8_branch2b = bn4b8_branch2b
        res4b8_branch2b_relu = mx.symbol.Activation(name='res4b8_branch2b_relu', data=scale4b8_branch2b,
                                                    act_type='relu')
        res4b8_branch2c = mx.symbol.Convolution(name='res4b8_branch2c', data=res4b8_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2c = mx.symbol.BatchNorm(name='bn4b8_branch2c', data=res4b8_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b8_branch2c = bn4b8_branch2c
        res4b8 = mx.symbol.broadcast_add(name='res4b8', *[res4b7_relu, scale4b8_branch2c])
        res4b8_relu = mx.symbol.Activation(name='res4b8_relu', data=res4b8, act_type='relu')
        res4b9_branch2a = mx.symbol.Convolution(name='res4b9_branch2a', data=res4b8_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2a = mx.symbol.BatchNorm(name='bn4b9_branch2a', data=res4b9_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b9_branch2a = bn4b9_branch2a
        res4b9_branch2a_relu = mx.symbol.Activation(name='res4b9_branch2a_relu', data=scale4b9_branch2a,
                                                    act_type='relu')
        res4b9_branch2b = mx.symbol.Convolution(name='res4b9_branch2b', data=res4b9_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b9_branch2b = mx.symbol.BatchNorm(name='bn4b9_branch2b', data=res4b9_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b9_branch2b = bn4b9_branch2b
        res4b9_branch2b_relu = mx.symbol.Activation(name='res4b9_branch2b_relu', data=scale4b9_branch2b,
                                                    act_type='relu')
        res4b9_branch2c = mx.symbol.Convolution(name='res4b9_branch2c', data=res4b9_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2c = mx.symbol.BatchNorm(name='bn4b9_branch2c', data=res4b9_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps = self.eps)
        scale4b9_branch2c = bn4b9_branch2c
        res4b9 = mx.symbol.broadcast_add(name='res4b9', *[res4b8_relu, scale4b9_branch2c])
        res4b9_relu = mx.symbol.Activation(name='res4b9_relu', data=res4b9, act_type='relu')
        res4b10_branch2a = mx.symbol.Convolution(name='res4b10_branch2a', data=res4b9_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2a = mx.symbol.BatchNorm(name='bn4b10_branch2a', data=res4b10_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b10_branch2a = bn4b10_branch2a
        res4b10_branch2a_relu = mx.symbol.Activation(name='res4b10_branch2a_relu', data=scale4b10_branch2a,
                                                     act_type='relu')
        res4b10_branch2b = mx.symbol.Convolution(name='res4b10_branch2b', data=res4b10_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b10_branch2b = mx.symbol.BatchNorm(name='bn4b10_branch2b', data=res4b10_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b10_branch2b = bn4b10_branch2b
        res4b10_branch2b_relu = mx.symbol.Activation(name='res4b10_branch2b_relu', data=scale4b10_branch2b,
                                                     act_type='relu')
        res4b10_branch2c = mx.symbol.Convolution(name='res4b10_branch2c', data=res4b10_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2c = mx.symbol.BatchNorm(name='bn4b10_branch2c', data=res4b10_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b10_branch2c = bn4b10_branch2c
        res4b10 = mx.symbol.broadcast_add(name='res4b10', *[res4b9_relu, scale4b10_branch2c])
        res4b10_relu = mx.symbol.Activation(name='res4b10_relu', data=res4b10, act_type='relu')
        res4b11_branch2a = mx.symbol.Convolution(name='res4b11_branch2a', data=res4b10_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2a = mx.symbol.BatchNorm(name='bn4b11_branch2a', data=res4b11_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b11_branch2a = bn4b11_branch2a
        res4b11_branch2a_relu = mx.symbol.Activation(name='res4b11_branch2a_relu', data=scale4b11_branch2a,
                                                     act_type='relu')
        res4b11_branch2b = mx.symbol.Convolution(name='res4b11_branch2b', data=res4b11_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b11_branch2b = mx.symbol.BatchNorm(name='bn4b11_branch2b', data=res4b11_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b11_branch2b = bn4b11_branch2b
        res4b11_branch2b_relu = mx.symbol.Activation(name='res4b11_branch2b_relu', data=scale4b11_branch2b,
                                                     act_type='relu')
        res4b11_branch2c = mx.symbol.Convolution(name='res4b11_branch2c', data=res4b11_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2c = mx.symbol.BatchNorm(name='bn4b11_branch2c', data=res4b11_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b11_branch2c = bn4b11_branch2c
        res4b11 = mx.symbol.broadcast_add(name='res4b11', *[res4b10_relu, scale4b11_branch2c])
        res4b11_relu = mx.symbol.Activation(name='res4b11_relu', data=res4b11, act_type='relu')
        res4b12_branch2a = mx.symbol.Convolution(name='res4b12_branch2a', data=res4b11_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2a = mx.symbol.BatchNorm(name='bn4b12_branch2a', data=res4b12_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b12_branch2a = bn4b12_branch2a
        res4b12_branch2a_relu = mx.symbol.Activation(name='res4b12_branch2a_relu', data=scale4b12_branch2a,
                                                     act_type='relu')
        res4b12_branch2b = mx.symbol.Convolution(name='res4b12_branch2b', data=res4b12_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b12_branch2b = mx.symbol.BatchNorm(name='bn4b12_branch2b', data=res4b12_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b12_branch2b = bn4b12_branch2b
        res4b12_branch2b_relu = mx.symbol.Activation(name='res4b12_branch2b_relu', data=scale4b12_branch2b,
                                                     act_type='relu')
        res4b12_branch2c = mx.symbol.Convolution(name='res4b12_branch2c', data=res4b12_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2c = mx.symbol.BatchNorm(name='bn4b12_branch2c', data=res4b12_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b12_branch2c = bn4b12_branch2c
        res4b12 = mx.symbol.broadcast_add(name='res4b12', *[res4b11_relu, scale4b12_branch2c])
        res4b12_relu = mx.symbol.Activation(name='res4b12_relu', data=res4b12, act_type='relu')
        res4b13_branch2a = mx.symbol.Convolution(name='res4b13_branch2a', data=res4b12_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2a = mx.symbol.BatchNorm(name='bn4b13_branch2a', data=res4b13_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b13_branch2a = bn4b13_branch2a
        res4b13_branch2a_relu = mx.symbol.Activation(name='res4b13_branch2a_relu', data=scale4b13_branch2a,
                                                     act_type='relu')
        res4b13_branch2b = mx.symbol.Convolution(name='res4b13_branch2b', data=res4b13_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b13_branch2b = mx.symbol.BatchNorm(name='bn4b13_branch2b', data=res4b13_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b13_branch2b = bn4b13_branch2b
        res4b13_branch2b_relu = mx.symbol.Activation(name='res4b13_branch2b_relu', data=scale4b13_branch2b,
                                                     act_type='relu')
        res4b13_branch2c = mx.symbol.Convolution(name='res4b13_branch2c', data=res4b13_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2c = mx.symbol.BatchNorm(name='bn4b13_branch2c', data=res4b13_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b13_branch2c = bn4b13_branch2c
        res4b13 = mx.symbol.broadcast_add(name='res4b13', *[res4b12_relu, scale4b13_branch2c])
        res4b13_relu = mx.symbol.Activation(name='res4b13_relu', data=res4b13, act_type='relu')
        res4b14_branch2a = mx.symbol.Convolution(name='res4b14_branch2a', data=res4b13_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2a = mx.symbol.BatchNorm(name='bn4b14_branch2a', data=res4b14_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b14_branch2a = bn4b14_branch2a
        res4b14_branch2a_relu = mx.symbol.Activation(name='res4b14_branch2a_relu', data=scale4b14_branch2a,
                                                     act_type='relu')
        res4b14_branch2b = mx.symbol.Convolution(name='res4b14_branch2b', data=res4b14_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b14_branch2b = mx.symbol.BatchNorm(name='bn4b14_branch2b', data=res4b14_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b14_branch2b = bn4b14_branch2b
        res4b14_branch2b_relu = mx.symbol.Activation(name='res4b14_branch2b_relu', data=scale4b14_branch2b,
                                                     act_type='relu')
        res4b14_branch2c = mx.symbol.Convolution(name='res4b14_branch2c', data=res4b14_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2c = mx.symbol.BatchNorm(name='bn4b14_branch2c', data=res4b14_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b14_branch2c = bn4b14_branch2c
        res4b14 = mx.symbol.broadcast_add(name='res4b14', *[res4b13_relu, scale4b14_branch2c])
        res4b14_relu = mx.symbol.Activation(name='res4b14_relu', data=res4b14, act_type='relu')
        res4b15_branch2a = mx.symbol.Convolution(name='res4b15_branch2a', data=res4b14_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2a = mx.symbol.BatchNorm(name='bn4b15_branch2a', data=res4b15_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b15_branch2a = bn4b15_branch2a
        res4b15_branch2a_relu = mx.symbol.Activation(name='res4b15_branch2a_relu', data=scale4b15_branch2a,
                                                     act_type='relu')
        res4b15_branch2b = mx.symbol.Convolution(name='res4b15_branch2b', data=res4b15_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b15_branch2b = mx.symbol.BatchNorm(name='bn4b15_branch2b', data=res4b15_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b15_branch2b = bn4b15_branch2b
        res4b15_branch2b_relu = mx.symbol.Activation(name='res4b15_branch2b_relu', data=scale4b15_branch2b,
                                                     act_type='relu')
        res4b15_branch2c = mx.symbol.Convolution(name='res4b15_branch2c', data=res4b15_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2c = mx.symbol.BatchNorm(name='bn4b15_branch2c', data=res4b15_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b15_branch2c = bn4b15_branch2c
        res4b15 = mx.symbol.broadcast_add(name='res4b15', *[res4b14_relu, scale4b15_branch2c])
        res4b15_relu = mx.symbol.Activation(name='res4b15_relu', data=res4b15, act_type='relu')
        res4b16_branch2a = mx.symbol.Convolution(name='res4b16_branch2a', data=res4b15_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2a = mx.symbol.BatchNorm(name='bn4b16_branch2a', data=res4b16_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b16_branch2a = bn4b16_branch2a
        res4b16_branch2a_relu = mx.symbol.Activation(name='res4b16_branch2a_relu', data=scale4b16_branch2a,
                                                     act_type='relu')
        res4b16_branch2b = mx.symbol.Convolution(name='res4b16_branch2b', data=res4b16_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b16_branch2b = mx.symbol.BatchNorm(name='bn4b16_branch2b', data=res4b16_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b16_branch2b = bn4b16_branch2b
        res4b16_branch2b_relu = mx.symbol.Activation(name='res4b16_branch2b_relu', data=scale4b16_branch2b,
                                                     act_type='relu')
        res4b16_branch2c = mx.symbol.Convolution(name='res4b16_branch2c', data=res4b16_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2c = mx.symbol.BatchNorm(name='bn4b16_branch2c', data=res4b16_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b16_branch2c = bn4b16_branch2c
        res4b16 = mx.symbol.broadcast_add(name='res4b16', *[res4b15_relu, scale4b16_branch2c])
        res4b16_relu = mx.symbol.Activation(name='res4b16_relu', data=res4b16, act_type='relu')
        res4b17_branch2a = mx.symbol.Convolution(name='res4b17_branch2a', data=res4b16_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2a = mx.symbol.BatchNorm(name='bn4b17_branch2a', data=res4b17_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b17_branch2a = bn4b17_branch2a
        res4b17_branch2a_relu = mx.symbol.Activation(name='res4b17_branch2a_relu', data=scale4b17_branch2a,
                                                     act_type='relu')
        res4b17_branch2b = mx.symbol.Convolution(name='res4b17_branch2b', data=res4b17_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b17_branch2b = mx.symbol.BatchNorm(name='bn4b17_branch2b', data=res4b17_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b17_branch2b = bn4b17_branch2b
        res4b17_branch2b_relu = mx.symbol.Activation(name='res4b17_branch2b_relu', data=scale4b17_branch2b,
                                                     act_type='relu')
        res4b17_branch2c = mx.symbol.Convolution(name='res4b17_branch2c', data=res4b17_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2c = mx.symbol.BatchNorm(name='bn4b17_branch2c', data=res4b17_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b17_branch2c = bn4b17_branch2c
        res4b17 = mx.symbol.broadcast_add(name='res4b17', *[res4b16_relu, scale4b17_branch2c])
        res4b17_relu = mx.symbol.Activation(name='res4b17_relu', data=res4b17, act_type='relu')
        res4b18_branch2a = mx.symbol.Convolution(name='res4b18_branch2a', data=res4b17_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2a = mx.symbol.BatchNorm(name='bn4b18_branch2a', data=res4b18_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b18_branch2a = bn4b18_branch2a
        res4b18_branch2a_relu = mx.symbol.Activation(name='res4b18_branch2a_relu', data=scale4b18_branch2a,
                                                     act_type='relu')
        res4b18_branch2b = mx.symbol.Convolution(name='res4b18_branch2b', data=res4b18_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b18_branch2b = mx.symbol.BatchNorm(name='bn4b18_branch2b', data=res4b18_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b18_branch2b = bn4b18_branch2b
        res4b18_branch2b_relu = mx.symbol.Activation(name='res4b18_branch2b_relu', data=scale4b18_branch2b,
                                                     act_type='relu')
        res4b18_branch2c = mx.symbol.Convolution(name='res4b18_branch2c', data=res4b18_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2c = mx.symbol.BatchNorm(name='bn4b18_branch2c', data=res4b18_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b18_branch2c = bn4b18_branch2c
        res4b18 = mx.symbol.broadcast_add(name='res4b18', *[res4b17_relu, scale4b18_branch2c])
        res4b18_relu = mx.symbol.Activation(name='res4b18_relu', data=res4b18, act_type='relu')
        res4b19_branch2a = mx.symbol.Convolution(name='res4b19_branch2a', data=res4b18_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2a = mx.symbol.BatchNorm(name='bn4b19_branch2a', data=res4b19_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b19_branch2a = bn4b19_branch2a
        res4b19_branch2a_relu = mx.symbol.Activation(name='res4b19_branch2a_relu', data=scale4b19_branch2a,
                                                     act_type='relu')
        res4b19_branch2b = mx.symbol.Convolution(name='res4b19_branch2b', data=res4b19_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b19_branch2b = mx.symbol.BatchNorm(name='bn4b19_branch2b', data=res4b19_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b19_branch2b = bn4b19_branch2b
        res4b19_branch2b_relu = mx.symbol.Activation(name='res4b19_branch2b_relu', data=scale4b19_branch2b,
                                                     act_type='relu')
        res4b19_branch2c = mx.symbol.Convolution(name='res4b19_branch2c', data=res4b19_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2c = mx.symbol.BatchNorm(name='bn4b19_branch2c', data=res4b19_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b19_branch2c = bn4b19_branch2c
        res4b19 = mx.symbol.broadcast_add(name='res4b19', *[res4b18_relu, scale4b19_branch2c])
        res4b19_relu = mx.symbol.Activation(name='res4b19_relu', data=res4b19, act_type='relu')
        res4b20_branch2a = mx.symbol.Convolution(name='res4b20_branch2a', data=res4b19_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2a = mx.symbol.BatchNorm(name='bn4b20_branch2a', data=res4b20_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b20_branch2a = bn4b20_branch2a
        res4b20_branch2a_relu = mx.symbol.Activation(name='res4b20_branch2a_relu', data=scale4b20_branch2a,
                                                     act_type='relu')
        res4b20_branch2b = mx.symbol.Convolution(name='res4b20_branch2b', data=res4b20_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b20_branch2b = mx.symbol.BatchNorm(name='bn4b20_branch2b', data=res4b20_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b20_branch2b = bn4b20_branch2b
        res4b20_branch2b_relu = mx.symbol.Activation(name='res4b20_branch2b_relu', data=scale4b20_branch2b,
                                                     act_type='relu')
        res4b20_branch2c = mx.symbol.Convolution(name='res4b20_branch2c', data=res4b20_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2c = mx.symbol.BatchNorm(name='bn4b20_branch2c', data=res4b20_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b20_branch2c = bn4b20_branch2c
        res4b20 = mx.symbol.broadcast_add(name='res4b20', *[res4b19_relu, scale4b20_branch2c])
        res4b20_relu = mx.symbol.Activation(name='res4b20_relu', data=res4b20, act_type='relu')
        res4b21_branch2a = mx.symbol.Convolution(name='res4b21_branch2a', data=res4b20_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2a = mx.symbol.BatchNorm(name='bn4b21_branch2a', data=res4b21_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b21_branch2a = bn4b21_branch2a
        res4b21_branch2a_relu = mx.symbol.Activation(name='res4b21_branch2a_relu', data=scale4b21_branch2a,
                                                     act_type='relu')
        res4b21_branch2b = mx.symbol.Convolution(name='res4b21_branch2b', data=res4b21_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b21_branch2b = mx.symbol.BatchNorm(name='bn4b21_branch2b', data=res4b21_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b21_branch2b = bn4b21_branch2b
        res4b21_branch2b_relu = mx.symbol.Activation(name='res4b21_branch2b_relu', data=scale4b21_branch2b,
                                                     act_type='relu')
        res4b21_branch2c = mx.symbol.Convolution(name='res4b21_branch2c', data=res4b21_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2c = mx.symbol.BatchNorm(name='bn4b21_branch2c', data=res4b21_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b21_branch2c = bn4b21_branch2c
        res4b21 = mx.symbol.broadcast_add(name='res4b21', *[res4b20_relu, scale4b21_branch2c])
        res4b21_relu = mx.symbol.Activation(name='res4b21_relu', data=res4b21, act_type='relu')
        res4b22_branch2a = mx.symbol.Convolution(name='res4b22_branch2a', data=res4b21_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2a = mx.symbol.BatchNorm(name='bn4b22_branch2a', data=res4b22_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b22_branch2a = bn4b22_branch2a
        res4b22_branch2a_relu = mx.symbol.Activation(name='res4b22_branch2a_relu', data=scale4b22_branch2a,
                                                     act_type='relu')
        res4b22_branch2b = mx.symbol.Convolution(name='res4b22_branch2b', data=res4b22_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b22_branch2b = mx.symbol.BatchNorm(name='bn4b22_branch2b', data=res4b22_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b22_branch2b = bn4b22_branch2b
        res4b22_branch2b_relu = mx.symbol.Activation(name='res4b22_branch2b_relu', data=scale4b22_branch2b,
                                                     act_type='relu')
        res4b22_branch2c = mx.symbol.Convolution(name='res4b22_branch2c', data=res4b22_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2c = mx.symbol.BatchNorm(name='bn4b22_branch2c', data=res4b22_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps = self.eps)
        scale4b22_branch2c = bn4b22_branch2c
        res4b22 = mx.symbol.broadcast_add(name='res4b22', *[res4b21_relu, scale4b22_branch2c])
        res4b22_relu = mx.symbol.Activation(name='res4b22_relu', data=res4b22, act_type='relu')

        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=res4b22_relu, num_filter=2048, pad=(0, 0),
                                              kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale5a_branch1 = bn5a_branch1
        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=res4b22_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2a = bn5a_branch2a
        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')
        res5a_branch2b_offset_weight = mx.symbol.Variable('res5a_branch2b_offset_weight', lr_mult=1.0)
        res5a_branch2b_offset_bias = mx.symbol.Variable('res5a_branch2b_offset_bias', lr_mult=2.0)
        res5a_branch2b_offset = mx.symbol.Convolution(name='res5a_branch2b_offset', data = res5a_branch2a_relu,
                                                      num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                                      weight=res5a_branch2b_offset_weight, bias=res5a_branch2b_offset_bias)
        res5a_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5a_branch2b', data=res5a_branch2a_relu, offset=res5a_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=1,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1, scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a, act_type='relu')
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')
        res5b_branch2b_offset_weight = mx.symbol.Variable('res5b_branch2b_offset_weight', lr_mult=1.0)
        res5b_branch2b_offset_bias = mx.symbol.Variable('res5b_branch2b_offset_bias', lr_mult=2.0)
        res5b_branch2b_offset = mx.symbol.Convolution(name='res5b_branch2b_offset', data = res5b_branch2a_relu,
                                                      num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                                      weight=res5b_branch2b_offset_weight, bias=res5b_branch2b_offset_bias)
        res5b_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5b_branch2b', data=res5b_branch2a_relu, offset=res5b_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=1,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu, scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b, act_type='relu')
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')
        res5c_branch2b_offset_weight = mx.symbol.Variable('res5c_branch2b_offset_weight', lr_mult=1.0)
        res5c_branch2b_offset_bias = mx.symbol.Variable('res5c_branch2b_offset_bias', lr_mult=2.0)
        res5c_branch2b_offset = mx.symbol.Convolution(name='res5c_branch2b_offset', data = res5c_branch2a_relu,
                                                      num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                                      weight=res5c_branch2b_offset_weight, bias=res5c_branch2b_offset_bias)
        res5c_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5c_branch2b', data=res5c_branch2a_relu, offset=res5c_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=1,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu, scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c, act_type='relu')

        return res5c_relu

    def get_train_symbol(self, num_classes):
        """
        get symbol for training
        :param num_classes: num of classes
        :return: the symbol for training
        """
        data = mx.symbol.Variable(name="data")
        seg_cls_gt = mx.symbol.Variable(name='label')

        # shared convolutional layers
        conv_feat = self.get_resnet_conv(data)

        # subsequent fc layers by haozhi
        fc6_bias = mx.symbol.Variable('fc6_bias', lr_mult=2.0)
        fc6_weight = mx.symbol.Variable('fc6_weight', lr_mult=1.0)

        fc6 = mx.symbol.Convolution(data=conv_feat, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="fc6",
                                    bias=fc6_bias, weight=fc6_weight, workspace=self.workspace)
        relu_fc6 = mx.sym.Activation(data=fc6, act_type='relu', name='relu_fc6')

        score_bias = mx.symbol.Variable('score_bias', lr_mult=2.0)
        score_weight = mx.symbol.Variable('score_weight', lr_mult=1.0)

        score = mx.symbol.Convolution(data=relu_fc6, kernel=(1, 1), pad=(0, 0), num_filter=num_classes, name="score",
                                      bias=score_bias, weight=score_weight, workspace=self.workspace)

        upsampling = mx.symbol.Deconvolution(data=score, num_filter=num_classes, kernel=(32, 32), stride=(16, 16),
                                             num_group=num_classes, no_bias=True, name='upsampling',
                                             attr={'lr_mult': '0.0'}, workspace=self.workspace)

        croped_score = mx.symbol.Crop(*[upsampling, data], offset=(8, 8), name='croped_score')
        softmax = mx.symbol.SoftmaxOutput(data=croped_score, label=seg_cls_gt, normalization='valid', multi_output=True,
                                          use_ignore=True, ignore_label=255, name="softmax")

        return softmax

    def get_test_symbol(self, num_classes):
        """
        get symbol for testing
        :param num_classes: num of classes
        :return: the symbol for testing
        """
        data = mx.symbol.Variable(name="data")

        # shared convolutional layers
        conv_feat = self.get_resnet_conv(data)

        fc6_bias = mx.symbol.Variable('fc6_bias', lr_mult=2.0)
        fc6_weight = mx.symbol.Variable('fc6_weight', lr_mult=1.0)

        fc6 = mx.symbol.Convolution(
            data=conv_feat, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="fc6", bias=fc6_bias, weight=fc6_weight,
            workspace=self.workspace)
        relu_fc6 = mx.sym.Activation(data=fc6, act_type='relu', name='relu_fc6')

        score_bias = mx.symbol.Variable('score_bias', lr_mult=2.0)
        score_weight = mx.symbol.Variable('score_weight', lr_mult=1.0)

        score = mx.symbol.Convolution(
            data=relu_fc6, kernel=(1, 1), pad=(0, 0), num_filter=num_classes, name="score", bias=score_bias,
            weight=score_weight, workspace=self.workspace)

        upsampling = mx.symbol.Deconvolution(
            data=score, num_filter=num_classes, kernel=(32, 32), stride=(16, 16), num_group=num_classes, no_bias=True,
            name='upsampling', attr={'lr_mult': '0.0'}, workspace=self.workspace)

        croped_score = mx.symbol.Crop(*[upsampling, data], offset=(8, 8), name='croped_score')

        softmax = mx.symbol.SoftmaxOutput(data=croped_score, normalization='valid', multi_output=True, use_ignore=True,
                                          ignore_label=255, name="softmax")

        return softmax

    def get_symbol(self, cfg, is_train=True):
        """
        return a generated symbol, it also need to be assigned to self.sym
        """

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES

        if is_train:
            self.sym = self.get_train_symbol(num_classes=num_classes)
        else:
            self.sym = self.get_test_symbol(num_classes=num_classes)

        return self.sym

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


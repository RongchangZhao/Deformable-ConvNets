"""
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py 
Original author Wei Wu
Referenced https://github.com/bamos/densenet.pytorch/blob/master/densenet.py
Original author bamos
Referenced https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
Original author andreasveit
Referenced https://github.com/Nicatio/Densenet/blob/master/mxnet/symbol_densenet.py
Original author Nicatio
Implemented the following paper:     DenseNet-BC
Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten. "Densely Connected Convolutional Networks"
Coded by Lin Xiong Mar-1, 2017
"""
import mxnet as mx
import math

def BasicBlock(data, growth_rate, stride, name, bottle_neck=True, drop_out=0.0, bn_mom=0.9, workspace=512):
    """Return BaiscBlock Unit symbol for building DenseBlock
    Parameters
    ----------
    data : str
        Input data
    growth_rate : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    drop_out : float
        Probability of an element to be zeroed. Default = 0.2
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    # import pdb
    # pdb.set_trace()

    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1   = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1  = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(growth_rate*4), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        if drop_out > 0:
            conv1 = mx.symbol.Dropout(data=conv1, p=drop_out, name=name + '_dp1')
        bn2   = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2  = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(growth_rate), kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if drop_out > 0:
            conv2 = mx.symbol.Dropout(data=conv2, p=drop_out, name=name + '_dp2')
        #return mx.symbol.Concat(data, conv2, name=name + '_concat0')
        return conv2
    else:
        bn1   = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1  = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(growth_rate), kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        if drop_out > 0:
            conv1 = mx.symbol.Dropout(data=conv1, p=drop_out, name=name + '_dp1')
        #return mx.symbol.Concat(data, conv1, name=name + '_concat0')
        return conv1

    
def DenseBlock(units_num, data, growth_rate, name, bottle_neck=True, drop_out=0.0, bn_mom=0.9, workspace=512):
    """Return DenseBlock Unit symbol for building DenseNet
    Parameters
    ----------
    units_num : int
        the number of BasicBlock in each DenseBlock
    data : str	
        Input data
    growth_rate : int
        Number of output channels
    drop_out : float
        Probability of an element to be zeroed. Default = 0.2
    workspace : int
        Workspace used in convolution operator
    """
    # import pdb
    # pdb.set_trace()

    for i in range(units_num):
        Block = BasicBlock(data, growth_rate=growth_rate, stride=(1,1), name=name + '_unit%d' % (i+1), 
                            bottle_neck=bottle_neck, drop_out=drop_out, 
                            bn_mom=bn_mom, workspace=workspace)
        data = mx.symbol.Concat(data, Block, name=name + '_concat%d' % (i+1))
    return data


def TransitionBlock(num_stage, data, num_filter, stride, name, nopool=False, drop_out=0.0, bn_mom=0.9, workspace=512):
    """Return TransitionBlock Unit symbol for building DenseNet
    Parameters
    ----------
    num_stage : int
        Number of stage
    data : str
        Input data
    num : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    name : str
        Base name of the operators
    drop_out : float
        Probability of an element to be zeroed. Default = 0.2
    workspace : int
        Workspace used in convolution operator
    """
    bn1   = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1  = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter,
                                kernel=(1,1), stride=stride, pad=(0,0), no_bias=True, 
                                workspace=workspace, name=name + '_conv1')
    if drop_out > 0:
        conv1 = mx.symbol.Dropout(data=conv1, p=drop_out, name=name + '_dp1')
    
    if nopool:
        return conv1
    else:
        return mx.symbol.Pooling(conv1, global_pool=False, kernel=(2,2), stride=(2,2), pool_type='avg', name=name + '_pool%d' % (num_stage+1))


def DenseNet(data, units, num_stage, growth_rate, num_class, data_type, decoder=False, \
             reduction=1.0, drop_out=0., bottle_neck=True, bn_mom=0.9, taskmode='CLS', workspace=512):
    
    """Return DenseNet symbol of imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    growth_rate : int
        Number of output channels
    num_class : int
        Ouput size of symbol
    data_type : str
        the type of dataset
    reduction : float
        Compression ratio. Default = 0.5
    drop_out : float
        Probability of an element to be zeroed. Default = 0.2
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stage)
    init_channels = 2 * growth_rate
    n_channels = init_channels
    #data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'imagenet':
        
        if taskmode == 'CLS':
            body = mx.sym.Convolution(data=data, num_filter=growth_rate*2, kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        elif taskmode == 'SEG':
            n_channels = 64 # DSOD Stem Block
            body = mx.sym.Convolution(data=data, num_filter=n_channels, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0stem0", workspace=workspace)
            body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0stem0')
            body = mx.sym.Activation(data=body, act_type='relu', name='relu0stem0')
            body = mx.sym.Convolution(data=data, num_filter=n_channels, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0stem1", workspace=workspace)
            body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0stem1')
            body = mx.sym.Activation(data=body, act_type='relu', name='relu0stem1')
            n_channels = n_channels * 2 
            body = mx.sym.Convolution(data=data, num_filter=n_channels, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0stem0", workspace=workspace)
            
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    elif data_type == 'vggface':
        body = mx.sym.Convolution(data=data, num_filter=growth_rate*2, kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    elif data_type == 'msface':
        body = mx.sym.Convolution(data=data, num_filter=growth_rate*2, kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    else:
        raise ValueError("do not support {} yet".format(data_type))
        
        
    if taskmode == 'CLS':
        
        nopoolplan = [False, False , False]
        
        for i in range(num_stage-1):
            body = DenseBlock(units[i], body, growth_rate=growth_rate, name='DBstage%d' % (i + 1), bottle_neck=bottle_neck, drop_out=drop_out, bn_mom=bn_mom, workspace=workspace)
            n_channels += units[i]*growth_rate
            n_channels = int(math.floor(n_channels*reduction))
            body = TransitionBlock(i, body, n_channels, stride=(1,1), nopool= nopoolplan[i], name='TBstage%d' % (i + 1), drop_out=drop_out, bn_mom=bn_mom, workspace=workspace)
        body = DenseBlock(units[num_stage-1], body, growth_rate=growth_rate, name='DBstage%d' % (num_stage), bottle_neck=bottle_neck, drop_out=drop_out, bn_mom=bn_mom, workspace=workspace)
        
        
        bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
        flat = mx.symbol.Flatten(data=pool1)
        fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_class, name='fc1')
        return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    
    
    elif taskmode == 'SEG':
        
        nopoolplan = [False, True , True]
        
        if decoder:
            decoding_list = []
        
        for i in range(num_stage-1):
            body = DenseBlock(units[i], body, growth_rate=growth_rate, name='DBstage%d' % (i + 1), bottle_neck=bottle_neck, drop_out=drop_out, bn_mom=bn_mom, workspace=workspace)
            n_channels += units[i]*growth_rate
            n_channels = int(math.floor(n_channels*reduction))
            body = TransitionBlock(i, body, n_channels, stride=(1,1), nopool= nopoolplan[i], name='TBstage%d' % (i + 1), drop_out=drop_out, bn_mom=bn_mom, workspace=workspace)
            
            if decoder:
                exec('body_{0} = body'.format(i))
                exec('decoding_list.append(body_{0})'.format(i))
                
            
        body = DenseBlock(units[num_stage-1], body, growth_rate=growth_rate, name='DBstage%d' % (num_stage), bottle_neck=bottle_neck, drop_out=drop_out, bn_mom=bn_mom, workspace=workspace)
        
        bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        
        if decoder:
            exec('decoding_list.append(relu1)'.format(i))
        
        if decoder:
            return decoding_list
        else:
            return relu1
        


class FC_Dense():
    
    def __init__(self, num_classes,
                 units, num_stage, growth_rate, data_type='imagenet', reduction=1.0, drop_out=0., bottle_neck=True,
                 conv_workspace=512, usemax = False,
                taskmode='CLS', dtype='float32', **kwargs):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.num_classes = num_classes
        self.units = units
        self.num_stage = num_stage
        self.growth_rate = growth_rate
        self.data_type = data_type
        self.reduction = reduction
        self.drop_out = drop_out
        self.bottle_neck = bottle_neck
        self.conv_workspace = conv_workspace
        self.taskmode = taskmode
        self.usemax = usemax
        

    def get_cls_symbol(self, **kwargs):
        
        data = mx.symbol.Variable(name="data")
        # units, num_stage, growth_rate, num_class, data_type, reduction=0.5, drop_out=0., bottle_neck=True, bn_mom=0.9, workspace=512
        return DenseNet(  data,
		          self.units,
                          self.num_stage,
                          self.growth_rate,
                          self.num_classes,
                          self.data_type,
                          reduction = self.reduction,
                          drop_out = self.drop_out,
                          bottle_neck = self.bottle_neck,
                          workspace=self.conv_workspace,
                          taskmode='CLS',
                          **kwargs)
        
    def get_seg_symbol(self, **kwargs):
        
        data = mx.symbol.Variable(name="data")
        seg_cls_gt = mx.symbol.Variable(name="softmax_label")
        
        
        conv_feat = DenseNet( data,  
 	     	          self.units,
                          self.num_stage,
                          self.growth_rate,
                          self.num_classes,
                          self.data_type,
                          reduction = self.reduction,
                          drop_out = self.drop_out,
                          bottle_neck = self.bottle_neck,
                          workspace=self.conv_workspace,
                          taskmode='SEG', 
                          **kwargs)
        
        fc6_bias = mx.symbol.Variable('fc6_bias', lr_mult=2.0)
        fc6_weight = mx.symbol.Variable('fc6_weight', lr_mult=1.0)
        fc6 = mx.symbol.Convolution(data=conv_feat, kernel=(1, 1), pad=(0, 0), num_filter=512, name="fc6",
                                    bias=fc6_bias, weight=fc6_weight, workspace=self.conv_workspace)
        relu_fc6 = mx.sym.Activation(data=fc6, act_type='relu', name='relu_fc6')
        
        # Fix
        upstride = 4
        # Fix
        atrouslist =  [2,3,6,12,18,24]
        atrouslistlen = len(atrouslist)
        
        # V3
        score_basic_bias = mx.symbol.Variable('score_basic_bias', lr_mult=2.0)
        score_basic_weight = mx.symbol.Variable('score_basic_weight',lr_mult=1.0)
        score_basic = mx.symbol.Convolution(data=relu_fc6, kernel=(1, 1),\
                    num_filter=self.num_classes, \
                    name="score_basic",bias=score_basic_bias, weight=score_basic_weight, \
                    workspace=self.conv_workspace)
        
        atrouslistsymbol = [score_basic]
        
        for i in range(atrouslistlen):
            thisatrous = atrouslist[i]
            exec('score_{ind}_bias = mx.symbol.Variable(\'score_{ind}_bias\', lr_mult=2.0)'.format(ind=i))
            exec('score_{ind}_weight = mx.symbol.Variable(\'score_{ind}_weight\', lr_mult=1.0)'.format(ind=i))
            exec('score_{ind} = mx.symbol.Convolution(data=relu_fc6, kernel=(3, 3), pad=(thisatrous, thisatrous),\
                    dilate=(thisatrous, thisatrous) ,num_filter=self.num_classes, \
                    name="score_{ind}",bias=score_{ind}_bias, weight=score_{ind}_weight, \
                    workspace=self.conv_workspace)'.format(ind=i))
            
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
            score = mx.sym.BatchNorm(data=score, fix_gamma=False, momentum=0.9, eps=2e-5, name='maxbn')
            score = mx.sym.Activation(data=score, act_type='relu')
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
                                         attr={'lr_mult': '0.1'}, workspace=self.conv_workspace)
        
        croped_score = mx.symbol.Crop(*[upsampling, data], offset=(upstride/2, upstride/2), name='croped_score')
        softmax = mx.symbol.SoftmaxOutput(data=croped_score, label=seg_cls_gt, normalization='valid', multi_output=True,
                                      attr={'lr_mult': '1.0'},use_ignore=True, ignore_label=255, name="softmax")

        return softmax

import cPickle
import mxnet as mx
# from utils.symbol import Symbol


###### UNIT LIST #######

def encoder_unit(data, num_filter, stride, dim_match, name, bottle_neck=0, \
                 dilation=1, irv2 = False, deform=0, sqex=1, bn_mom=0.9, unitbatchnorm=False, workspace=256, memonger=False):
    
    """
    Return UNet Encoder Unit Symbol
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
        
        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), 
                                   pad=(dilation,dilation), dilate=(dilation,dilation), 
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        
        
        
        if unitbatchnorm:
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        else:
            act1 = mx.sym.Activation(data=conv1, act_type='relu', name=name + '_relu1')
            
        
        
        if deform == 0:
            conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1),
                                   pad=(dilation,dilation), dilate=(dilation,dilation),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        else:
            
            offsetweight = mx.symbol.Variable(name+'_offset_weight', lr_mult=1.0)
            offsetbias = mx.symbol.Variable(name+'_offset_bias', lr_mult=2.0)
            offset = mx.symbol.Convolution(name=name+'_offset', data = act1,
                                                      num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                                      weight=offsetweight, bias=offsetbias)
            
            
            conv2 = mx.contrib.symbol.DeformableConvolution(data=act1, offset=offset,
                     num_filter=num_filter, pad=(dilation,dilation), kernel=(3, 3), num_deformable_group=1,
                     stride=(1, 1), dilate=(dilation,dilation), no_bias=True)
        

        if unitbatchnorm:
            bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        else:
            act2 = mx.sym.Activation(data=conv2, act_type='relu', name=name + '_relu2')
            
        if sqex == 0:
            out = act2
        else:
            pool_se = mx.symbol.Pooling(data=act2, cudnn_off=True, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_se_pool')
            flat = mx.symbol.Flatten(data=pool_se)
            se_fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=int(num_filter/4), name=name + '_se_fc1') #, lr_mult=0.25)
            se_relu = mx.sym.Activation(data=se_fc1, act_type='relu')
            se_fc2 = mx.symbol.FullyConnected(data=se_relu, num_hidden=num_filter, name=name + '_se_fc2') #, lr_mult=0.25)
            se_act = mx.sym.Activation(se_fc2, act_type="sigmoid")
            se_reshape = mx.symbol.Reshape(se_act, shape=(-1, num_filter, 1, 1), name="se_reshape")
            se_scale = mx.sym.broadcast_mul(act2, se_reshape)
            out = se_scale  
        
        return out
    

def decoder_unit(data, num_filter, stride, dim_match, name, bottle_neck=0, kernel=3, dilation=1, pad=1, irv2 = False, sqex=1, bn_mom=0.9, unitbatchnorm=False, workspace=256, memonger=False):
    
    """
    Return UNet Decoder Blocked Symbol
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
        
        conv1 = mx.sym.Deconvolution(data=data, num_filter=num_filter, kernel=(kernel,kernel), stride=(stride,stride), 
                                   pad=(pad,pad),  
                                   no_bias=True, workspace=workspace, name=name + '_deconv1')
        
        
        if unitbatchnorm:
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        else:
            act1 = mx.sym.Activation(data=conv1, act_type='relu', name=name + '_relu1')
            
        
        act2 = act1
            
        if sqex == 0:
            out = act2
        else:
            pool_se = mx.symbol.Pooling(data=act2, cudnn_off=True, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_se_pool')
            flat = mx.symbol.Flatten(data=pool_se)
            se_fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=int(num_filter/4), name=name + '_se_fc1') #, lr_mult=0.25)
            se_relu = mx.sym.Activation(data=se_fc1, act_type='relu')
            se_fc2 = mx.symbol.FullyConnected(data=se_relu, num_hidden=num_filter, name=name + '_se_fc2') #, lr_mult=0.25)
            se_act = mx.sym.Activation(se_fc2, act_type="sigmoid")
            se_reshape = mx.symbol.Reshape(se_act, shape=(-1, num_filter, 1, 1), name="se_reshape")
            se_scale = mx.sym.broadcast_mul(act2, se_reshape)
            out = se_scale  
        
        return out
    
    
def UNet(data, num_filter, bottle_neck=0, \
                 deform=0, sqex=1, bn_mom=0.9, unitbatchnorm=False, expandmode='exp',workspace=256, memonger=False, **kwargs):
    
    en_kwargs = {"deform":deform, "sqex":sqex, "unitbatchnorm":unitbatchnorm}
    de_kwargs = {"sqex":sqex, "unitbatchnorm":unitbatchnorm}
    
    if expandmode=='exp':
        def expander(i):
            return i**2/2
    elif expandmode=='lin':
        def expander(i):
            return i
    
    e0 = encoder_unit(data, num_filter, 1, True, "e0", **en_kwargs)
    syn0 = encoder_unit(e0, num_filter*expander(2), 1, True, "syn0", **en_kwargs) # 1120
    
    e1 = mx.sym.Pooling(data=syn0, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    e2 = encoder_unit(e1, num_filter*expander(2), 1, True, "e2", **en_kwargs)
    syn1 = encoder_unit(e2, num_filter*expander(3), 1, True, "syn1", **en_kwargs) # 560

    e3 = mx.sym.Pooling(data=syn1, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    e4 = encoder_unit(e3, num_filter*expander(3), 1, True, "e4", **en_kwargs)
    syn2 = encoder_unit(e4, num_filter*expander(4), 1, True, "syn2", **en_kwargs) # 280
    
    e5 = mx.sym.Pooling(data=syn2, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    e6 = encoder_unit(e5, num_filter*expander(4), 1, True, "e6" , **en_kwargs)
    #e7 = encoder_unit(e6, num_filter*expander(5), 1, True, "e7" , **en_kwargs) # 140
    
    ### Fix V1 Start From Here, Uncomment Last To Recover
    syn3 = encoder_unit(e6, num_filter*expander(5), 1, True, "syn3" , **en_kwargs) # 140 
    
    e7 = mx.sym.Pooling(data=syn3, kernel=(3,3), stride=(2,2),pad=(1,1),pool_type='max') 
    e8 = encoder_unit(e7, num_filter*expander(5), 1, True, "e8", **en_kwargs )
    #e9 = encoder_unit(e8, num_filter*expander(6), 1, True, "e9", **en_kwargs) # 70
    
    ### Fix V2 Start From Here, Uncomment Last To Recover
    
    syn4 = encoder_unit(e8, num_filter*expander(6), 1, True, "syn4", **en_kwargs) # 70
    
    e9 = mx.sym.Pooling(data=syn4, kernel=(3,3), stride=(2,2),pad=(1,1),pool_type='max') 
    e10 = encoder_unit(e9, num_filter*expander(6), 1, True, "e9", **en_kwargs )
    e11 = encoder_unit(e10, num_filter*expander(7), 1, True, "e10", **en_kwargs) # 35
    
    d15 = mx.symbol.Concat(*[decoder_unit(e11, num_filter*expander(7),2,True,"e11",kernel=2,pad=0,**de_kwargs),syn4])
    d14 = decoder_unit(d15, num_filter * expander(6), 1, True, "d14", **de_kwargs)
    d13 = decoder_unit(d14, num_filter * expander(6), 1, True, "d13", **de_kwargs)
    d12 = mx.symbol.Concat(*[decoder_unit(d13, num_filter*expander(6),2,True,"d12",kernel=2,pad=0,**de_kwargs),syn3])
    
    ### Fix V2 End Till Here, Uncomment Next To Recover
    #d12 = mx.symbol.Concat(*[decoder_unit(e9, num_filter*expander(6),2,True,"d12",kernel=2,pad=0,**de_kwargs),syn3])
    d11 = decoder_unit(d12, num_filter * expander(5), 1, True, "d11", **de_kwargs)
    d10 = decoder_unit(d11, num_filter * expander(5), 1, True, "d10", **de_kwargs)
    
    d9 = mx.symbol.Concat(*[decoder_unit(d10, num_filter * expander(5) , 2 , True, "d9" , kernel=2, pad=0, **de_kwargs ), syn2])
    ### Fix V1 End Till Here, Uncomment Next To Recover
    #d9 = mx.symbol.Concat(*[decoder_unit(e7, num_filter * expander(5) , 2 , True, "d9" , kernel=2, pad=0, **de_kwargs ), syn2])
    
    
    d8 = decoder_unit(d9, num_filter * expander(4), 1, True, "d8", **de_kwargs)
    d7 = decoder_unit(d8, num_filter * expander(4), 1, True, "d7", **de_kwargs)

    
    d6 = mx.symbol.Concat(*[decoder_unit(d7, num_filter*expander(4), 2, True, "d6" , kernel=2, pad=0, **de_kwargs), syn1] )

    
    d5 = decoder_unit(d6, num_filter * expander(3), 1, True, "d5", **de_kwargs)
    d4 = decoder_unit(d5, num_filter * expander(3), 1, True, "d4", **de_kwargs)

    d3 = mx.symbol.Concat(*[decoder_unit(d4, num_filter * expander(3), 2, True, "d3", kernel=2, pad=0, **de_kwargs), syn0])

    d2 = decoder_unit(d3, num_filter * expander(2), 1, True, "d2", **de_kwargs)
    d1 = decoder_unit(d2, num_filter * expander(2), 1, True, "d1", **de_kwargs)

    d0 = decoder_unit(d1, num_filter * expander(2), 1, True, "d0", **de_kwargs)
    
    return d0
    
    
class UNet_dcn():
    
    def __init__(self, num_classes , num_filter=32, bottle_neck=0, \
                 deform=0, sqex = 0, expandmode='exp',conv_workspace=256,
                 dtype='float32', **kwargs):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 4096
        self.num_classes = num_classes
        self.num_filter = num_filter
        self.bottle_neck = bottle_neck
        self.expandmode = expandmode
        self.deform = deform
        self.sqex = sqex
    
    def get_seg_symbol(self, **kwargs):
    
        data = mx.symbol.Variable(name="data")
        seg_cls_gt = mx.symbol.Variable(name="softmax_label")
        
        
        
        conv_feat = UNet(data,
                            self.num_filter,
                            bottle_neck=self.bottle_neck,
                            expandmode=self.expandmode,
                            deform=self.deform,
                            sqex=self.sqex,
                            conv_workspace=256,
                            dtype='float32',
                            **kwargs)
        
        
        fc6_bias = mx.symbol.Variable('fc6_bias', lr_mult=2.0)
        fc6_weight = mx.symbol.Variable('fc6_weight', lr_mult=1.0)
        fc6 = mx.symbol.Convolution(data=conv_feat, kernel=(1, 1), pad=(0, 0), num_filter=self.num_classes, name="fc6",
                                    bias=fc6_bias, weight=fc6_weight, workspace=self.workspace)
        #croped_score = mx.symbol.Crop(*[fc6, data], offset=(3,3),name='croped_score')
        softmax = mx.symbol.SoftmaxOutput(data=fc6, label=seg_cls_gt, normalization='valid', multi_output=True,
                                          use_ignore=True, ignore_label=255, name="softmax")
        
        
        return softmax
        
        
        
    
    
    
    


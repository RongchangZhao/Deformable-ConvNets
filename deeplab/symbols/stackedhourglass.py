import cPickle
import mxnet as mx
# from utils.symbol import Symbol


numofparts = 15
numoflinks = 13

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
    
    elif bottle_neck == 1:
        
        conv1 = mx.sym.Convolution(data=data, num_filter=numfilter, kernel=(1,1), stride=(1,1),
                                  no_bias=True, workspace=workspace, name=name + '_conv1')
        if unitbatchnorm:
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        else:
            act1 = mx.sym.Activation(data=conv1, act_type='relu', name=name + '_relu1')
        
        conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), 
                                   pad=(dilation,dilation), dilate=(dilation,dilation), 
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        
        if unitbatchnorm:
            bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        else:
            act2 = mx.sym.Activation(data=conv2, act_type='relu', name=name + '_relu2')
        
        
        conv3 = mx.sym.Convolution(data=act1, num_filter=num_filter,kernel=(1,1),stride=(1,1),
                                  no_bias=True, workspace=workspace, name=name + '_conv3')
        if unitbatchnorm:
            bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
            act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu1')
        else:
            act3 = mx.sym.Activation(data=conv1, act_type='relu', name=name + '_relu1')
            
        return act3
        
    

def decoder_unit(data, num_filter, stride, dim_match, name, bottle_neck=0, kernel=3, dilation=1, pad=1, useup = False, irv2 = False, sqex=1, bn_mom=0.9, unitbatchnorm=False, workspace=256, memonger=False):
    
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
    
    elif bottle_neck == 1:
        
        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=(1,1),  
                                   no_bias=True, workspace=workspace, name=name + '_deconv1')
        
        
        if unitbatchnorm:
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        else:
            act1 = mx.sym.Activation(data=conv1, act_type='relu', name=name + '_relu1')
            
        
        conv2 = mx.sym.Deconvolution(data=act1, num_filter=num_filter, kernel=(kernel,kernel), stride=(stride,stride), 
                                   pad=(pad,pad),  
                                   no_bias=True, workspace=workspace, name=name + '_deconv2')
        if unitbatchnorm:
            bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        else:
            act2 = mx.sym.Activation(data=conv2, act_type='relu', name=name + '_relu2')
    
    
        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1,1), stride=(1,1), 
                                   no_bias=True, workspace=workspace, name=name + '_deconv3')
        
        if unitbatchnorm:
            bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn3')
            act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        else:
            act3 = mx.sym.Activation(data=conv3, act_type='relu', name=name + '_relu3')
            
        return act3
    

## 1. Hourglass:
# inp, n, f, hg_id


def hourglass(data, num_filter, name, codec_structure=4, \
              bn_mom = 0.9, unitbatchnorm=False, expandmode='lin', workspace=256, memonger=False):
    
    out = None
    if expandmode=='exp':
        def expander(i):
            return i**2
    elif expandmode=='lin':
        def expander(i):
            return i+1
    
    encodedown_0 = data
    
    for idx in range(codec_structure):
        s = 'encodeup_{i} = encoder_unit(encodedown_{idx}, {num_filter},1, True, "{name}"+"_encodeup_"+"{i}",bottle_neck=0)'.format(idx = idx, i = idx+1, num_filter = expander(idx)*num_filter, name=name)
        print s
        exec(s)
        
        s = 'pool_{i} = mx.sym.Pooling(data=encodeup_{i}, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type="max")'.format(i = idx+1)
        print s
        exec(s)
        
        s = 'encodedown_{i} = encoder_unit(pool_{i}, {num_filter},1, True, "{name}"+"_encodedown_"+"{i}",bottle_neck=0)'.format(idx = idx, i = idx+1, num_filter = expander(idx+1)*num_filter, name=name)
        print s
        exec(s)
        
    s = 'decodeup_{cst} = encodedown_{cst}'.format(cst = codec_structure)
    print s
    exec(s)
    
    
    for idx in range(codec_structure)[::-1] :
        
        s = 'decodedown_{ix} = decoder_unit(decodeup_{i}, {nf}, 1, True, "{nm}"+"_decodedown_"+"{ix}", bottle_neck=0, kernel=3, pad=1) + pool_{i} '.format(ix = idx, i = idx+1, nf = num_filter*expander(idx), nm = name) # 
        print s
        exec(s)
        s = 'unpool_{ix} = decoder_unit(decodedown_{ix}, {nf}, 2, True, "{nm}"+"_deconvcode_"+"{ix}", bottle_neck=0, kernel=2, pad=0) + encodeup_{i}  '.format(ix=idx, i = idx+1, nf=num_filter*expander(idx), nm = name) # 
        print s
        exec(s) 
        
        
        if idx == 0:
            s = 'decodeup_{ix} = decoder_unit(unpool_{ix}, {nf}, 1, True, "{nm}"+"_decodeup_"+"{ix}", bottle_neck=0, kernel=3, pad = 1) + encodedown_{ix}'.format(ix = idx, i = idx+1, nf = num_filter*expander(idx), nm = name)
        else:
            s = 'decodeup_{ix} = decoder_unit(unpool_{ix}, {nf}, 1, True, "{nm}"+"_decodeup_"+"{ix}", bottle_neck=0, kernel=3, pad = 1) + encodedown_{ix}'.format(ix = idx, i = idx+1, nf = num_filter*expander(idx), nm = name)
        print s
        exec(s) # 
    
    
    exec('out = decodeup_0')
    return out



def stackedhourglass(repetition, num_filter, name,  numofparts, numoflinks, codec_structure=3, layout_layer=2,\
              bn_mom = 0.9, unitbatchnorm=False, expandmode='lin', workspace=256, memonger=False, **kwargs):
    
    
    data = mx.symbol.Variable(name='data')
    ## heat map of human parts
    heatmaplabel = mx.sym.Variable("heatmaplabel")
    ## part affinity graph
    partaffinityglabel = mx.sym.Variable('partaffinityglabel')
    heatweight = mx.sym.Variable('heatweight')
    vecweight = mx.sym.Variable('vecweight')
    
    
    
    if repetition<=0:
        return data
    
    body = mx.sym.Convolution(data=data, num_filter=64, kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    
    
    body = mx.sym.Convolution(data=body, num_filter=128, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv1", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu1')
    '''
    body = mx.sym.Convolution(data=body, num_filter=128, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv2", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn2')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu2')
    
    body = mx.sym.Convolution(data=body, num_filter=128, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv3", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn3')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu3')
    '''
    
    body = mx.sym.Convolution(data=body, num_filter=num_filter, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv4", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn4')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu4')
    
    
    out = body
    
    k = 2**layout_layer
    
    partaffinityglabelr = mx.symbol.Reshape(data=partaffinityglabel, shape=(-1, ), name='partaffinityglabelr')
    vecweightw = mx.symbol.Reshape(data=vecweight, shape=(-1,), name='vecweightw')
    heatmaplabelr = mx.symbol.Reshape(data=heatmaplabel, shape=(-1, ), name='heatmaplabelr')
    heatweightw = mx.symbol.Reshape(data=heatweight, shape=(-1,), name='heatweightw')
    
    grouped = []
    for i in range(repetition):
        
        out  = hourglass(out, num_filter, 'hourglass_{0}_'.format(i), \
                               codec_structure=codec_structure, \
                               bn_mom = bn_mom, unitbatchnorm=unitbatchnorm, expandmode=expandmode,\
                               workspace=workspace, memonger=memonger)
        
        prepare1 = mx.symbol.Convolution(out, num_filter=num_filter, kernel=(1,1), 
                                              no_bias=True, workspace=workspace, name=name + 'hgprepare1_'+str(i))
        prepare2 = mx.symbol.Convolution(out, num_filter=num_filter, kernel=(1,1), 
                                              no_bias=True, workspace=workspace, name=name + 'hgprepare2_'+str(i))
        
        out1 = mx.symbol.Pooling(prepare1, kernel=(k,k), stride=(k,k), pad=(0,0), pool_type="max")
        out2 = mx.symbol.Pooling(prepare2, kernel=(k,k), stride=(k,k), pad=(0,0), pool_type="max")
            
        shortcut1 = mx.symbol.Convolution(out1, num_filter=numoflinks*2, kernel=(1,1), 
                                              no_bias=True, workspace=workspace, name=name + 'shortcut1_'+str(i))
        shortcut2 = mx.symbol.Convolution(out2, num_filter=numofparts, kernel=(1,1),
                                              no_bias=True, workspace=workspace, name=name + 'shortcut2_'+str(i))
        
    
    
        shortcut1r = mx.symbol.Reshape(data=shortcut1, shape=(-1,), name='shortcut1r')
        shortcut1_loss = mx.symbol.square(shortcut1r-partaffinityglabelr)
    
        attention1= shortcut1_loss*vecweightw
        Loss1  = mx.symbol.MakeLoss(attention1)
        shortcut2r = mx.symbol.Reshape(data=shortcut2, shape=(-1,), name='shortcut2r')
        shortcut2_loss = mx.symbol.square(shortcut2r-heatmaplabelr)
        attention2 = shortcut2_loss*heatweightw
        Loss2  = mx.symbol.MakeLoss(attention2)
    
        grouped.extend([Loss1,Loss2])
        
    return mx.symbol.Group(grouped)
        
    
    
    
    



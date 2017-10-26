import mxnet as mx
from symbol_dpn import *

k_R = 96

G   = 32

k_sec  = {  2: 3, \
            3: 4, \
            4: 20, \
            5: 3   }

inc_sec= {  2: 16, \
            3: 32, \
            4: 24, \
            5: 128 }




#stride_plan = [1,2,1,1]
#dilation_plan = [1,1,2,4]


def get_before_pool(taskmode='CLS'):

    
    if taskmode == 'CLS' :
        stride_plan = [1,2,2,2]
        dilation_plan = [1,1,1,1]
        
    elif taskmode == 'KEY' :
        stride_plan = [1,2,1,1]
        dilation_plan = [1,1,2,4]
    
    ## define Dual Path Network
    data = mx.symbol.Variable(name="data")

    # conv1
    conv1_x_1  = Conv(data=data,  num_filter=64,  kernel=(7, 7), name='conv1_x_1', pad=(3,3), stride=(2,2))
    conv1_x_1  = BN_AC(conv1_x_1, name='conv1_x_1__relu-sp')
    conv1_x_x  = mx.symbol.Pooling(data=conv1_x_1, pool_type="max", kernel=(3, 3), pad=(1,1), stride=(2,2), name="pool1")

    # conv2
    bw = 256
    inc= inc_sec[2]
    R  = (k_R*bw)/256 
    conv2_x_x  = DualPathFactory(     conv1_x_x,   R,   R,   bw,  'conv2_x__1',           inc,   G,  'proj' if stride_plan[0]==1 else 'down', dilation = dilation_plan[0]  )
    for i_ly in range(2, k_sec[2]+1):
        conv2_x_x  = DualPathFactory( conv2_x_x,   R,   R,   bw, ('conv2_x__%d'% i_ly),   inc,   G,  'normal' , dilation = dilation_plan[0])

    # conv3
    bw = 512
    inc= inc_sec[3]
    R  = (k_R*bw)/256
    conv3_x_x  = DualPathFactory(     conv2_x_x,   R,   R,   bw,  'conv3_x__1',           inc,   G,  'proj' if stride_plan[1]==1 else 'down' , dilation = dilation_plan[1] )
    for i_ly in range(2, k_sec[3]+1):
        conv3_x_x  = DualPathFactory( conv3_x_x,   R,   R,   bw, ('conv3_x__%d'% i_ly),   inc,   G,  'normal', dilation = dilation_plan[1] )

    # conv4
    bw = 1024
    inc= inc_sec[4]
    R  = (k_R*bw)/256
    conv4_x_x  = DualPathFactory(     conv3_x_x,   R,   R,   bw,  'conv4_x__1',           inc,   G,  'proj' if stride_plan[2]==1 else 'down' , dilation = dilation_plan[2] )
    for i_ly in range(2, k_sec[4]+1):
        conv4_x_x  = DualPathFactory( conv4_x_x,   R,   R,   bw, ('conv4_x__%d'% i_ly),   inc,   G,  'normal' , dilation = dilation_plan[2] )

    # conv5
    bw = 2048
    inc= inc_sec[5]
    R  = (k_R*bw)/256
    conv5_x_x  = DualPathFactory(     conv4_x_x,   R,   R,   bw,  'conv5_x__1',           inc,   G,  'proj' if stride_plan[3]==1 else 'down' , dilation = dilation_plan[3] )
    for i_ly in range(2, k_sec[5]+1):
        conv5_x_x  = DualPathFactory( conv5_x_x,   R,   R,   bw, ('conv5_x__%d'% i_ly),   inc,   G,  'normal' , dilation = dilation_plan[3] )

    # output: concat
    conv5_x_x  = mx.symbol.Concat(*[conv5_x_x[0], conv5_x_x[1]],  name='conv5_x_x_cat-final')
    conv5_x_x = BN_AC(conv5_x_x, name='conv5_x_x__relu-sp')
    return conv5_x_x


def get_key():
    before_pool = get_before_pool(taskmode='KEY')
    return before_pool


def get_linear(num_classes = 1000):
    before_pool = get_before_pool()
    # - - - - -
    pool5     = mx.symbol.Pooling(data=before_pool, pool_type="avg", global_pool='True', kernel=(7, 7), stride=(1,1), name="pool5")
    flat5     = mx.symbol.Flatten(data=pool5, name='flatten')
    fc6       = mx.symbol.FullyConnected(data=flat5, num_hidden=num_classes, name='fc6')
    return fc6

def get_symbol(num_classes = 1000, **kwargs):
    fc6       = get_linear(num_classes)
    softmax   = mx.symbol.SoftmaxOutput( data=fc6,  name='softmax')
    sys_out   = softmax
    return sys_out

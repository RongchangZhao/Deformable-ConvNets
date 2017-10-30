import sys
sys.path.append('./')
from modelCPMWeight2 import *
from config.config import config
from symbols.irnext_v2_deeplab_v3_dcn_w_hypers import *
from symbols.dpns import *
from symbols.inceptions import *
import argparse


class AIChallengerIterweightBatch:
    def __init__(self, datajson,
                 data_names, data_shapes, label_names,
                 label_shapes, batch_size = 1):

        self._data_shapes = data_shapes
        self._label_shapes = label_shapes
        self._provide_data = zip([data_names], [data_shapes])
        self._provide_label = zip(label_names, label_shapes) * 6
        self._batch_size = batch_size

        with open(datajson, 'r') as f:
            data = json.load(f)

        self.num_batches = len(data)

        self.data = data
        
        self.cur_batch = 0

        self.keys = data.keys()

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:
            
            transposeImage_batch = []
            heatmap_batch = []
            pagmap_batch = []
            heatweight_batch = []
            vecweight_batch = []
            
            for i in range(batch_size):
                if self.cur_batch >= 45174:
                    break
                image, mask, heatmap, pagmap = getImageandLabel(self.data[self.keys[self.cur_batch]])
                maskscale = mask[0:368:8, 0:368:8, 0]
                heatweight = np.ones((numofparts, 46, 46))
                vecweight = np.ones((numoflinks*2, 46, 46))

                for i in range(numofparts):
                    heatweight[i,:,:] = maskscale

                for i in range(numoflinks*2):
                    vecweight[i,:,:] = maskscale
                
                transposeImage = np.transpose(np.float32(image), (2,0,1))/256 - 0.5
            
                self.cur_batch += 1
                
                transposeImage_batch.append(transposeImage)
                heatmap_batch.append(heatmap)
                pagmap_batch.append(pagmap)
                heatweight_batch.append(heatweight)
                vecweight_batch.append(vecweight)
                
            return DataBatchweight(mx.nd.array(transposeImage_batch),
                                   mx.nd.array(heatmap_batch),
                                   mx.nd.array(pagmap_batch),
                                   mx.nd.array(heatweight_batch),
                                   mx.nd.array(vecweight_batch))
        else:
            raise StopIteration

start_prefix = 0

class poseModule(mx.mod.Module):

    def fit(self, train_data, num_epoch, batch_size, carg_params=None, begin_epoch=0):
        
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=[('data', (batch_size, 3, 368, 368))], label_shapes=[
        ('heatmaplabel', (batch_size, numofparts, 46, 46)),
        ('partaffinityglabel', (batch_size, numoflinks*2, 46, 46)),
        ('heatweight', (batch_size, numofparts, 46, 46)),
        ('vecweight', (batch_size, numoflinks*2, 46, 46))])
   
        
        # self.init_params(mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=1))
        # mx.initializer.Uniform(scale=0.07),
        # mx.initializer.Uniform(scale=0.01)
        # mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=0.01)
        self.init_params(arg_params = carg_params, aux_params={}, allow_missing = True)
        #self.set_params(arg_params = carg_params, aux_params={},
        #                allow_missing = True)
        self.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.00004), ))
        losserror_list = []

        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            i=0
            sumerror=0
            while not end_of_batch:
                data_batch = next_data_batch
                cmodel.forward(data_batch, is_train=True)       # compute predictions  
                prediction=cmodel.get_outputs()
                i=i+1
                sumloss=0
                numpixel=0
                print 'iteration in epoch: ', i
                
                '''
                print 'length of prediction:', len(prediction)
                for j in range(len(prediction)):
                    
                    lossiter = prediction[j].asnumpy()
                    cls_loss = np.sum(lossiter)
                    print 'loss: ', cls_loss
                    sumloss += cls_loss
                    numpixel +=lossiter.shape[0]
                    
                '''
                
                lossiter = prediction[1].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                sumerror = sumerror + cls_loss
                print 'start heat: ', cls_loss
                    
                lossiter = prediction[0].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                sumerror = sumerror + cls_loss
                print 'start paf: ', cls_loss
                
                lossiter = prediction[-1].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                sumerror = sumerror + cls_loss
                print 'end heat: ', cls_loss
                
                lossiter = prediction[-2].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                sumerror = sumerror + cls_loss
                print 'end paf: ', cls_loss   
               
         
                cmodel.backward()   
                self.update()           
                
                #if i > 10:
                #    break
                    
                try:
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch)
                except StopIteration:
                    end_of_batch = True
                nbatch += 1
            
                    
            print '------Error-------'
            print sumerror/i
            losserror_list.append(sumerror/i)
            
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)
            #self.save_checkpoint(config.TRAIN.output_model, epoch)
            
            train_data.reset()
        print losserror_list
        text_file = open("OutputLossError.txt", "w")
        text_file.write(' '.join([str(i) for i in losserror_list]))
        text_file.close()
        


def init_from_irnext_cls(ctx, irnext_cls_symbol, irnext_cls_args, irnext_cls_auxs, data_shape_dict, block567=False):
    
    deeplab_args = irnext_cls_args.copy()
    deeplab_auxs = irnext_cls_args.copy()
    
    arg_name = irnext_cls_symbol.list_arguments()
    aux_name = irnext_cls_symbol.list_auxiliary_states()
    arg_shape, _, aux_shape = irnext_cls_symbol.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(arg_name, arg_shape))
    aux_shape_dict = dict(zip(aux_name, aux_shape))

    
    deeplab_args = dict({k:deeplab_args[k] for k in deeplab_args if (('fc' not in k) and ('fullyconnected' not in k)) })
    
    print deeplab_args.keys()
    
    for k,v in deeplab_args.items():
        if(v.context != ctx):
            deeplab_args[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(deeplab_args[k])
        if k.startswith('fc6_'):
            if k.endswith('_weight'):
                print('initializing',k)
                deeplab_args[k] = mx.random.normal(0, 0.01, shape=v)
            elif k.endswith('_bias'):
                print('initializing',k)
                deeplab_args[k] = mx.nd.zeros(shape=v)
        if block567:
            if k.startswith('stage'):
                stage_id = int(k[5])
            if stage_id>4:
                rk = "stage4"+k[6:]
                if rk in irnext_cls_args:
                    print('initializing', k, rk)
                    if arg_shape_dict[rk]==v:
                        deeplab_args[k] = deeplab_args[rk].copy()
                    else:
                        if k.endswith('_beta'):
                            deeplab_args[k] = mx.nd.zeros(shape=v)
                        elif k.endswith('_gamma'):
                            deeplab_args[k] = mx.nd.random_uniform(shape=v)
                        else:
                            deeplab_args[k] = mx.random.normal(0, 0.01, shape=v)
        if 'se' in k:
            deeplab_args[k] = mx.nd.zeros(shape=v)
        if 'offset' in k:
            if 'weight' in k:
                deeplab_args[k] = mx.random.normal(0, 0.01, shape=v)
            elif 'bias' in k:
                deeplab_args[k] = mx.nd.zeros(shape=v)
        
        
        
    for k,v in deeplab_auxs.items():
        if(v.context != ctx):
            deeplab_auxs[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(deeplab_auxs[k])
        if block567:
            if k.startswith('stage'):
                stage_id = int(k[5])
            if stage_id>4:
                rk = "stage4"+k[6:]
                if rk in irnext_cls_auxs:
                    print('initializing', k, rk)
                    if aux_shape_dict[rk]==v:
                        deeplab_args[k] = deeplab_args[rk].copy()
                    else:
                        if k.endswith('_beta'):
                            deeplab_args[k] = mx.nd.zeros(shape=v)
                        elif k.endswith('_gamma'):
                            deeplab_args[k] = mx.nd.random_uniform(shape=v)
                        else:
                            deeplab_args[k] = mx.random.normal(0, 0.01, shape=v)
        if 'se' in k:
            deeplab_args[k] = mx.nd.zeros(shape=v)
        if 'offset' in k:
            if 'weight' in k:
                deeplab_args[k] = mx.random.normal(0, 0.01, shape=v)
            elif 'bias' in k:
                deeplab_args[k] = mx.nd.zeros(shape=v)

    
    data_shape=(32,3,368,368)
    arg_names = irnext_cls_symbol.list_arguments()
    print arg_names
    print "Step"
    arg_shapes, _, _ = irnext_cls_symbol.infer_shape(**data_shape_dict) # data=data_shape
    print zip(arg_names,arg_shapes)
    '''
    rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
            if x[0] in ['score_weight', 'score_bias', 'score_pool4_weight', 'score_pool4_bias', \
                        'score_pool3_weight', 'score_pool3_bias', 'score_0_weight', 'score_0_bias', \
                        'score_1_weight', 'score_1_bias', 'score_2_weight', 'score_2_bias', \
                        'score_3_weight', 'score_3_bias']])
    deeplab_args.update(rest_params)
    print "Step"
    deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes)
            if x[0] in ["upsampling_weight"]])
    
    for k, v in deconv_params.items():
        filt = upsample_filt(v[3])
        initw = np.zeros(v)
        initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        deeplab_args[k] = mx.nd.array(initw, ctx)
    '''
    return deeplab_args, deeplab_auxs
        
        
## MAKE SYMBOL

parser = argparse.ArgumentParser(description="train imagenet1k",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.set_defaults(
        # network
        network          = 'irnext',
        num_layers       = 152,
        outfeature       = 2048,
        bottle_neck      = 1,
        expansion        = 4, 
        num_group        = 1,
        dilpat           = '',#'DEEPLAB.HEAD', 
        irv2             = False, 
        deform           = 0,
        sqex             = 0,
        ratt             = 0,
        usemaxavg        = 0,
        scale            = 1, # 0.25, 
        block567         = 0,
        lmar             = 0,
        lmarbeta         = 1000,
        lmarbetamin      = 0,
        lmarscale        = 0.9997,
        # data
        
        train_image_root = '/data1/deepinsight/aichallenger/scene',
        val_image_root = '/data1/deepinsight/aichallenger/scene',
        
        #num_classes      = 80,
        #num_examples     = 53878,
        #image_shape      = '3,224,224',
        #lastout          = 7,
        min_random_scale = 1.0 , # if input image has min size k, suggest to use
                              # 256.0/x, e.g. 0.533 for 480
        # train
        batch_size       = 32,
        num_epochs       = 100,
        lr               = 0.003,
        lr_step_epochs   = '30,60',
        gpus             = '0,1,2,3,4,5,6,7',
        #dtype            = 'float32',
        
        # load , please tune
    
        load_ft_epoch       = 0,
        model_ft_prefix     = '/home/deepinsight/frankwang/Deformable-ConvNets/deeplab/runs_CAIScene/CLS-ResNeXt-152L64X1D4XP'
            
)

args = parser.parse_args()

sym = CPMModel(**vars(args)) 

## Load parameters from RESNET
_ , arg_params, aux_params = mx.model.load_checkpoint(args.model_ft_prefix, args.load_ft_epoch)


## Init
ctx = mx.cpu()
data_shape_dict = {'data': (args.batch_size, 3, 368, 368), \
                   'heatmaplabel': (args.batch_size, numofparts, 46, 46), \
                   'partaffinityglabel': (args.batch_size, numoflinks*2, 46, 46),
                   'heatweight': (args.batch_size, numofparts, 46, 46),
                   'vecweight': (args.batch_size, numoflinks*2, 46, 46)}
#arg_params, aux_params = init_from_irnext_cls(ctx, \
#                            sym, arg_params, aux_params, data_shape_dict, block567=args.block567)


fixshape = {}
for k,v in arg_params.iteritems():
    fixshape[k] = v.shape

arg_name = sym.list_arguments()
arg_shape, _, aux_shape = sym.infer_shape(**data_shape_dict)
arg_shape_dict = dict(zip(arg_name, arg_shape))

for k,v in arg_shape_dict.items():
  if k in fixshape and v!=fixshape[k]:
    print('a',k,fixshape[k])
    print('b',k,v)

'''
newargs = {}
for ikey in config.TRAIN.vggparams:
    newargs[ikey] = arg_params[ikey]
'''

batch_size = args.batch_size
aidata = AIChallengerIterweightBatch('pose_io/AI_data_train.json', # 'pose_io/COCO_data.json',
                          'data', (batch_size, 3, 368, 368),
                          ['heatmaplabel','partaffinityglabel','heatweight','vecweight'],
                          [(batch_size, numofparts, 46, 46),
                           (batch_size, numoflinks*2, 46, 46),
                           (batch_size, numofparts, 46, 46),
                           (batch_size, numoflinks*2, 46, 46)])

# 
print "Start Pose Module"



cmodel = poseModule(symbol=sym, context= [mx.gpu(int(i)) for i in args.gpus.split(',')],
                    label_names=['heatmaplabel',
                                 'partaffinityglabel',
                                 'heatweight',
                                 'vecweight'])
starttime = time.time()

#print sym

'''
output_prefix = config.TRAIN.output_model
testsym, newargs, aux_params = mx.model.load_checkpoint(output_prefix, start_prefix)
'''


print "Start Fit"
cmodel.fit(aidata, num_epoch = args.num_epochs, batch_size = batch_size, carg_params = arg_params)
print "End Fit "

cmodel.save_checkpoint(config.TRAIN.output_model, start_prefix + args.num_epochs)
endtime = time.time()

print 'cost time: ', (endtime-starttime)/60


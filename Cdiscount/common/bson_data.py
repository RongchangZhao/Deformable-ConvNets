import mxnet as mx
import random
import sys
import copy
from mxnet.io import DataBatch, DataIter
import numpy as np
from mxnet.image import *
import bson
import struct
import pickle
import logging

def add_data_args(parser):
    data = parser.add_argument_group('Data', 'the input images')
    data.add_argument('--data-dir', type=str, default='./data',
                      help='the data dir')
    data.add_argument('--train-image-root', type=str)
    data.add_argument('--val-image-root', type=str)
    #data.add_argument('--data-train', type=str, help='the training data')
    #data.add_argument('--data-val', type=str, help='the validation data')
    data.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939',
                      help='a tuple of size 3 for the mean rgb')
    data.add_argument('--pad-size', type=int, default=0,
                      help='padding the input image')
    data.add_argument('--image-shape', type=str,
                      help='the image shape feed into the network, e.g. (3,224,224)')
    data.add_argument('--num-classes', type=int, help='the number of classes')
    #data.add_argument('--num-examples', type=int, help='the number of training examples')
    data.add_argument('--data-nthreads', type=int, default=4,
                      help='number of threads for data decoding')
    data.add_argument('--benchmark', type=int, default=0,
                      help='if 1, then feed the network with synthetic data')
    data.add_argument('--dtype', type=str, default='float32',
                      help='data type: float32 or float16')
    return data

def add_data_aug_args(parser):
    aug = parser.add_argument_group(
        'Image augmentations', 'implemented in src/io/image_aug_default.cc')
    aug.add_argument('--random-crop', type=int, default=1,
                     help='if or not randomly crop the image')
    aug.add_argument('--random-mirror', type=int, default=1,
                     help='if or not randomly flip horizontally')
    aug.add_argument('--max-random-h', type=int, default=0,
                     help='max change of hue, whose range is [0, 180]')
    aug.add_argument('--max-random-s', type=int, default=0,
                     help='max change of saturation, whose range is [0, 255]')
    aug.add_argument('--max-random-l', type=int, default=0,
                     help='max change of intensity, whose range is [0, 255]')
    aug.add_argument('--max-random-aspect-ratio', type=float, default=0,
                     help='max change of aspect ratio, whose range is [0, 1]')
    aug.add_argument('--max-random-rotate-angle', type=int, default=0,
                     help='max angle to rotate, whose range is [0, 360]')
    aug.add_argument('--max-random-shear-ratio', type=float, default=0,
                     help='max ratio to shear, whose range is [0, 1]')
    aug.add_argument('--max-random-scale', type=float, default=1,
                     help='max ratio to scale')
    aug.add_argument('--min-random-scale', type=float, default=1,
                     help='min ratio to scale, should >= img_size/input_shape. otherwise use --pad-size')
    return aug

def set_data_aug_level(aug, level):
    if level >= 1:
        aug.set_defaults(random_crop=1, random_mirror=1)
    if level >= 2:
        aug.set_defaults(max_random_h=36, max_random_s=50, max_random_l=50)
    if level >= 3:
        aug.set_defaults(max_random_rotate_angle=10, max_random_shear_ratio=0.1, max_random_aspect_ratio=0.25)




class BsonImageIter(io.DataIter):

    def __init__(self, batch_size, data_shape, 
                 path_bson=None, path_index = None, path_labelmap = None,
                 shuffle=False, part_index=0, num_parts=1, aug_list=None,
                 smooth_param = '', rgb_mean = None,
                 data_name='data', label_name='softmax_label', 
                 **kwargs):
        super(BsonImageIter, self).__init__()
        assert path_bson
        assert path_index is None or len(path_index)>0
        num_threads = os.environ.get('MXNET_CPU_WORKER_NTHREADS', 1)
        logging.info('Using %s threads for decoding...', str(num_threads))
        #logging.info('Set enviroment variable MXNET_CPU_WORKER_NTHREADS to a'
        #             ' larger number to use more threads.')
        class_name = self.__class__.__name__
        self.seq = []
        self.path_bson = path_bson
        self.inputf = open(path_bson, 'rb')
        f = self.inputf
        item_data = []
        length_size = 4
        offset = 0
        print('loading bson offset..')
        if path_index is not None and os.path.exists(path_index):
          print('loading index')
          self.seq = pickle.load(open(path_index, 'rb'))
          print('index loaded')
        else:
          while True:        
            f.seek(offset)
            item_length_bytes = f.read(length_size)     
            if len(item_length_bytes) == 0:
              break                
            # Decode item length:
            length = struct.unpack("<i", item_length_bytes)[0]
            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length, "%i vs %i" % (len(item_data), length)
            #item = bson.BSON.decode(item_data)
            item = bson.BSON(item_data).decode()
            #print(item)
            img_size = len(item['imgs'])

            for i in xrange(img_size):
              self.seq.append( (offset, length, i) )
            offset += length 
          self.inputf.close()
          if path_index is not None:
            pickle.dump(self.seq, open(path_index, 'wb'), pickle.HIGHEST_PROTOCOL)
        print('loaded, item count', len(self.seq))
        self.labelmap = {}
        print('loading labelmap')
        if path_labelmap is not None:
          for line in open(path_labelmap, 'r'):
            vec = line.strip().split()
            self.labelmap[int(vec[0])] = int(vec[1])
        print('labelmap loaded')

        self.rgb_mean = rgb_mean
        if self.rgb_mean:
          self.rgb_mean = np.array(self.rgb_mean, dtype=np.float32).reshape(1,1,3)
          self.rgb_mean = nd.array(self.rgb_mean)
        if len(smooth_param)==0:
          self.label_width = 1
          self.provide_label = [(label_name, (batch_size, ))]
          self.smoothed_label = None
        else:
          _vec = smooth_param.split(',')
          assert(len(_vec)==4)
          self.confusion_matrix = np.load(_vec[0])
          print(self.confusion_matrix.shape)
          self.smoothed_label = np.zeros( self.confusion_matrix.shape, dtype=np.float32)
          LS_K = int(_vec[1])
          LS_A = float(_vec[2])
          LS_B = float(_vec[3])
          for i in xrange(self.confusion_matrix.shape[0]):
            am = np.argsort(self.confusion_matrix[i])[::-1]
            assert i==am[0]
            self.smoothed_label[i][i] = 1.0-LS_A-LS_B
            for j in xrange(1, LS_K):
              self.smoothed_label[i][am[j]] += LS_A/(LS_K-1)
            for j in xrange(LS_K, len(am)):
              self.smoothed_label[i][am[j]] += LS_B/(len(am)-LS_K)
          self.label_width = self.smoothed_label.shape[0]
          self.provide_label = [(label_name, (batch_size, self.label_width))]


        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape

        self.shuffle = shuffle
        if aug_list is None:
            self.auglist = CreateAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        self.buffer = None
        self.cur = [0,0]
        self.reset()


    def num_samples(self):
      return len(self.seq)

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        if self.shuffle:
          random.shuffle(self.seq)
        self.cur = [0,0]
        self.inputf.seek(0)

    def _next_sample(self):
        """Helper function for reading in next sample."""
        while True:
          if self.cur[0] >= len(self.seq):
              raise StopIteration
          if self.cur[1]==0:
            offset, length = self.seq[self.cur[0]]
            self.inputf.seek(offset)
            content = self.inputf.read(length)
            #item = bson.BSON.decode(content)
            item = bson.BSON(content).decode()
            self.buffer = item
          else:
            item = self.buffer
          if self.cur[1]>=len(item['imgs']):
            self.cur[0]+=1
            self.cur[1] = 0
            continue
          label = item['_id'] #used in test mode
          if 'category_id' in item:
            label = item['category_id']
            label = self.labelmap[label]
          pic = item['imgs'][self.cur[1]]
          img = pic['picture']
          #print(img.__class__)
          self.cur[1]+=1
          return float(label), img

    def __next_sample(self):
        """Helper function for reading in next sample."""
        if self.cur[0] >= len(self.seq):
            raise StopIteration

        offset, length, idx = self.seq[self.cur[0]]
        self.cur[0]+=1
        f = self.inputf
        if idx==0:
          f.seek(offset)
          print('seek',offset)
          content = f.read(length)
          #item = bson.BSON.decode(content)
          item = bson.BSON(content).decode()
          self.buffer = item
        else:
          item = self.buffer
        label = item['_id'] #used in test mode
        if 'category_id' in item:
          label = item['category_id']
        pic = item['imgs'][idx]
        img = pic['picture']
        #print(img.__class__)
        return float(label), img

    def next_sample(self):
        """Helper function for reading in next sample."""
        if self.cur[0] >= len(self.seq):
            raise StopIteration

        offset, length, idx = self.seq[self.cur[0]]
        self.cur[0]+=1
        with open(self.path_bson, 'rb') as f:
          f.seek(offset)
          #print('seek',offset)
          content = f.read(length)
        #item = bson.BSON.decode(content)
        item = bson.BSON(content).decode()
        label = item['_id'] #used in test mode
        if 'category_id' in item:
          label = item['category_id']
        pic = item['imgs'][idx]
        img = pic['picture']
        #print(img.__class__)
        return float(label), img

    def next(self):
        """Returns the next batch of data."""
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, s = self.next_sample()
                data = self.imdecode(s)
                if self.rgb_mean is not None:
                  data = nd.cast(data, dtype='float32')
                  #print('apply mean', self.rgb_mean)
                  data -= self.rgb_mean
                  data *= 0.0078125
                  #_npdata = data.asnumpy()
                  #_npdata = _npdata.astype(np.float32)
                  #_npdata -= self.mean
                  #_npdata *= 0.0078125
                  #data = mx.nd.array(_npdata)
                #print(data.shape)
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                data = self.augmentation_transform(data)
                assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                batch_data[i] = self.postprocess_data(data)
                if self.smoothed_label is None:
                  batch_label[i] = label
                else:
                  _label = int(label.asnumpy()[0])
                  _label = self.smoothed_label[_label]
                  batch_label[i] = nd.array(_label)
                i += 1
        except StopIteration:
            if i==0:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size - i)

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        return imdecode(s)

    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.

        Example usage:
        ----------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img

    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        for aug in self.auglist:
            data = aug(data)
        return data

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))


def get_rec_iter(args, kv=None):
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    rgb_mean = None
    if len(args.rgb_mean)>0:
      rgb_mean = [float(x) for x in args.rgb_mean.split(',')]
    dtype = np.float32;
    if 'dtype' in args:
        if args.dtype == 'float16':
            dtype = np.float16
    if kv:
        (rank, nworker) = (kv.rank, kv.num_workers)
    else:
        (rank, nworker) = (0, 1)
    #print(rank, nworker, args.batch_size)
    #train_resize = int(image_shape[1]*1.5)
    #train_resize = image_shape[1]+32
    #if train_resize>640:
    #  train_resize = None
    train = ImageIter2(
        path_root	          = args.train_image_root, 
        path_imglist        = args.train_lst,
        #smooth_param        = './data/confusion_matrix.npy,3,0.1,0.1',
        balance             = 2000,
        data_shape          = image_shape,
        batch_size          = args.batch_size,
        #resize              = train_resize,
        rand_crop           = True,
        rand_resize         = True,
        rand_mirror         = True,
        shuffle             = True,
        brightness          = 0.4,
        contrast            = 0.4,
        saturation          = 0.4,
        pca_noise           = 0.1,
        rgb_mean            = rgb_mean,
        #data_name           = 'data_source',
        #label_name          = 'label_source',
        num_parts           = nworker,
        part_index          = rank)
    if args.val_lst is not None:
      val = ImageIter2(
          path_root	          = args.val_image_root, 
          path_imglist        = args.val_lst,
          #smooth_param        = './data/confusion_matrix.npy,1,0.0,0.0',
          batch_size          = args.batch_size,
          data_shape          =  image_shape,
          resize		      = int(image_shape[1]*1.125), 
          rand_crop       = False,
          rand_resize     = False,
          rand_mirror     = False,
          rgb_mean            = rgb_mean,
          num_parts       = nworker,
          part_index      = rank)
    else:
      val = None
    return (train, val)

def test_st():
  it = BsonImageIter(path_bson='/raid5data/dplearn/cdiscount/train.bson', 
      batch_size=128, data_shape=(3,180,180), 
      path_index='/raid5data/dplearn/cdiscount/train.bson.index',
      path_labelmap='/raid5data/dplearn/cdiscount/synset.txt',
      )
  i = 0
  while True:
    i+=1
    if i%100==0:
      print('i',i)
    batch = it.next()
    data = batch.data[0]
    label = batch.label[0]
    #print(label)
    #print(data.shape)


if __name__ == '__main__':
  os.environ['MXNET_CPU_WORKER_NTHREADS'] = '15'
  logging.basicConfig(level=logging.DEBUG)
  test_st()



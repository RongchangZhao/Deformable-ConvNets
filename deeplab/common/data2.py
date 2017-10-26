import mxnet as mx
import random
import sys
import copy
from mxnet.io import DataBatch, DataIter
import numpy as np
from mxnet.image import *

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




class ImageIter2(io.DataIter):

    def __init__(self, batch_size, data_shape, 
                 path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None,
                 shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None,
                 balance = 0, smooth_param = '', rgb_mean = None,
                 data_name='data', label_name='softmax_label', 
                 **kwargs):
        super(ImageIter2, self).__init__()
        assert path_imgrec or path_imglist or (isinstance(imglist, list))
        num_threads = os.environ.get('MXNET_CPU_WORKER_NTHREADS', 1)
        logging.info('Using %s threads for decoding...', str(num_threads))
        #logging.info('Set enviroment variable MXNET_CPU_WORKER_NTHREADS to a'
        #             ' larger number to use more threads.')
        class_name = self.__class__.__name__
        self.imgrec = None

        if path_imglist:
            logging.info('%s: loading image list %s...', class_name, path_imglist)
            with open(path_imglist) as fin:
                imglist = {}
                imgkeys = []
                for line in iter(fin.readline, ''):
                    line = line.strip().split('\t')
                    label = nd.array([float(i) for i in line[1:-1]])
                    key = int(line[0])
                    imglist[key] = (label, line[-1])
                    imgkeys.append(key)
                self.imglist = imglist
        elif isinstance(imglist, list):
            logging.info('%s: loading image list...', class_name)
            result = {}
            imgkeys = []
            index = 1
            for img in imglist:
                key = str(index)  # pylint: disable=redefined-variable-type
                index += 1
                if len(img) > 2:
                    label = nd.array(img[:-1])
                elif isinstance(img[0], numeric_types):
                    label = nd.array([img[0]])
                else:
                    label = nd.array(img[0])
                result[key] = (label, img[-1])
                imgkeys.append(str(key))
            self.imglist = result
        else:
            self.imglist = None
        self.path_root = path_root
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
        self.seq = imgkeys
        self.oseq = copy.copy(self.seq)
        self.balance = balance
        if self.balance>0:
          assert(self.shuffle)
        #self.balance()

        if num_parts > 1:
            assert part_index < num_parts
            N = len(self.seq)
            C = N // num_parts
            self.seq = self.seq[part_index * C:(part_index + 1) * C]
        if aug_list is None:
            self.auglist = CreateAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        self.cur = 0
        self.reset()

    def do_balance(self):
      label_dist = {}
      for idx in self.oseq:
        _label = int(self.imglist[idx][0].asnumpy()[0])
        #print(idx, _label)
        v = label_dist.get(_label, [])
        v.append(idx)
        label_dist[_label] = v
      items = sorted(label_dist.items(), key = lambda x : len(x[1]), reverse=True)
      self.seq = []
      tcount = min(len(items[0][1]), self.balance)
      print('tcount', tcount)
      for item in items:
        _label = item[0]
        v = item[1]
        random.shuffle(v)
        _tcount = tcount
        #_tcount = len(v)
        for i in xrange(_tcount):
          ii = i%len(v)
          idx = v[ii]
          self.seq.append(idx)
      print(len(self.seq))
      for i in xrange(self.batch_size):
        if len(self.seq)%self.batch_size==0:
          break
        ii = i%len(items)
        idx = items[ii][1][0]
        self.seq.append(idx)
      random.shuffle(self.seq)
      print(len(self.seq))

    def num_samples(self):
      return len(self.seq)

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        if self.shuffle:
          if self.balance>0:
            self.do_balance()
          else:
            random.shuffle(self.seq)
        self.cur = 0

    def next_sample(self):
        """Helper function for reading in next sample."""
        if self.seq is not None:
            if self.cur >= len(self.seq):
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            if self.imgrec is not None:
                s = self.imgrec.read_idx(idx)
                header, img = recordio.unpack(s)
                if self.imglist is None:
                    return header.label, img
                else:
                    return self.imglist[idx][0], img
            else:
                label, fname = self.imglist[idx]
                return label, self.read_image(fname)
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img

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

def read_lst(file):
  ret = []
  with open(file, 'r') as f:
    for line in f:
      vec = line.strip().split("\t")
      label = int(vec[1])
      img_path = vec[2]
      ret.append( [label, img_path] )
  return ret

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
    if not args.retrain:
      train = ImageIter2(
          path_root	          = args.train_image_root, 
          path_imglist        = args.train_lst,
          #smooth_param        = './data/confusion_matrix.npy,3,0.1,0.1',
          balance             = 1000,
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
    else:
      #rnd = random.randint(0,1)
      #if rnd==1:
      #  train_resize = None
      train_resize = None
      train = ImageIter2(
          path_root	          = args.train_image_root, 
          path_imglist        = args.train_lst,
          smooth_param        = './data/confusion_matrix.npy,3,0.1,0.1',
          balance             = 9999999,
          data_shape          = image_shape,
          batch_size          = args.batch_size,
          resize              = train_resize,
          rand_resize         = True,
          rand_crop           = True,
          rand_mirror         = True,
          shuffle             = True,
          brightness          = 0.1,
          contrast            = 0.1,
          saturation          = 0.1,
          pca_noise           = 0.1,
          #data_name           = 'data_source',
          #label_name          = 'label_source',
          num_parts           = nworker,
          part_index          = rank)
      val = ImageIter2(
          path_root	          = args.val_image_root, 
          path_imglist        = args.val_lst,
          smooth_param        = './data/confusion_matrix.npy,1,0.0,0.0',
          batch_size          = args.batch_size,
          data_shape          =  image_shape,
          resize		      = int(image_shape[1]*1.125), 
          rand_crop       = False,
          rand_resize     = False,
          rand_mirror     = False,
          num_parts       = nworker,
          part_index      = rank)
    return (train, val)

def test_st():
  pass


if __name__ == '__main__':
  test_st()



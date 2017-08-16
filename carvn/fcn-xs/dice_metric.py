import mxnet as mx
import numpy as np

class DiceMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(DiceMetric, self).__init__(
        'dice-coef', axis=self.axis,
        output_names=None, label_names=None)

  def update(self, labels, preds):
    for label, pred_label in zip(labels, preds):
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy().astype('int32').flatten()
        #print(label)
        assert label.shape==pred_label.shape
        self.num_inst += 1.0
        pred_label_sum = np.sum(pred_label)
        label_sum = np.sum(label)
        if pred_label_sum==0 and label_sum==0:
          self.sum_metric += 1.0
        else:
          intersection = np.sum(pred_label * label)
          ret = (2. * intersection) / (pred_label_sum + label_sum)
          self.sum_metric += ret

class AccMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(AccMetric, self).__init__(
        'acc', axis=self.axis,
        output_names=None, label_names=None)

  def update(self, labels, preds):
    for label, pred_label in zip(labels, preds):
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy().astype('int32').flatten()
        #print(label)
        assert label.shape==pred_label.shape
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)


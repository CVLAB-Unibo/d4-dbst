import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class ConfusionMatrix():
    """Constructs a confusion matrix for a multi-class classification problems.
    Does not support multi-label, multi-class problems.
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix
        The shape of the confusion matrix is K x K, where K is the number
        of classes.
        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.
        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        if np.ndim(target) != 1:
            assert target.shape[1] == self.num_classes, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (target.max() < self.num_classes) and (target.min() >= 0), \
                'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


class IoU():
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).
    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:
        IoU = true_positive / (true_positive + false_positive + false_negative).
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None, valid=19, **kwargs):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)
        self.valid = valid
        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.
        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.
        """
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):
        """Computes the IoU and mean IoU.
        The mean computation ignores NaN elements of the IoU array.
        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf_matrix[:, self.ignore_index] = 0
                conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        
        # print('===> mIoU13: ' + str(round(np.mean(iou[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)))
        # print('===> mIoU16: ' + str(round(np.mean(iou[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)))
        # print('===> mIoU19: ' + str(round(np.nanmean(iou) * 100, 2)))

        if self.valid==13:
            return [np.nanmean(iou[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]), 
                    iou[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]].tolist()]
        if self.valid==16:
            return [np.nanmean(iou[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]), 
                                iou[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]].tolist()]
        else:
            return [np.nanmean(iou), iou.tolist()]

class Accuracy():

    def __init__(self, num_classes, normalized=False, ignore_index=None, **kwargs):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):

        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf_matrix[:, self.ignore_index] = 0
                conf_matrix[self.ignore_index, :] = 0

        return [np.diag(conf_matrix).sum() / conf_matrix.sum()]


class MSE():

    def __init__(self, relative=False, max_depth=100, min_depth=0.001, **kwargs):
        super(MSE, self).__init__()
        self.errors = 0
        self._num_examples = 0
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.relative = relative
        self.eps = 1e-6
    
    def reset(self):
        self.errors = 0
        self._num_examples = 0        
    
    def add(self, predicted, target, **kwargs):
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        predicted = predicted.squeeze(axis=1)
        predicted = predicted*self.max_depth
        predicted[predicted<self.min_depth] = self.min_depth
        predicted[predicted>self.max_depth] = self.max_depth
        mask = (target > self.min_depth) & (target < self.max_depth)

        #set elements that we do not want to count to 1 so that error is 0
        predicted[~mask] = 1
        target[~mask] = 1

        if self.relative:
            norms = np.sum(((predicted - target)**2)/target, axis=(1,2))/np.maximum(np.sum(mask, axis=(1,2), dtype=np.float32), self.eps)
        else:
            norms = np.sum((predicted - target)**2, axis=(1,2))/np.maximum(np.sum(mask, axis=(1,2), dtype=np.float32), self.eps)
        
        self.errors += np.sum(norms)
        #take invalid targets. a target is invalid if all the elements of its mask all set to 0. 
        num_invalids = np.sum(np.all(mask==0, axis=(1,2)))
        #invalid mask should not be counted when computing mean
        self._num_examples += predicted.shape[0]-num_invalids

    def value(self):
      error = self.errors / self._num_examples
      return [error]


class MAE(MSE):

    def add(self, predicted, target):
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        predicted = predicted.squeeze(axis=1)
        predicted = predicted*self.max_depth
        predicted[predicted<self.min_depth] = self.min_depth
        predicted[predicted>self.max_depth] = self.max_depth
        mask = (target > self.min_depth) & (target < self.max_depth)
        #set elements that we do not want to count to 1 so that error is 0
        predicted[~mask] = 1
        target[~mask] = 1
        if self.relative:
            norms = np.sum(np.abs(predicted - target)/target, axis=(1,2))/np.maximum(np.sum(mask, axis=(1,2)), self.eps)
        else:
            norms = np.sum(np.abs(predicted - target), axis=(1,2))/np.maximum(np.sum(mask, axis=(1,2)), self.eps)
        self.errors += np.sum(norms)

        #take invalid targets. a target is invalid if all the elements of its mask all set to 0. 
        num_invalids = np.sum(np.all(mask==0, axis=(1,2)))
        #invalid mask should not be counted when computing mean
        self._num_examples += predicted.shape[0]-num_invalids

class RMSE(MSE):
       
    def add(self, predicted, target):
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        predicted = predicted.squeeze(axis=1)
        predicted = predicted*self.max_depth
        predicted[predicted<self.min_depth] = self.min_depth
        predicted[predicted>self.max_depth] = self.max_depth
        mask = (target > self.min_depth) & (target < self.max_depth)

        #set elements that we do not want to count to 0 so that error is 0
        predicted[~mask] = 1
        target[~mask] = 1

        norms = np.sum((predicted - target)**2, axis=(1,2))/np.maximum(np.sum(mask, axis=(1,2)), self.eps)
        norms = np.sqrt(norms)
        self.errors += np.sum(norms)

        #take invalid targets. a target is invalid if all the elements of its mask all set to 0. 
        num_invalids = np.sum(np.all(mask==0, axis=(1,2)))
        #invalid mask should not be counted when computing mean
        self._num_examples += predicted.shape[0]-num_invalids

    def value(self):
      error = self.errors / self._num_examples
      return [error]


class RMSELog(MSE):

    def add(self, predicted, target):
        
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        
        predicted = predicted.squeeze(axis=1)
        predicted = predicted*self.max_depth
        predicted[predicted<self.min_depth] = self.min_depth
        predicted[predicted>self.max_depth] = self.max_depth
        #take valid values
        mask = (target > self.min_depth) & (target < self.max_depth)
        #set invalid values to 1 (error in this case is 0)
        predicted[~mask] = 1
        target[~mask] = 1

        norms = np.sum((np.log(predicted) - np.log(target))**2, axis=(1,2))/np.maximum(np.sum(mask, axis=(1,2)), self.eps)
        norms = np.sqrt(norms)
        self.errors += np.sum(norms)

        #take invalid targets. a target is invalid if all the elements of its mask all set to 0. 
        num_invalids = np.sum(np.all(mask==0, axis=(1,2)))
        #invalid mask should not be counted when computing mean
        self._num_examples += predicted.shape[0]-num_invalids


class Threshold():
    def __init__(self, threshold=1.25, max_depth=100, min_depth=0.001):
        super(Threshold, self).__init__()
        self.errors = 0
        self._num_examples = 0
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.threshold = threshold
        self.eps = 1e-6
    
    def reset(self):
        self.errors = 0
        self._num_examples = 0        
    
    def add(self, predicted, target):
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        predicted = predicted.squeeze(axis=1)
        predicted = predicted*self.max_depth
        predicted[predicted<self.min_depth] = self.min_depth
        predicted[predicted>self.max_depth] = self.max_depth
        
        mask = (target > self.min_depth) & (target < self.max_depth)
        ratios = np.maximum((predicted/np.maximum(target, self.eps)), (target/np.maximum(predicted, self.eps)))
        ratios = np.where(mask, ratios, 0)
        ratios_per_image = np.sum((ratios < self.threshold) & (ratios>=1), axis=(1,2)) / np.maximum(np.count_nonzero(ratios, axis=(1,2)), self.eps)
        self.errors += np.sum(ratios_per_image)

        #the maximum between a ratio and its iverse its always positive, hence if this is not the case the corresponding target is ivalid
        self._num_examples += np.count_nonzero(ratios_per_image)
        
        # test = []
        # for i in range(len(predicted)):
        #     mask = (target[i] > self.min_depth) & (target[i] < self.max_depth)
        #     ratio = np.maximum((predicted[i][mask]/target[i][mask]), (target[i][mask]/predicted[i][mask]))
        #     test.append(np.mean(ratio<self.threshold))
        # print(test, ratios_per_image)


    def value(self):
      error = self.errors / self._num_examples
      return [error]

def get_metrics(metrics_name, params):
    if metrics_name=='iou':
        return IoU(num_classes=params.num_classes+1, ignore_index=params.ignore_index, valid=19)
    if metrics_name=='accuracy':
        return Accuracy(num_classes=params.num_classes+1, ignore_index=params.ignore_index)
    if metrics_name=='mse':
        return MSE()        
    if metrics_name=='rmse':
        return RMSE()
    if metrics_name=='rmse_log':
        return RMSELog()
    if metrics_name=='mae':
        return MAE(threshold=params.threshold, min_depth=params.min_depth)
    if metrics_name=='abs_rel':
        return MAE(relative=True, max_depth=params.threshold, min_depth=params.min_depth)
    if metrics_name=='sq_rel':
        return MSE(relative=True, max_depth=params.threshold, min_depth=params.min_depth)
    if metrics_name=='delta1':
        return Threshold(threshold=1.25, max_depth=params.threshold, min_depth=params.min_depth)
    if metrics_name=='delta2':
        return Threshold(threshold=1.25**2, max_depth=params.threshold, min_depth=params.min_depth)
    if metrics_name=='delta1':
        return Threshold(threshold=1.25**3, max_depth=params.threshold, min_depth=params.min_depth)                    
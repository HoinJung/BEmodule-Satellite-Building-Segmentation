"""PyTorch Callbacks."""

import os
import numpy as np

import torch
import time
 
now = time.localtime()
 
class TorchEarlyStopping(object):
    """Tracks if model training should stop based on rate of improvement.

    Arguments
    ---------
    patience : int, optional
        The number of epochs to wait before stopping the model if the metric
        didn't improve. Defaults to 5.
    threshold : float, optional
        The minimum metric improvement required to count as "improvement".
        Defaults to ``0.0`` (any improvement satisfies the requirement).
    verbose : bool, optional
        Verbose text output. Defaults to off (``False``). _NOTE_ : This
        currently does nothing.
    """

    def __init__(self, patience=5, threshold=0.0, verbose=False):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, metric_score):

        if self.best is None:
            self.best = metric_score
            self.counter = 0
        else:
            if self.best - self.threshold < metric_score:
                self.counter += 1
            else:
                self.best = metric_score
                self.counter = 0

        if self.counter >= self.patience:
            self.stop = True


class TorchTerminateOnNaN(object):
    """Sets a stop condition if the model loss achieves an NaN or inf value.

    Arguments
    ---------
    patience : int, optional
        The number of epochs that must display an NaN loss value before
        stopping. Defaults to ``1``.
    verbose : bool, optional
        Verbose text output. Defaults to off (``False``). _NOTE_ : This
        currently does nothing.
    """

    def __init__(self, patience=1, verbose=False):
        self.patience = patience
        self.counter = 0
        self.stop = False

    def __call__(self, loss):
        if np.isnan(loss) or np.isinf(loss):
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.counter = 0



class TorchModelCheckpoint(object):
    """Save the model at specific points using Keras checkpointing args.

    Arguments
    ---------
    filepath : str, optional
        Path to save the model file to. The end of the path (before the
        file extension) will have ``'_[epoch]'`` added to it to ID specific
        checkpoints.
    monitor : str, optional
        The loss value to monitor. Options are
        ``['loss', 'val_loss', 'periodic']`` or a metric from the keys in
        :const:`solaris.nets.metrics.metric_dict` . Defaults to ``'loss'`` . If
        ``'periodic'``, it saves every n epochs (see `period` below).
    verbose : bool, optional
        Verbose text output. Defaults to ``False`` .
    save_best_only : bool, optional
        Save only the model with the best value? Defaults to no (``False`` ).
    mode : str, optional
        One of ``['auto', 'min', 'max']``. Is a better value higher or lower?
        Defaults to ``'auto'`` in which case it tries to infer it (if
        ``monitor='loss'`` or ``monitor='val_loss'`` , it assumes ``'min'`` ,
        if it's a metric it assumes ``'max'`` .) If ``'min'``, it assumes lower
        values are better; if ``'max'`` , it assumes higher values are better.
    period : int, optional
        If using ``monitor='periodic'`` , this saves models every `period`
        epochs. Otherwise, it sets the minimum number of epochs between
        checkpoints.
    """

    def __init__(self, filepath='', path_aoi='',monitor='loss', verbose=False,
                 save_best_only=False, mode='auto', period=1,
                 weights_only=True):

        self.filepath = filepath
        self.monitor = monitor
        self.aoi = path_aoi
        if self.monitor not in ['loss', 'val_loss', 'periodic']:
            self.monitor = metric_dict[self.monitor]
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.period = period
        self.weights_only = weights_only
        self.mode = mode
        if self.mode == 'auto':
            if self.monitor in ['loss', 'val_loss']:
                self.mode = 'min'
            else:
                self.mode = 'max'

        self.epoch = 0
        self.last_epoch = 0
        self.last_saved_value = None

    def __call__(self, model, file_path, loss_value=None, y_true=None, y_pred=None):
        """Run a round of model checkpointing for an epoch.

        Arguments
        ---------
        model : model object
            The model to be saved during checkpoints. Must be a PyTorch model.
        loss_value : numeric, optional
            The numeric output of the loss function. Only required if using
            ``monitor='loss'`` or ``monitor='val_loss'`` .
        y_true : :class:`np.array` , optional
            The labels for the validation data. Only required if using
            a metric as the monitored value.
        y_pred : :class:`np.array` , optional
            The predicted values from the model. Only required if using
            a metric as the monitored value.
        """

        self.epoch += 1
        if self.monitor == 'periodic': # update based on period
            if self.last_epoch + self.period <= self.epoch:
                # self.last_saved_value = loss_value if loss_value else 0
                self.save(model, file_path,self.weights_only)
                self.last_epoch = self.epoch


        elif self.monitor in ['loss', 'val_loss']:
            if self.last_saved_value is None:
                self.last_saved_value = loss_value
                if self.last_epoch + self.period <= self.epoch:
                    self.save(model,file_path, self.weights_only)
                    self.last_epoch = self.epoch
            if self.last_epoch + self.period <= self.epoch:
                if self.check_is_best_value(loss_value):
                    self.last_saved_value = loss_value
                    self.save(model, file_path,self.weights_only)
                    self.last_epoch = self.epoch

        else:
            if self.last_saved_value is None:
                self.last_saved_value = self.monitor(y_true, y_pred)
                if self.last_epoch + self.period <= self.epoch:
                    self.save(model,file_path, self.weights_only)
                    self.last_epoch = self.epoch
            if self.last_epoch + self.period <= self.epoch:
                metric_value = self.monitor(y_true, y_pred)
                if self.check_is_best_value(metric_value):
                    self.last_saved_value = metric_value
                    self.save(model, file_path, self.weights_only)
                    self.last_epoch = self.epoch

    def check_is_best_value(self, value):
        """Check if `value` is better than the best stored value."""
        if self.mode == 'min' and self.last_saved_value > value:
            return True
        elif self.mode == 'max' and self.last_saved_value < value:
            return True
        else:
            return False

    def save(self, model, file_path, weights_only):
        """Save the model.

        Arguments
        ---------
        model : :class:`torch.nn.Module`
            A PyTorch model instance to save.
        weights_only : bool, optional
            Should the entire model be saved, or only its weights (also known
            as the state_dict)? Defaults to ``False`` (saves entire model). The
            entire model must be saved to resume training without re-defining
            the model architecture, optimizer, and loss function.
        """
#         print("saved time  : %04d/%02d/%02d %02d:%02d:%02d"% (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
        
#         print(self.aoi)
#         print("aoi")
        lossvalue=np.round(self.last_saved_value,3)
#         save_path = self.filepath + self.aoi + '_' + str(now.tm_mon) + str(now.tm_mday)+str(now.tm_hour)+str(now.tm_min)+ '/'
        save_path = file_path
        save_name = save_path + 'best'+ '_epoch{}_{}'.format(self.epoch, str(lossvalue))
        #save_name = save_path + 'best'+ '_epoch_{}_{}'.format( str(self.epoch).zfill(3), str(lossvalue))
        
#         save_name = self.filepath + self.aoi + '_' + str(now.tm_mon) + str(now.tm_mday)+ '/' + 'best'+ '_epoch{}_{}'.format(
#             self.epoch, np.round(self.last_saved_value, 1))
#         save_name = os.path.splitext(self.filepath)[0] + self.aoi + + '_epoch{}_{}'.format(
#             self.epoch, np.round(self.last_saved_value, 3))
        
        save_name = save_name + '.pth'
        print("saved path : ", save_path)
        print()
        print()
        if not os.path.exists(save_path) : 
            os.makedirs(save_path)
        else : 
            pass
#         os.makedirs(save_path)
        if isinstance(model, torch.nn.DataParallel):
            to_save = model.module
        else:
            to_save = model
        if weights_only:
#             os.makedirs(save_path)
#             torch.save(save_path, save_name)
            torch.save(to_save.state_dict(), save_name)

        else:
            torch.save(to_save, save_name)


torch_callback_dict = {
    "early_stopping": TorchEarlyStopping,
    "model_checkpoint": TorchModelCheckpoint,
    "terminate_on_nan": TorchTerminateOnNaN,
    }

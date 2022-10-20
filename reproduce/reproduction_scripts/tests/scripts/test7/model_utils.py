import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import _addindent
from torch.utils.data import Dataset
import numpy as np 
from collections import Counter
import pdb
import matplotlib.pyplot as plt
import os
import torchtuples as tt

# summarize Pytorch model
def summary(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr
        
"""
    Dataset for Benign vs. Aggressive Task
"""
class ClassificationDataset(Dataset):
    def __init__(self, features, labels):
        super(ClassificationDataset, self).__init__()
        self.features = features
        self.labels = labels
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx])

"""
    Dataset for Survival Analysis Task
"""
class SurvivalDataset(Dataset):
    def __init__(self, features, event_indicator, survival_times):
        super(SurvivalDataset, self).__init__()
        self.features = features
        self.event_indicator = event_indicator
        self.survival_times = survival_times
    def __len__(self):
        return self.event_indicator.shape[0]
    def __getitem__(self, idx):
        return (self.features[idx], self.event_indicator[idx], self.survival_times[idx])

"""
    Dataset for Survival Analysis Task
"""
class ClassificationSurvivalDataset(Dataset):
    def __init__(self, features, class_labels, event_indicator, survival_times):
        super(ClassificationSurvivalDataset, self).__init__()
        self.features = features
        self.labels = labels
        self.event_indicator = event_indicator
        self.survival_times = survival_times
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx], self.event_indicator[idx], self.survival_times[idx])

# changed and used from a MIT licensed repo on github
# reference: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

"""
Learning rate scheduler wrapper for pycox models
"""
class LRScheduler(tt.cb.Callback):
    '''Wrapper for pytorch.optim.lr_scheduler objects.
    Parameters:
        scheduler: A pytorch.optim.lr_scheduler object.
       get_score: Function that returns a score when called.
    '''
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self):
        score = self.get_last_score()
        self.scheduler.step(score)
        stop_signal = False
        return stop_signal
    
    def get_last_score(self):
        return self.model.val_metrics.scores['loss']['score'][-1]
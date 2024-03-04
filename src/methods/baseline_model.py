from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from torch import nn


class Baseline(nn.Module):
    """Abstract class for model."""

    def __init__(self, **kwargs):

        super(Baseline, self).__init__()

        self._optimizer = None

    # move to device
    def to(self, device):
        self.device = device
        return super(Baseline, self).to(device)



    def build_model(self, optimizer, lr, wd=1e-9, **kwargs):
        self._optimizer = optimizer(self.parameters(), lr=lr, weight_decay=wd)

    def num_params(self):
        num_params = sum([np.prod(p.shape) for p in self.parameters()])
        return num_params

    def num_trainable_params(self):
        num_params = sum([np.prod(p.shape) for p in self.parameters() if p.requires_grad] )
        return num_params

    def freeze_module(self, module_name):

        module = getattr(self, module_name)
        for param in module.parameters():
            param.requires_grad = False

    def update_learning_rate(self, loss):
        pass



    def encode(self, x):
        ''' Return representation given a sample '''
        raise NotImplementedError()

    def save_state(self, path):
        ''' Save model state, including criterion and optimiizer '''
        raise NotImplementedError()

    def load_state(self, path):
        ''' Load model state, including criterion and optimiizer '''
        raise NotImplementedError()

    def sample(self, num=25):
        ''' Random sample of num images from distribution of the representation '''
        raise NotImplementedError()

    def decode(self, code):
        ''' Random sample of num images from distribution of the representation '''
        raise NotImplementedError()

    @property
    def get_criterion(self):
        return self._criterion

    @property
    def get_optimizer(self):
        return self._optimizer

    def zero_grad(self, set_to_none=True):
        return self._optimizer.zero_grad(set_to_none=True)

    def compute_loss(self, y):
        '''Compute outputs and loss'''

        y_hat = self.forward(y)
        return self._criterion(y, y_hat), y_hat

    def backward(self, loss):

        # credit assignment
        loss.backward()

    def update_weights(self):
        '''Compute gradients and update weights'''

        # update model weights
        self._optimizer.step()


    def forward(self, x):
        return super(Baseline, self).forward(x)


    def init_weights(self, m):

        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


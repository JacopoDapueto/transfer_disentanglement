


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.optim.lr_scheduler import MultiStepLR, ConstantLR, ReduceLROnPlateau


def constant_scheduler(optimizer):
    return ConstantLR(optimizer,  total_iters=0)


def multistep_scheduler(optimizer):
    return MultiStepLR(optimizer, milestones=[10, 20, 30])


def reduce_on_plateau(optimizer):
    return ReduceLROnPlateau(optimizer, patience=5)


def get_named_scheduler(name, optimizer):

    if name=="constant":
        return constant_scheduler(optimizer)

    if name=="multistep":
        return multistep_scheduler(optimizer)

    if name=="reduceonplateau":
        return reduce_on_plateau(optimizer)

    raise "Scheduler name not valid"

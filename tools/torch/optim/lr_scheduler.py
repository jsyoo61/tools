from torch.optim.lr_scheduler import _LRScheduler
class AdaptiveLR(_LRScheduler):
    """Modify the learning rate of each parameter group,
    depending on the current & previous loss values.

    If current_loss < previous_loss:
        lr += a*lr (Increment lr by a * lr)
    If current loss > previous loss:
        lr -= b*lr (Decrement lr by b * lr)

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        a: (float or list): A multiplicative factor to increment learning rate
            when the current loss is lower than the previous loss.
        b: (float or list): A multiplicative factor to decrement learning rate
            when the current loss is higher than the previous loss.

        The increment/decrement of learning rate is done on
        each lr for each group in optimizer.param_groups
    Example:
        >>> # Assuming optimizer has two groups.
        >>> a_list = [0.1, 0.2]
        >>> b_list = [0.5, 0.9]
        >>> scheduler = AdaptiveLR(optimizer, a=a_list, b=b_list)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step(loss)
    """
    requires_loss = True
    def __init__(self, optimizer, a, b, last_epoch = -1, threshold = -float('inf'), fix = False, last_loss = float('inf')):
        self.last_loss = last_loss
        self.loss = self.last_loss
        self.threshold = threshold
        self.fix = fix
        self.factor = 1
        self.a = a
        self.b = b
        # if (not isinstance(a, list) and not isinstance(a, tuple)) and (not isinstance(b, list) and not isinstance(b, tuple)):
        #     self.a_s = [a] * len(optimizer.param_groups)
        #     self.b_s = [b] * len(optimizer.param_groups)
        # elif (isinstance(a, list) or isinstance(a, tuple)) and (isinstance(b, list) or isinstance(b, tuple)):
        #     if len(a) != len(optimizer.param_groups) or len(b) != len(optimizer.param_groups):
        #         raise ValueError("Expected {} 'a's and 'b's, but got {}, {}".format(
        #             len(optimizer.param_groups), len(a), len(b)))
        #     self.a_s = list(a)
        #     self.b_s = list(b)
        # else:
        #     raise ValueError("a and b has to be the same type, of either iterable or numeric values."
        #                     "Got: a: {}, b: {}".format(type(a), type(b)))
        super(AdaptiveLR, self).__init__(optimizer, last_epoch)

    def step(self, loss = 0, epoch=None):
        if self.last_epoch >= 0:
            self.last_loss = self.loss
            self.loss = loss
        # self.last_epoch gets incremented afterwards
        super(AdaptiveLR, self).step(epoch=epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)
        if self.last_epoch > 0:
            if self.factor < self.threshold:
                # lr fixed after threshold
                if self.fix:
                    return [base_lr * self.threshold for base_lr in self.base_lrs]
                # lr not fixed after threshold
                elif self.loss <= self.last_loss:
                    self.factor += self.factor * self.a
                    return [base_lr * self.factor for base_lr in self.base_lrs]
                    # return [group['lr'] + group['lr'] * self.a for group in self.optimizer.param_groups]
                else:
                    return [base_lr * self.factor for base_lr in self.base_lrs]
                    # return [group['lr'] for group in self.optimizer.param_groups]
            else:
                if self.loss <= self.last_loss:
                    self.factor += self.factor * self.a
                    return [base_lr * self.factor for base_lr in self.base_lrs]
                    # return [group['lr'] + group['lr'] * self.a for group in self.optimizer.param_groups]
                else:
                    self.factor -= self.factor * self.b
                    return [base_lr * self.factor for base_lr in self.base_lrs]
                    # return [group['lr'] - group['lr'] * self.b for group in self.optimizer.param_groups]
        else:
            return self.base_lrs

class LinearLR(_LRScheduler):
    """Decay the learning rate of each parameter group by delta every epoch.
    Every step,
    lr = lr - initial_lr * delta
    -----
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        delta: (float or list): A subtractive factor to decrement learning rate.
        last_epoch (int): The index of last epoch. Default: -1.
    Example:
        >>> # Assuming optimizer has two groups.
        >>> delta_list = [0.1, 0.2]
        >>> scheduler = AdaptiveLR(optimizer, delta=delta_list)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(self, optimizer, delta, last_epoch = -1):
        self.deltas = delta
        if not isinstance(delta, list) and not isinstance(delta, tuple):
            self.deltas = [delta] * len(optimizer.param_groups)
        else:
            if len(delta) != len(optimizer.param_groups):
                raise ValueError("Expected {} deltas, but got {}".format(
                    len(optimizer.param_groups), len(delta)))
            self.deltas = list(delta)
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)
        if self.last_epoch > 0:
            return [group['lr'] - base_lr * delta for group, base_lr, delta in zip(self.optimizer.param_groups, self.base_lrs, self.deltas)]
        else:
            return self.base_lrs

    def _get_closed_form_lr(self):
        return [base_lr*(1 - delta * self.epoch) for base_lr, delta in zip(self.base_lrs, self.deltas)]


class NoneLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch = -1):
        super(NoneLR, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        return self.base_lrs
#
# import torch
# import torch.optim as optim
# w = torch.randn((1,5), requires_grad=True)
# y = torch.randn((2,3), requires_grad=True)
# optimizer = optim.Adam([w,y])
# optimizer
# lr_schduler = AdaptiveLR(optimizer, a=[0.1,0.2], b=0.5)
# help(AdaptiveLR1)
# lr_schduler.state_dict()
#
# for
# optimizer.step()
# lr_schduler.step()
# lr_schduler.state_dict()
#
# class AdaptiveLR():
#     def __init__(self, optimizer, a, b, last_epoch = 0):
#         self.optimizer = optimizer
#         self.a = a
#         self.b = b
#         self.last_loss = float('inf')
#         self.base_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
#         self.last_epoch = last_epoch
#         self._last_lr = self.base_lrs
#
#     def state_dict(self):
#         content = dict(
#         optimizer = self.optimizer,
#         a = self.a,
#         b = self.b,
#         last_loss = self.last_loss,
#         base_lrs = self.base_lrs,
#         last_epoch = self.last_epoch,
#         _last_lr = self._last_lr,
#         )
#         return content
#
#     def step(self, loss):
#         for i, param_group in enumerate(self.optimizer.param_groups):
#             lr = param_group['lr']
#             if loss <= self.last_loss:
#                 self._last_lr[i] += lr * self.a
#             else:
#                 self._last_lr[i] -= lr * self.b
#             self.optimizer.param_groups[i]['lr'] = self._last_lr[i]
#         self.last_loss = loss
#         self.last_epoch += 1
#
#
#     def get_last_lr(self):
#         return self._last_lr

def lr_schedule_VAE(epoch):
    if epoch <= 100:
        lr_target = 1e-4
        lr_decay = (lr_target) ** (1/100 * epoch)
        return lr_decay
    else:
        return 1e-3

"""Functions for ramping hyperparameters up or down
Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""

import numpy as np


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        lr = 1.0
    else:
        lr = current / rampup_length
    return lr


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

  
    lr_list = []

    def adjust_learning_rate(epoch, lr_sch, step_in_epoch, total_steps_in_epoch):

        lr = 0.4
        lr_rampup = 5.0
        initial_lr = 0.1

        epoch = epoch + step_in_epoch / total_steps_in_epoch
        lr = linear_rampup(epoch, lr_rampup) * (lr - initial_lr) + initial_lr
        lr = lr * (0.1 ** (epoch // lr_sch))

        return lr


    for epoch in range(20):
        for step_in_epoch in range(5):
            lr_temp = adjust_learning_rate(epoch, 6, step_in_epoch, 5)
            lr_list.append(lr_temp)

    plt.plot(np.asarray(lr_list))
    plt.show()
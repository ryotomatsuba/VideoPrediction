import numpy as np
import torch


# hyparameters
reverse_scheduled_sampling = 0
r_sampling_step_1 = 25000
r_sampling_step_2 = 50000
r_exp_alpha = 5000
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002


def schedule_sampling(eta, itr, batch, input_num,total_length = 10):
    """teacher forcing rate schedule
    Args:
        eta: teacher forcing rate
        itr: current iteration
        batch: batch size
        input_num: number of input
        total_length: total number of image sequence
    Returns:
        mask : 1 if teacher forcing is used
    """

    total_length = 10
    mask_shape=(batch, total_length-input_num-1)
    zeros = np.zeros(mask_shape)
    if not scheduled_sampling:
        return 0.0, zeros

    if itr < sampling_stop_iter:
        eta -= sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(mask_shape)
    true_token = (random_flip < eta)
    true_token = torch.FloatTensor(true_token).view(batch,-1,1,1,1)
    return eta, true_token



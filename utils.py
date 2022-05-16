import numpy as np
import math
import Configs

def reshape_patch(img_tensor, patch_size):

    assert 5 == img_tensor.ndim

    batch_size = np.shape(img_tensor)[0]
    seq_length = np.shape(img_tensor)[1]
    n_ch = np.shape(img_tensor)[2]
    img_height = np.shape(img_tensor)[3]
    img_width = np.shape(img_tensor)[4]

    a = np.reshape(img_tensor, [batch_size, seq_length, n_ch,
                                patch_size, img_height // patch_size,
                                patch_size, img_width // patch_size])
    b = np.transpose(a, [0, 1, 2, 3, 5, 4, 6])
    patch_tensor = np.reshape(b, [batch_size, seq_length, n_ch * patch_size * patch_size,
                                  img_height // patch_size,
                                  img_width // patch_size])
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):

    assert 5 == patch_tensor.ndim

    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    n_ch = np.shape(patch_tensor)[2]
    patch_height = np.shape(patch_tensor)[3]
    patch_width = np.shape(patch_tensor)[4]

    image_ch = n_ch // (patch_size * patch_size)

    a = np.reshape(patch_tensor, [batch_size, seq_length, image_ch,
                                  patch_size, patch_size, patch_height, patch_width])
    b = np.transpose(a, [0, 1, 2, 3, 5, 4, 6])
    img_tensor = np.reshape(b, [batch_size, seq_length, image_ch,
                                patch_height * patch_size,
                                patch_width * patch_size])

    return img_tensor

def reserve_scheduled_sampling_exp(itr, configs):
    if itr < configs.r_sampling_step_1:
        r_eta = 0.5
    elif itr < configs.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - configs.r_sampling_step_1) / configs.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < configs.r_sampling_step_1:
        eta = 0.5
    elif itr < configs.r_sampling_step_2:
        eta = 0.5 - (0.5 / (configs.r_sampling_step_2 - configs.r_sampling_step_2)) * (itr - configs.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample((configs.batch_size, configs.input_length-1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample((configs.batch_size, configs.total_length - configs.input_length - 1))
    true_token = (random_flip < eta)

    # images are the same in height and width
    ones = np.ones((configs.patch_size * configs.patch_size * configs.img_ch,
                    configs.img_width // configs.patch_size,
                    configs.img_width // configs.patch_size))

    zeros = np.zeros((configs.patch_size * configs.patch_size * configs.img_ch,
                    configs.img_width // configs.patch_size,
                    configs.img_width // configs.patch_size))

    real_input_flag = []
    for i in range(configs.batch_size):
        for j in range(configs.total_length - 2):
            if j < configs.input_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (configs.input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (configs.batch_size, configs.total_length - 2,
                                  configs.patch_size * configs.patch_size * configs.img_ch,
                                  configs.img_width // configs.patch_size,
                                  configs.img_width // configs.patch_size))

    return real_input_flag

def schedule_sampling(eta, itr, configs):
    zeros = np.zeros((configs.batch_size,
                      configs.total_length - configs.input_length - 1,
                      configs.patch_size ** 2 * configs.img_ch,
                      configs.img_width // configs.patch_size,
                      configs.img_width // configs.patch_size))
    if not configs.scheduled_sampling:
        return 0.0, zeros

    if itr < configs.sampling_stop_iter:
        eta -= configs.sampling_changing_rate
    else:
        eta = 0.0

    random_flip = np.random.random_sample(
        (configs.batch_size, configs.total_length - configs.input_length - 1)
    )
    true_token = (random_flip < eta)
    ones = np.ones((configs.patch_size ** 2 * configs.img_ch,
                    configs.img_width // configs.patch_size,
                    configs.img_width // configs.patch_size))
    zeros = np.zeros((configs.patch_size ** 2 * configs.img_ch,
                    configs.img_width // configs.patch_size,
                    configs.img_width // configs.patch_size))
    real_input_flag = []
    for i in range(configs.batch_size):
        for j in range(configs.total_length - configs.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (configs.batch_size,
                                  configs.total_length - configs.input_length - 1,
                                  configs.patch_size ** 2 * configs.img_ch,
                                  configs.img_width // configs.patch_size,
                                  configs.img_width // configs.patch_size))
    return eta, real_input_flag
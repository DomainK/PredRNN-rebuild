import numpy as np
import torch
import torch.nn as nn

class Configs:

    def __init__(self):

        # training/test
        self.is_training = 1
        self.device = torch.device('cuda')

        # data
        self.data_dir = "/content/drive/Shareddrives/MachineLearning/Io_Prediction/data/Moving-MNIST/mnist_test_seq.npy"
        self.model_dir = "/content/drive/Shareddrives/MachineLearning/Io_Prediction/predRNN/model/"
        self.input_length = 10
        self.total_length = 20
        self.img_width = 64
        self.img_ch = 1

        # model
        self.model_name = 'PredRNN_v2'
        self.n_hidden = [64, 64, 64, 64]
        self.kernel_size = 5
        self.stride = 1
        self.patch_size = 4
        self.layer_norm = 1
        self.decouple_beta = 0.1

        # reverse scheduled sampling
        # 1:25000 2:50000 al:5000
        self.reverse_scheduled_sampling = 0
        self.r_sampling_step_1 = 25
        self.r_sampling_step_2 = 50
        self.r_exp_alpha = 5

        # scheduled sampling
        # iter:50000 rate:0.00002
        self.scheduled_sampling = 1
        self.sampling_stop_iter = 50
        self.sampling_start_value = 1.0
        self.sampling_changing_rate = 0.02

        # optimization
        # max:80000 inter:5000
        self.lr = 0.001
        self.reverse_input = 1
        self.batch_size = 80
        self.max_iterations = 80
        self.display_interval = 100
        self.val_interval = 5
        self.n_save_samples = 10


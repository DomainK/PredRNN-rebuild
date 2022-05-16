import numpy as np
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error

def batch_frame_metrics(gt_seqs, gen_seqs):

    ssim = 0.0
    psnr = 0.0
    mse = 0.0
    batchsize = gt_seqs.shape[0]
    frames = gt_seqs.shape[1]

    for i in range(batchsize):
        for j in range(frames):
            ssim += structural_similarity(gt_seqs[i, j, 0, :, :], gen_seqs[i, j, 0, :, :],
                                          data_range=1.0, multichannel=False, channel_axis=None)
            psnr += peak_signal_noise_ratio(gt_seqs[i, j, 0, :, :], gen_seqs[i, j, 0, :, :], data_range=1.0)
            mse += mean_squared_error(gt_seqs[i, j, 0, :, :], gen_seqs[i, j, 0, :, :])
    return ssim / (batchsize * frames), psnr / (batchsize * frames), mse / (batchsize * frames)

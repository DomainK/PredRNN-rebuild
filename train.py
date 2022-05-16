import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Configs import Configs
from PredRNN import PredRNN
from MovingMNIST import MovingMnist
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import utils
import Metrics

def main():

    configs = Configs()
    model = PredRNN(4, configs.n_hidden, configs).to(configs.device)

    train_data = MovingMnist(configs.data_dir, True)
    train_dataloader = DataLoader(train_data, batch_size=configs.batch_size, shuffle=True)
    val_data = MovingMnist(configs.data_dir, False)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)

    optim = torch.optim.Adam(model.parameters(), lr=configs.lr)

    loss_recorder = []

    since = time.time()

    for i in range(1, configs.max_iterations + 1):
        model.train()

        print('-' * 10)
        print('Epoch {}/{}'.format(i, configs.max_iterations))
        print('-' * 10)

        loss_epoch = 0.0

        eta = configs.sampling_start_value
        if configs.reverse_scheduled_sampling == 1:
            real_input_flag = utils.reserve_scheduled_sampling_exp(i, configs)
        else:
            eta, real_input_flag = utils.schedule_sampling(eta, i, configs)

        for seq in train_dataloader:
            seq = utils.reshape_patch(seq, configs.patch_size)
            seq = torch.FloatTensor(seq).to(configs.device)
            mask = torch.FloatTensor(real_input_flag).to(configs.device)

            optim.zero_grad()

            _, loss = model(seq, mask)
            loss.backward()
            optim.step()

            loss_rev = 0
            if configs.reverse_input:
                seq_rev = torch.flip(seq, dims=[1])
                _, loss_rev = model(seq_rev, mask)
                loss_rev.backward()
                optim.step()

            loss_epoch += ((loss + loss_rev) / 2).cpu().detach().numpy()

        print("Train Loss: {:.4f}".format(loss_epoch))
        loss_recorder.append(loss_epoch)

        if i % 5000 == 0:
            model.eval()
            print("  " + '-' * 10)
            print('Validation stage')
            print("  " + '-' * 10)

            img_mse, ssim, img_psnr = 0.0, 0.0, 0.0
            ssim_max = 0.0
            psnr_max = 0.0

            if configs.reverse_scheduled_sampling == 1:
                mask_input = 1
            else:
                mask_input = configs.input_length

            real_input_flag = np.zeros(
                (configs.batch_size,
                 configs.total_length - mask_input - 1,
                 configs.patch_size ** 2 * configs.img_ch,
                 configs.img_width // configs.patch_size,
                 configs.img_width // configs.patch_size)
            )
            if configs.reverse_scheduled_sampling == 1:
                real_input_flag[:, :configs.input_length - 1, :, :, :] = 1.0

            with torch.no_grad():
                loss_val = 0.0
                is_show = True
                for seq_val in val_dataloader:
                    tensor_val = utils.reshape_patch(seq_val, configs.patch_size)
                    tensor_val = torch.FloatTensor(tensor_val).to(configs.device)
                    mask_val = torch.FloatTensor(real_input_flag).to(configs.device)
                    next_seq, loss = model(tensor_val, mask_val)

                    loss_val += loss.cpu().detach().numpy()

                    show_seq = utils.reshape_patch_back(next_seq.cpu().detach().numpy(), configs.patch_size)
                    ssim, img_psnr, img_mse = Metrics.batch_frame_metrics(seq_val, show_seq)

                    if is_show:
                        is_show = False
                        plt.figure()
                        for k in range(configs.input_length):
                            plt.subplot(2, configs.input_length, k + 1)
                            plt.imshow(seq_val[0][k], cmap='gray')
                            plt.subplot(1, configs.input_length, k + 1)
                            plt.imshow(show_seq[0][k], cmap='gray')
                        plt.show()

                        plt.figure()
                        for k in range(configs.total_length - configs.input_length):
                            plt.subplot(2, configs.total_length - configs.input_length, k + 1)
                            plt.imshow(seq_val[0][k + 10], cmap='gray')
                            plt.subplot(1, configs.total_length - configs.input_length, k + 1)
                            plt.imshow(show_seq[0][k + 10], cmap='gray')
                        plt.show()

                print("  Val Loss : {:.4f}".format(loss_val))
                print("  ssim : {:.4f}".format(ssim))
                print("  psnr : {:.4f}".format(img_psnr))
                print("  mse : {:.8f}".format(img_mse))

            if ssim >= ssim_max and img_psnr >= psnr_max:
                # save model and test
                # torch.save(model, model_dir + file_name + str(i))
                print('Model saved!')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    plt.plot(np.array(loss_recorder), 'r')
    plt.show()
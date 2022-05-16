import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Configs import Configs
import utils

class STLSTMCell(nn.Module):

    def __init__(self, ch_in, n_hidden, width, kernal_size, stride, layer_norm):
        super(STLSTMCell, self).__init__()

        self.n_hidden = n_hidden
        self.padding = kernal_size // 2
        self.forget_bias = 1.0

        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(ch_in, self.n_hidden * 7, kernel_size=kernal_size, stride=stride, padding=self.padding,
                          bias=False),
                nn.LayerNorm([self.n_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(self.n_hidden, self.n_hidden * 4, kernel_size=kernal_size, stride=stride, padding=self.padding,
                          bias=False),
                nn.LayerNorm([self.n_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(self.n_hidden, self.n_hidden * 3, kernel_size=kernal_size, stride=stride, padding=self.padding,
                          bias=False),
                nn.LayerNorm([self.n_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(self.n_hidden * 2, self.n_hidden, kernel_size=kernal_size, stride=stride, padding=self.padding,
                          bias=False),
                nn.LayerNorm([self.n_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(ch_in, self.n_hidden * 7, kernel_size=kernal_size, stride=stride, padding=self.padding,
                          bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(self.n_hidden, self.n_hidden * 4, kernel_size=kernal_size, stride=stride, padding=self.padding,
                          bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(self.n_hidden, self.n_hidden * 3, kernel_size=kernal_size, stride=stride, padding=self.padding,
                          bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(self.n_hidden * 2, self.n_hidden, kernel_size=kernal_size, stride=stride, padding=self.padding,
                          bias=False),
            )
        self.conv_last = nn.Conv2d(n_hidden * 2, self.n_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, h_prev, c_prev, m_prev):
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h_prev)
        m_concat = self.conv_m(m_prev)
        i_x, f_x, g_x, i_x_p, f_x_p, g_x_p, o_x = torch.split(x_concat, self.n_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.n_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.n_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_m)
        f_t = torch.sigmoid(f_x + f_m)
        g_t = torch.tanh(g_x + g_h)

        c_delta = i_t * g_t
        c_t = f_t * c_prev + c_delta

        i_t_p = torch.sigmoid(i_x_p + i_m)
        f_t_p = torch.sigmoid(f_x_p + f_m)
        g_t_p = torch.tanh(g_x_p + g_m)

        m_delta = i_t_p * g_t_p
        m_t = f_t_p * m_prev + m_delta

        cm = torch.cat([c_t, m_t], dim=1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(cm))
        h_t = o_t * torch.tanh(self.conv_last(cm))

        return h_t, c_t, m_t, c_delta, m_delta


class PredRNN(nn.Module):

    def __init__(self, n_layer, n_hidden, configs):
        super(PredRNN, self).__init__()

        self.configs = configs
        self.frame_ch = configs.patch_size * configs.patch_size * configs.img_ch
        self.n_layer = n_layer
        self.n_hidden = n_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(self.n_layer):
            ch_in = self.frame_ch if i == 0 else n_hidden[i-1]
            cell_list.append(
                STLSTMCell(ch_in, n_hidden[i], width, configs.kernel_size,
                           configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(n_hidden[n_layer-1], self.frame_ch,
                                   kernel_size=1, stride=1, padding=0, bias=False)

        adaptor_n_hidden = self.n_hidden[0]
        self.adaptor = nn.Conv2d(adaptor_n_hidden, adaptor_n_hidden, kernel_size=1, stride=1,
                                 padding=0, bias=False)

    def forward(self, frames, mask_true):

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        m_delta_list = []
        c_delta_list = []

        decouple_loss = []

        for i in range(self.n_layer):
            zeros = torch.zeros([batch, self.n_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            m_delta_list.append(zeros)
            c_delta_list.append(zeros)

        memory = torch.zeros([batch, self.n_hidden[0], height, width]).to(self.configs.device)

        for t in range(self.configs.total_length - 1):
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t-1] * frames[:, t] + (1-mask_true[:, t-1] * x_gen)
            else:
                if t < self.configs.input_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.input_length] * x_gen)

            h_t[0], c_t[0], memory, c_delta, m_delta = self.cell_list[0](net, h_t[0], c_t[0], memory)
            c_delta_list[0] = F.normalize(self.adaptor(c_delta).view(c_delta.shape[0], c_delta.shape[1],
                                                                     -1), dim=2)
            m_delta_list[0] = F.normalize(self.adaptor(m_delta).view(m_delta.shape[0], m_delta.shape[1],
                                                                     -1), dim=2)

            for i in range(1, self.n_layer):
                h_t[i], c_t[i], memory, c_delta, m_delta = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                c_delta_list[i] = F.normalize(self.adaptor(c_delta).view(c_delta.shape[0], c_delta.shape[1],
                                                                         -1), dim=2)
                m_delta_list[i] = F.normalize(self.adaptor(m_delta).view(m_delta.shape[0], m_delta.shape[1],
                                                                         -1), dim=2)

            x_gen = self.conv_last(h_t[self.n_layer - 1])
            next_frames.append(x_gen)

            for i in range(self.n_layer):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(c_delta_list[i], m_delta_list[i], dim=2)))
                )

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        next_frames = torch.stack(next_frames, dim=1)
        loss = self.MSE_criterion(next_frames, frames[:, 1:]) + self.configs.decouple_beta * decouple_loss
        return next_frames, loss

# def main():
#     configs = Configs()
#     test = torch.randn([8, 20, 1, 64, 64])
#     test_reshaped = utils.reshape_patch(test, 4).to(configs.device)
#     print(test_reshaped.shape)
#     model = PredRNN(4, [16, 16, 16, 16], configs).to(configs.device)
#     real_input_flag = utils.reserve_scheduled_sampling_exp(1, configs)
#     mask = torch.FloatTensor(real_input_flag).to(configs.device)
#     result, loss = model(test_reshaped, mask)
#     print(result.shape)
#     print(loss)
#     result_back = utils.reshape_patch_back(result.cpu().detach().numpy(), 4)
#     print(result_back.shape)
#
# if __name__ == '__main__':
#     main()

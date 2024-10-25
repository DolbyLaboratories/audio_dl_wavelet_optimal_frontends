"""
Copyright (c) 2024 Dolby Laboratories

Redistribution and use in source and binary forms, with or without modification, are permitted 
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions 
and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or 
promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
"""

import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch
from torch import nn

from wavefront.utilities.confusion_matrix import *
from wavefront.utilities.cumulative_freq_response import (
    get_cumulative_frequency_response,
)
from wavefront.models.mlp_model import MLP, act_fun
from wavefront.models.mlp_model import Model as Md


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :,
        getattr(
            torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda]
        )().long(),
        :,
    ]
    return x.view(xsize)


def sinc(band, t_right):
    y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = flip(y_right, 0)

    y = torch.cat([y_left, Variable(torch.ones(1)).cuda(), y_right])

    return y


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        min_low_hz=50,
        min_band_hz=50,
    ):
        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = (
                "SincConv only support one input channel (here, in_channels = {%i})"
                % (in_channels)
            )
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(
            self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
        )
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = (
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        )  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        out = F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )
        return out

    def get_filters(self):
        return self.filters


class sinc_conv(nn.Module):
    def __init__(self, N_filt, Filt_dim, fs):
        super(sinc_conv, self).__init__()

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = 2595 * np.log10(1 + (fs / 2) / 700)  # Convert Hz to Mel
        mel_points = np.linspace(
            low_freq_mel, high_freq_mel, N_filt
        )  # Equally spaced in Mel scale
        f_cos = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (fs / 2) - 100

        self.freq_scale = fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2 - b1) / self.freq_scale))

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

    def forward(self, x):
        filters = Variable(torch.zeros((self.N_filt, self.Filt_dim))).cuda()
        N = self.Filt_dim
        t_right = Variable(
            torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)) / self.fs
        ).cuda()

        min_freq = 50.0
        min_band = 50.0

        filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (
            torch.abs(self.filt_band) + min_band / self.freq_scale
        )

        n = torch.linspace(0, N, steps=N)

        # Filter window (hamming)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / N)
        window = Variable(window.float().cuda())

        for i in range(self.N_filt):
            low_pass1 = (
                2
                * filt_beg_freq[i].float()
                * sinc(filt_beg_freq[i].float() * self.freq_scale, t_right)
            )
            low_pass2 = (
                2
                * filt_end_freq[i].float()
                * sinc(filt_end_freq[i].float() * self.freq_scale, t_right)
            )
            band_pass = low_pass2 - low_pass1

            band_pass = band_pass / torch.max(band_pass)

            filters[i, :] = band_pass.cuda() * window

        out = F.conv1d(x, filters.view(self.N_filt, 1, self.Filt_dim))

        return out


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SincNet(nn.Module):
    def __init__(self, options, input_dim, fs):
        super(SincNet, self).__init__()

        self.cnn_N_filt = options.cnn_N_filt
        self.cnn_len_filt = options.cnn_len_filt
        self.cnn_max_pool_len = options.cnn_max_pool_len
        self.cnn_act = options.cnn_act
        self.cnn_drop = options.cnn_drop
        self.cnn_use_laynorm = options.cnn_use_laynorm
        self.cnn_use_batchnorm = options.cnn_use_batchnorm
        self.cnn_use_laynorm_inp = options.cnn_use_laynorm_inp
        self.cnn_use_batchnorm_inp = options.cnn_use_batchnorm_inp
        self.input_dim = int(input_dim)
        self.fs = fs
        self.N_cnn_lay = len(options.cnn_N_filt)
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.cnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        current_input = self.input_dim
        # n_filt = 0

        for i in range(self.N_cnn_lay):
            n_filt = int(self.cnn_N_filt[i])
            len_filt = int(self.cnn_len_filt[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))

            # layer norm initialization
            self.ln.append(
                LayerNorm(
                    [
                        n_filt,
                        int(
                            (current_input - self.cnn_len_filt[i] + 1)
                            / self.cnn_max_pool_len[i]
                        ),
                    ]
                )
            )

            self.bn.append(
                nn.BatchNorm1d(
                    n_filt,
                    # """int(
                    #    (current_input - self.cnn_len_filt[i] + 1)
                    #    / self.cnn_max_pool_len[i]
                    # ), huge mistake of authors to put this here
                    momentum=0.05,
                )
            )

            if i == 0:
                self.conv.append(
                    SincConv_fast(self.cnn_N_filt[0], self.cnn_len_filt[0], self.fs)
                )

            else:
                self.conv.append(
                    nn.Conv1d(
                        self.cnn_N_filt[i - 1], self.cnn_N_filt[i], self.cnn_len_filt[i]
                    )
                )

            current_input = int(
                (current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]
            )

        self.out_dim = current_input * n_filt

    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[1]

        if bool(self.cnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.cnn_use_batchnorm_inp):
            x = self.bn0((x))

        x = x.view(batch, 1, seq_len)

        for i in range(self.N_cnn_lay):
            if self.cnn_use_laynorm[i]:
                if i == 0:
                    x = self.drop[i](
                        self.act[i](
                            self.ln[i](
                                F.max_pool1d(
                                    torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i]
                                )
                            )
                        )
                    )
                else:
                    x = self.drop[i](
                        self.act[i](
                            self.ln[i](
                                F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])
                            )
                        )
                    )

            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](
                    self.act[i](
                        self.bn[i](
                            F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])
                        )
                    )
                )

            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                x = self.drop[i](
                    self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))
                )

        x = x.view(batch, -1)

        return x

    def get_filters(self):
        return self.conv[0].get_filters()


class Model(Md):
    def __init__(
        self,
        n_classes,
        opts,
        sample_rate=None,
        weights=None,
        classes_indices=None,
        classes_labels=None,
        contains_silence=False,
        loss_weights=None,
    ):
        super().__init__(
            n_classes,
            opts,
            sample_rate,
            weights,
            classes_indices,
            classes_labels,
            contains_silence,
            loss_weights,
        )
        # v51, 54 models
        sinc_net_arch = opts.net.SincNetArch
        dnn1_arch = opts.net.DNN1Arch
        dnn2_arch = opts.net.DNN2Arch

        sincnet = SincNet(sinc_net_arch, input_dim=self.wlen, fs=self.fs)
        dnn1 = MLP(dnn1_arch, input_dim=sincnet.out_dim)
        dnn2 = MLP(dnn2_arch, input_dim=dnn1_arch.lay[-1], n_classes=n_classes)

        self.layers = nn.Sequential(sincnet, dnn1, dnn2)
        # cross entropy loss
        self.cost = nn.CrossEntropyLoss(weight=loss_weights)

        self.automatic_optimization = False

    def forward(self, x):  # this is to make prediction (or test)
        return self.layers(x)

    def get_filters(self):
        return self.layers[0].get_filters()

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)

        opt_frontend, opt_backend = self.optimizers()
        # sch1, sch2 = self.lr_schedulers()

        opt_frontend.zero_grad()
        opt_backend.zero_grad()

        self.manual_backward(loss)

        opt_frontend.step()
        opt_backend.step()

        return loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        # Specific configuration for ReduceLROnPlateau scheduler that monitor val_loss !!!
        sch1, sch2 = self.lr_schedulers()

        sch1.step(self.trainer.callback_metrics["val_loss"])
        sch2.step(self.trainer.callback_metrics["val_loss"])

        filters = self.get_filters()
        h_acc = get_cumulative_frequency_response(filters.to("cpu"), self.fs)
        self.logger.experiment.add_figure(
            "Accumulated frequency response", h_acc, self.current_epoch
        )

    def configure_optimizers(self):
        optimizer_frontend = torch.optim.AdamW(
            self.layers[0].parameters(), lr=self.lr, weight_decay=0.01
        )
        optimizer_backend = torch.optim.AdamW(
            self.layers[1:].parameters(), lr=self.lr, weight_decay=0.01
        )
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_frontend, factor=0.5, patience=5
        )
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_backend, factor=0.5, patience=5
        )

        return (
            {
                "optimizer": optimizer_frontend,
                "lr_scheduler": {"scheduler": scheduler1, "monitor": "val_loss"},
            },
            {
                "optimizer": optimizer_backend,
                "lr_scheduler": {"scheduler": scheduler2, "monitor": "val_loss"},
            },
        )

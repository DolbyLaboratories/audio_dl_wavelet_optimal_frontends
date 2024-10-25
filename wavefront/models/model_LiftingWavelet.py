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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as tr

from wavefront.models.mlp_model import CNN, MLP, act_fun
from wavefront.models.mlp_model import Model as Md
from wavefront.utilities.confusion_matrix import *
from wavefront.utilities.utilities import get_wavelet_plots


class InterpolatingWaveletTransform(nn.Module):
    def __init__(
        self,
        learnable=True,
        order=2,
        window_size=25,
        window_stride=10,
        output_dim=80,
        dual=False,
        sr=44100,
    ):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.output_dim = output_dim
        self.dual = dual
        self.order = order
        self.sr = sr
        self.avg_diff = []
        self.avg_details = []
        self.deltas = tr.ComputeDeltas()
        if learnable:  # we found better results without a learnable wavelet
            print("Using learnable wavelet!")
            if self.order == 2:
                self.predict_weights = nn.Parameter(
                    torch.tensor([0.5, 0.5], dtype=torch.float32)
                )
                self.update_weights = nn.Parameter(
                    torch.tensor([0.25, 0.25], dtype=torch.float32)
                )
            elif self.order == 4:
                self.predict_weights = nn.Parameter(
                    torch.tensor(
                        [-1 / 16, 9 / 16, 9 / 16, -1 / 16], dtype=torch.float32
                    )
                )
                self.update_weights = nn.Parameter(
                    torch.tensor(
                        [-1 / 32, 9 / 32, 9 / 32, -1 / 32], dtype=torch.float32
                    )
                )
            elif self.order == 6:
                self.predict_weights = nn.Parameter(
                    torch.tensor(
                        [3 / 256, -25 / 256, 150 / 256, 150 / 256, -25 / 256, 3 / 256],
                        dtype=torch.float32,
                    )
                )
                self.update_weights = nn.Parameter(
                    torch.tensor(
                        [3 / 512, -25 / 512, 150 / 512, 150 / 512, -25 / 512, 3 / 512],
                        dtype=torch.float32,
                    )
                )
            elif self.order > 7 and self.order % 2 == 0:
                predict_weights = torch.tensor(
                    [3 / 256, -25 / 256, 150 / 256, 150 / 256, -25 / 256, 3 / 256],
                    dtype=torch.float32,
                )
                update_weights = torch.tensor(
                    [3 / 512, -25 / 512, 150 / 512, 150 / 512, -25 / 512, 3 / 512],
                    dtype=torch.float32,
                )
                current_len = predict_weights.shape[0]
                pad_len = int((self.order - current_len) // 2)

                pad_values = torch.randn(pad_len) * 0.01  # Adjust the scale as needed

                # Concatenate the random values to the original vectors
                self.predict_weights = nn.Parameter(
                    torch.cat((pad_values, predict_weights, pad_values))
                )
                self.update_weights = nn.Parameter(
                    torch.cat((pad_values, update_weights, pad_values))
                )
            else:
                raise ValueError(
                    "Unsupported wavelet order. Wavelet order must be an even value."
                )

        else:
            if self.order == 2:
                self.predict_weights = torch.tensor([0.5, 0.5], dtype=torch.float32)
                self.update_weights = torch.tensor([0.25, 0.25], dtype=torch.float32)
            elif self.order == 4:
                self.predict_weights = torch.tensor(
                    [-1 / 16, 9 / 16, 9 / 16, -1 / 16], dtype=torch.float32
                )
                self.update_weights = torch.tensor(
                    [-1 / 32, 9 / 32, 9 / 32, -1 / 32], dtype=torch.float32
                )
            elif self.order == 6:
                self.predict_weights = torch.tensor(
                    [3 / 256, -25 / 256, 150 / 256, 150 / 256, -25 / 256, 3 / 256],
                    dtype=torch.float32,
                )
                self.update_weights = torch.tensor(
                    [3 / 512, -25 / 512, 150 / 512, 150 / 512, -25 / 512, 3 / 512],
                    dtype=torch.float32,
                )
            elif self.order > 7 and self.order % 2 == 0:
                predict_weights = torch.tensor(
                    [3 / 256, -25 / 256, 150 / 256, 150 / 256, -25 / 256, 3 / 256],
                    dtype=torch.float32,
                )
                update_weights = torch.tensor(
                    [3 / 512, -25 / 512, 150 / 512, 150 / 512, -25 / 512, 3 / 512],
                    dtype=torch.float32,
                )
                current_len = predict_weights.shape[0]
                pad_len = int((self.order - current_len) // 2)

                pad_values = torch.randn(pad_len) * 0.01  # Adjust the scale as needed

                # Concatenate the random values to the original vectors
                self.predict_weights = torch.cat(
                    (pad_values, predict_weights, pad_values)
                )
                self.update_weights = torch.cat(
                    (pad_values, update_weights, pad_values)
                )

            else:
                raise ValueError(
                    "Unsupported wavelet order. Wavelet order must be an even value."
                )

    def split(self, data):
        """Split data into even and odd indexed elements."""
        even = data[:, :, ::2]
        odd = data[:, :, 1::2]
        if even.shape[2] != odd.shape[2]:
            odd = torch.cat((odd, data[:, :, -1].view(data.shape[0], -1, 1)), dim=2)
        return even, odd

    def predict(self, even):
        """Predict odd elements using even elements and return the details."""

        B, L, C = even.shape
        N = self.predict_weights.shape[0]
        # Calculate the padding size
        pad_size = N // 2

        # Reshape the input tensor to (B*L, 1, C) to apply conv1d on the C dimension
        input_padded_reshaped = even.reshape(B * L, 1, C)  # (B*L, 1, C)
        input_padded = F.pad(
            input_padded_reshaped, (pad_size - 1, pad_size), mode="reflect"
        )  # Padding on the C dimension
        # Reshape the filter to match the required shape for conv1d
        filter_reshaped = self.predict_weights.view(
            1, 1, N
        )  # (out_channels, in_channels, kernel_size)

        pred = F.conv1d(input_padded, filter_reshaped, padding=0).view(
            B, L, C
        )  # reshape to (B, L, C)

        return pred

    def update(self, details):
        """Update even elements using details."""

        B, L, C = details.shape
        N = self.update_weights.shape[0]
        # Calculate the padding size
        pad_size = N // 2

        # Reshape the input tensor to (B*L, 1, C) to apply conv1d on the C dimension
        input_padded_reshaped = details.reshape(B * L, 1, C)  # (B*L, 1, C)
        input_padded = F.pad(
            input_padded_reshaped, (pad_size, pad_size - 1), mode="reflect"
        )  # Padding on the C dimension
        # Reshape the filter to match the required shape for conv1d
        filter_reshaped = self.update_weights.view(
            1, 1, N
        )  # (out_channels, in_channels, kernel_size)
        # Apply the convolution
        update = F.conv1d(input_padded, filter_reshaped, padding=0).view(
            B, L, C
        )  # (B, L, C)

        return update

    def get_filters(self, sample_rate, n=1, device="cpu"):
        # Generate the filters

        if n is None:
            n_steps = self.n_steps
        else:
            n_steps = n
        impulse = torch.zeros((1, 2048, 1), device=device)
        impulse[0, 2048 // 2, 0] = 1
        details = [
            torch.zeros(
                (impulse.shape[0], impulse.shape[1] * (2**i), impulse.shape[2]),
                device=device,
            )
            for i in range(n_steps)
        ]
        details[0] = impulse
        # Apply the lifting scheme to the impulse signal
        # zero_array = [torch.zeros(el.shape, device=device) for el in details]
        # Interpolate to get the lowpass and highpass filters
        lowpass_filter = (
            self.forward(impulse, inverse=True, n_steps=n_steps).view(-1).cpu()
        )
        highpass_filter = (
            self.forward(
                torch.zeros(impulse.shape, device=device),
                details=details,
                inverse=True,
                n_steps=n_steps,
            )
            .view(-1)
            .cpu()
        )

        impulse_fig, fft_fig, fft_fig_log = get_wavelet_plots(
            lowpass_filter, highpass_filter, fs=sample_rate
        )
        return impulse_fig, fft_fig, fft_fig_log

    def get_avg_diff(self):
        diff = self.avg_diff
        avg_details = self.avg_details
        self.avg_diff = []
        self.avg_details = []
        return diff, avg_details

    def forward(
        self, x, details=None, inverse=False, return_details=False, n_steps=None
    ):
        self.predict_weights = self.predict_weights.to(x.device)
        self.update_weights = self.update_weights.to(x.device)
        if inverse:
            if n_steps is None:
                n_steps = self.n_steps

            x = x.permute(0, 2, 1)
            if self.dual:
                for i in range(n_steps):
                    if details is None:
                        det = torch.zeros(x.shape, device=x.device)
                    else:
                        det = details.pop(0).permute(0, 2, 1)
                    odd = det + self.predict(x)
                    even = x - self.update(odd)

                    # merge
                    out = torch.stack((even, odd), dim=3).contiguous()

                    x = out.view(
                        out.shape[0], out.shape[1], out.shape[2] * out.shape[3]
                    )
            else:
                for i in range(n_steps):
                    if details is None:
                        det = torch.zeros(x.shape, device=x.device)
                    else:
                        det = details.pop(0).permute(0, 2, 1)
                        # backward predict
                    even = x - self.update(det)
                    odd = det + self.predict(even)

                    # merge
                    out = torch.stack((even, odd), dim=3).contiguous()
                    # out = out.permute(0, 1, 3, 2).contiguous()
                    x = out.view(
                        out.shape[0], out.shape[1], out.shape[2] * out.shape[3]
                    )
            return x.permute(0, 2, 1)

        else:
            x = x.unfold(
                1, self.window_size, self.window_stride
            )  # shape (B, L_out, N_channs)
            avg = torch.mean(x)
            if n_steps is None:
                self.n_steps = int(
                    np.round(np.log2(self.window_size / self.output_dim))
                )

            details_list = []
            if self.dual:
                for i in range(int(self.n_steps)):
                    even, odd = self.split(x)
                    x = even + self.update(odd)
                    details = odd - self.predict(x)
                    """transform = T.Resample(self.sr, int(self.sr/2**(int(self.n_steps-i-1))), lowpass_filter_width=64,
                                rolloff=0.9475937167399596,
                                resampling_method="sinc_interp_kaiser",
                                beta=14.769656459379492)"""
                    # details = transform(details.cpu()).to(x.device)
                    details_list.append(details)

            else:
                for i in range(int(self.n_steps)):
                    even, odd = self.split(x)
                    details = odd - self.predict(even)
                    x = even + self.update(details)
                    """transform = T.Resample(self.sr, int(self.sr/2**(int(self.n_steps-i-1))), lowpass_filter_width=64,
                                rolloff=0.9475937167399596,
                                resampling_method="sinc_interp_kaiser",
                                beta=14.769656459379492)
                    details = transform(details.cpu()).to(x.device)"""
                    details_list.append(details)

            avg_coarse = torch.mean(x)
            self.avg_diff.append(torch.abs(avg - avg_coarse))
            self.avg_details.append(torch.abs(torch.mean(details)))

            # x in shape (B, L_out, N_channs)
            details_list.append(x)
            fo_deltas = self.deltas(x)
            so_deltas = self.deltas(fo_deltas)
            output = torch.cat([details, fo_deltas, so_deltas], dim=2).permute(
                0, 2, 1
            )  # Return in dims (B,Nout,Cout*5)

            if return_details:
                return output, details_list
            else:
                return output


class LiftingWavelet(nn.Module):
    def __init__(self, options, input_dim, fs):
        super(LiftingWavelet, self).__init__()

        self.N_filt = options.N_filt
        self.act = options.act
        self.drop = options.drop
        self.use_laynorm = options.use_laynorm
        self.use_batchnorm = options.use_batchnorm
        self.use_laynorm_inp = options.use_laynorm_inp
        self.use_batchnorm_inp = options.use_batchnorm_inp
        self.input_dim = int(input_dim)
        self.fs = fs
        self.wlen = int(np.round(options.wlen * fs / 1000))
        self.wshift = int(options.wshift * fs / 1000)
        self.wavelet_order = options.wavelet_order

        if self.use_laynorm_inp:
            self.ln0 = nn.LayerNorm(self.input_dim)

        if self.use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        current_input = self.input_dim
        current_output = (self.input_dim - self.wlen) // self.wshift + 1

        # dropout
        n_dims = int(np.round(np.log2(self.wlen / self.N_filt))) + 1
        n_filt = 3 * int(
            np.round(self.wlen / (2 ** (np.round(np.log2(self.wlen / self.N_filt)))))
        )
        self.drop = nn.Dropout(p=self.drop)

        # activation
        self.act = act_fun(self.act)

        # layer norm initialization (why over two last dimensions???)
        self.ln = nn.LayerNorm([n_filt, current_output])

        self.bn = nn.BatchNorm1d(
            current_output,
            momentum=0.05,
        )

        self.conv = InterpolatingWaveletTransform(
            learnable=options.learnable_wavelet,
            order=self.wavelet_order,
            window_size=self.wlen,
            window_stride=self.wshift,
            output_dim=self.N_filt,
            dual=options.dual,
        )

        self.out_dim = [n_filt, current_output]

    def forward(self, x):
        if bool(self.use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.use_batchnorm_inp):
            x = self.bn0((x))

        if self.use_laynorm:
            x = self.drop(self.act(self.ln(self.conv(x))))

        elif self.use_batchnorm:
            x = self.drop(self.act(self.bn(self.conv(x))))
        else:
            x = self.drop(self.act(self.conv(x)))

        return x

    def get_filters(self, device="cpu", n=1):
        return self.conv.get_filters(self.fs, device=device, n=n)

    def get_avg_diff(self):
        return self.conv.get_avg_diff()


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
    ):
        super().__init__(
            n_classes,
            opts,
            sample_rate,
            weights,
            classes_indices,
            classes_labels,
            contains_silence,
        )
        # v51, 54 models
        self.frontend_is_frozen = False
        self.freeze_after_n_epochs = 1500

        wavelet_net_arch = opts.net.WaveletNetArch
        cnn_arch = opts.net.CNNArch
        dnn1_arch = opts.net.DNN1Arch
        dnn2_arch = opts.net.DNN2Arch

        wavenet = LiftingWavelet(wavelet_net_arch, input_dim=self.wlen, fs=sample_rate)
        cnn = CNN(cnn_arch, input_dim=wavenet.out_dim)
        dnn1 = MLP(dnn1_arch, input_dim=cnn.out_dim)
        dnn2 = MLP(dnn2_arch, input_dim=dnn1_arch.lay[-1], n_classes=n_classes)

        self.layers = nn.Sequential(wavenet, cnn, dnn1, dnn2)
        # cross entropy loss
        w = torch.ones(n_classes, dtype=torch.float32)
        w[-1] = 0.242
        self.cost = nn.CrossEntropyLoss(
            weight=weights
        )  # put cross entropy loss, in SincNet they use NLLLoss
        # self.cost = nn.NLLLoss()

        self.automatic_optimization = (
            False  # Add separate optimizers for the frontend and rest of the layers
        )

    def forward(self, x):  # this is to make prediction (or test)
        return self.layers(x)

    def step(self, batch, batch_idx):
        loss, x, y, h = super().step(batch, batch_idx)

        avg_diff, avg_details = self.layers[0].get_avg_diff()
        if len(avg_diff) == 1 and len(avg_details) == 1:
            avg_diff = avg_diff[0]
            avg_details = avg_details[0]
        else:
            avg_diff = torch.cat(avg_diff)
            avg_details = torch.cat(avg_details)
        loss = loss + torch.mean(avg_diff) + torch.mean(avg_details)
        return loss, x, y, h

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)

        opt_frontend, opt_backend = self.optimizers()

        opt_frontend.zero_grad()
        opt_backend.zero_grad()

        self.manual_backward(loss)

        opt_frontend.step()
        opt_backend.step()

        return loss

    def get_filters(self, n=1):
        return self.layers[0].get_filters(device=self.device, n=n)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        """if not self.frontend_is_frozen:
            if self.freeze_after_n_epochs == self.current_epoch:
                for el in self.layers[0].conv.parameters(): el.requires_grad_(False)
                self.frontend_is_frozen = True"""

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        # Specific configuration for ReduceLROnPlateau scheduler that monitor val_loss !!!
        sch1, sch2 = self.lr_schedulers()

        sch1.step(self.trainer.callback_metrics["val_loss"])
        sch2.step(self.trainer.callback_metrics["val_loss"])

        filters, freq_responses, freq_responses_log = self.get_filters()

        self.logger.experiment.add_figure(
            "Impulse responses", filters, self.current_epoch
        )

        self.logger.experiment.add_figure(
            "Frequency responses", freq_responses, self.current_epoch
        )

        self.logger.experiment.add_figure(
            "Frequency responses log-log", freq_responses_log, self.current_epoch
        )

        (
            filters_all_epochs,
            freq_responses_all_epochs,
            freq_responses_all_epochs_log,
        ) = self.get_filters(n=None)

        self.logger.experiment.add_figure(
            "Impulse responses all epochs", filters_all_epochs, self.current_epoch
        )

        self.logger.experiment.add_figure(
            "Frequency responses all epochs",
            freq_responses_all_epochs,
            self.current_epoch,
        )

        self.logger.experiment.add_figure(
            "Frequency responses all epochs log-log",
            freq_responses_all_epochs_log,
            self.current_epoch,
        )

    def configure_optimizers(self):
        optimizer_frontend = torch.optim.AdamW(
            self.layers[0:2].parameters(), lr=self.lr * 5, weight_decay=0.01
        )
        optimizer_backend = torch.optim.AdamW(
            self.layers[2:].parameters(), lr=self.lr, weight_decay=0.01
        )
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_frontend, factor=0.5, patience=10
        )
        # torch.optim.lr_scheduler.CyclicLR(optimizer_frontend, base_lr=0.001, max_lr=0.01, step_size_up=200000, mode='triangular'))
        #
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_backend, factor=0.5, patience=10
        )
        # torch.optim.lr_scheduler.CyclicLR(optimizer_backend, base_lr=0.0005, max_lr=0.001, step_size_up=200000, mode='triangular')
        # optimizer = torch.optim.RMSprop(self.layers.parameters(), lr=(self.lr or self.learning_rate), alpha=0.9,
        #                                eps=1e-8)

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

        # optimizer_frontend, optimizer_backend  # , optimizer_dnn1, optimizer_dnn2

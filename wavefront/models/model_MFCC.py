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
from torch import nn
import torchaudio.transforms as tr

from wavefront.models.mlp_model import MLP, act_fun
from wavefront.models.mlp_model import Model as Md


class MFCCWithDeltas(nn.Module):
    def __init__(self, n_filts, sr, wlen, lhop, n_mels):
        super(MFCCWithDeltas, self).__init__()
        self.mfccs = tr.MFCC(
            sample_rate=sr,
            n_mfcc=n_filts // 3,
            melkwargs={
                "win_length": wlen,
                "hop_length": lhop,
                "n_mels": n_mels,
                "f_max": sr / 2.0,
            },
        )
        self.deltas = tr.ComputeDeltas()

    def forward(self, x):
        """
        Parameters
        ----------
        x : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, n_filts, n_samples_out)
            Batch of sinc filters activations.
        """
        x = self.mfccs(x)
        fo_deltas = self.deltas(x)
        so_deltas = self.deltas(fo_deltas)
        return torch.cat((x, fo_deltas, so_deltas), dim=1)


class MFCCNet(nn.Module):
    def __init__(self, options, input_dim, fs):
        super(MFCCNet, self).__init__()

        self.cnn_N_filt = options.cnn_N_filt
        self.cnn_len_filt = options.cnn_len_filt
        self.cnn_act = options.cnn_act
        self.cnn_drop = options.cnn_drop
        self.cnn_use_laynorm = options.cnn_use_laynorm
        self.cnn_use_batchnorm = options.cnn_use_batchnorm
        self.cnn_use_laynorm_inp = options.cnn_use_laynorm_inp
        self.cnn_use_batchnorm_inp = options.cnn_use_batchnorm_inp
        self.input_dim = int(input_dim)
        self.fs = fs
        wlen = options.mfcc_wlen
        wshift = options.mfcc_wshift
        self.wlen_mfcc = int(self.fs * wlen / 1000.0)
        self.wshift_mfcc = int(self.fs * wshift / 1000.0)
        self.N_cnn_lay = len(options.cnn_N_filt)
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.cnn_use_laynorm_inp:
            self.ln0 = nn.LayerNorm(self.input_dim)

        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        current_input = self.input_dim
        current_output = current_input // self.wshift_mfcc + 1

        for i in range(self.N_cnn_lay):
            n_filt = int(self.cnn_N_filt[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))

            # layer norm initialization
            self.ln.append(
                nn.LayerNorm(
                    [
                        n_filt,
                        current_output,
                    ]
                )
            )

            self.bn.append(
                nn.BatchNorm1d(
                    n_filt,
                    current_output,
                    momentum=0.05,
                )
            )

            if i == 0:
                self.conv.append(  ##append mfccs
                    MFCCWithDeltas(
                        self.cnn_N_filt[i],
                        self.fs,
                        self.wlen_mfcc,
                        self.wshift_mfcc,
                        options.n_mel_fbank,
                    )
                )
                current_input = current_output
                current_output = int(current_input - self.cnn_len_filt[i] + 1)
            else:
                self.conv.append(
                    nn.Conv1d(
                        self.cnn_N_filt[i - 1],
                        self.cnn_N_filt[i],
                        self.cnn_len_filt[i - 1],
                    )
                )
                current_input = int(current_input - self.cnn_len_filt[i - 1] + 1)

                if i != self.N_cnn_lay - 1:
                    current_output = int(current_input - self.cnn_len_filt[i] + 1)

        self.out_dim = current_input * n_filt

    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[1]

        if bool(self.cnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.cnn_use_batchnorm_inp):
            x = self.bn0((x))

        # x = x.view(batch, 1, seq_len)

        for i in range(self.N_cnn_lay):
            if self.cnn_use_laynorm[i]:
                x = self.drop[i](self.act[i](self.ln[i](self.conv[i](x))))

            elif self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](self.conv[i](x))))

            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](self.conv[i](x)))

        x = x.view(batch, -1)

        return x

    def get_filters(self):
        return 0


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
        mfcc_net_arch = opts.net.MFCCNetArch
        dnn1_arch = opts.net.DNN1Arch
        dnn2_arch = opts.net.DNN2Arch

        mfccnet = MFCCNet(mfcc_net_arch, input_dim=self.wlen, fs=self.fs)
        dnn1 = MLP(dnn1_arch, input_dim=mfccnet.out_dim)
        dnn2 = MLP(dnn2_arch, input_dim=dnn1_arch.lay[-1], n_classes=n_classes)

        self.layers = nn.Sequential(mfccnet, dnn1, dnn2)
        # cross entropy loss
        self.cost = nn.CrossEntropyLoss(weight=loss_weights)

    def forward(self, x):  # this is to make prediction (or test)
        return self.layers(x)

    def get_filters(self):
        return self.layers[0].get_filters()

    def configure_optimizers(self):
        optimizer_frontend = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_frontend, factor=0.5, patience=10
        )

        return {
            "optimizer": optimizer_frontend,
            "lr_scheduler": {"scheduler": scheduler1, "monitor": "val_loss"},
        }

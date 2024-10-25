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

import os
import torch
import numpy as np
import torchaudio
from typing import TypedDict
import torchaudio.transforms as T

from wavefront.utilities.utilities import audio2frames
from pedalboard import Compressor, Pedalboard


class DataSet(torch.utils.data.Dataset):
    def __init__(self, modelconf, device, is_train: bool = True):
        """
        :param is_train: bool parameter, selects a portion for train and _another_ portion for test/validation
        """
        self.is_train = is_train
        self.is_augment = True if is_train else False
        self.is_multilabel = False  # single SoftMax (categoricalXentropy), multi sigmoid with binaryXentropy
        self.device = device
        self.sample_rate = modelconf.windowing.fs
        self.n_channels = None
        self.N_batches = modelconf.optimization.N_batches
        self.batch_size = modelconf.optimization.batch_size
        self.contains_silence = False

        self.rs = modelconf.windowing.resample
        if self.rs != self.sample_rate:
            self.resampler = T.Resample(
                self.sample_rate,
                self.rs,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
        self.spectrogram = T.Spectrogram(power=None)
        self.inv_spec = T.InverseSpectrogram()
        self.stretch = T.TimeStretch()
        self.compressor = Compressor(threshold_db=-50, ratio=25)

        return

    def __len__(self):
        return (
            self.N_batches * self.batch_size if self.is_train else len(self.features_ld)
        )

    def __getitem__(self, idx):
        if self.is_train:
            real_idx = idx % len(self.features_ld)
        else:
            real_idx = idx

        tdata = self.features_ld[real_idx]["data"]
        tclass = self.features_ld[real_idx]["class"]

        if self.is_augment:
            # do data augmentation (fast algorithms ideas)
            # 1 # if audio is of variable length, you get a window in a random place, ELSE from the beginning
            # 2 # random gain ELSE normalize (normalize inside the model?)
            # 3 # just swap phase (*-1)
            # 4 # julius (library) random eq or lowpass or bandpass ... etc
            # 5 # maybe some IR convolution?
            # 6 # noise...

            p_augment = 0.8

            if torch.rand(1) < 0.2:  # Time stretching
                min_stretch = 0.5
                max_stretch = 1.0
                rand_stretch = min_stretch + (max_stretch - min_stretch) * torch.rand(1)
                spec = self.spectrogram(tdata)
                stretch_spec = self.stretch(spec, rand_stretch.item())
                tdata = self.inv_spec(stretch_spec)

            snt_len = len(tdata)
            wlen = self.n_features
            if snt_len <= wlen + 1:
                print(snt_len)
            snt_beg = np.random.randint(snt_len - wlen - 1)
            tdata = tdata[snt_beg : snt_beg + wlen]

            if torch.rand(1) < p_augment:  # Random gain
                min_amp = 0.5
                max_amp = 1.5
                rand_amp = min_amp + (max_amp - min_amp) * torch.rand(1)
                tdata = tdata * rand_amp

            if torch.rand(1) < p_augment:  # Phase swap
                tdata = (-1) * tdata

            signal_rms = 10 * np.log10(np.mean(tdata.numpy() ** 2) + 1e-10)
            if torch.rand(1) < p_augment:  # add another class' signal as noise
                target_snr_dB = np.random.uniform(-50, -12)  # for example

                noise_id = np.random.randint(0, len(self.features_ld))
                while self.features_ld[noise_id]["class"] == tclass:
                    noise_id = np.random.randint(0, len(self.features_ld))
                noise = self.features_ld[noise_id]["data"]
                snt_beg = np.random.randint(len(noise) - wlen - 1)
                noise = noise[snt_beg : snt_beg + wlen]

                # rescale based on RMS
                noise_rms = 10 * np.log10(np.mean(noise.numpy() ** 2) + 1e-7)
                boost_noise_by_dBs = (
                    signal_rms - noise_rms
                )  # this would bring the noise to the same rms than the signal
                boost_noise_by_linear = 10 ** (
                    (boost_noise_by_dBs + target_snr_dB) / 20
                )  # this is the gain
                noise_out = noise * boost_noise_by_linear
                tdata = tdata + noise_out

            if torch.rand(1) < p_augment:  # compress using pedalboard
                # board = Pedalboard([self.compressor])
                thr_dB = int(signal_rms + np.random.randint(0, 12))
                thr_dB = np.min((0, thr_dB))
                ratio = np.random.randint(1, 6)
                board = Pedalboard([Compressor(threshold_db=thr_dB, ratio=ratio)])

                tdata = torch.Tensor(board(tdata.numpy(), sample_rate=self.sample_rate))

        else:
            frames = audio2frames(tdata, self.n_features, self.wshift, last="skip")
            tclass = torch.ones(frames.shape[0], dtype=torch.int) * tclass
            tdata = frames

        return tdata, tclass

    def load_audio_in_memory(self, audio_list):
        audio_list_dictionary = []
        samplerate = None
        for file in audio_list:
            file_path = os.path.join(self.esc_path_audio, file)
            audio, samplerate = torchaudio.load(file_path)
            audio = audio.view(-1)  # assuming audio is mono

            if samplerate != self.rs:
                audio = self.resampler(audio)

            dictionary = {
                "name": file,
                "data": audio,
                "class": self.get_class_from_audio_name(file),
            }
            audio_list_dictionary.append(dictionary)

        self.sample_rate = self.rs
        self.n_channels = 1 if (audio.ndim == 1) else audio.shape[0]
        return audio_list_dictionary

    def check_silence_frame(self, audio_frame):
        eps = 1e-20
        energy = torch.sum(audio_frame**2) / audio_frame.numel()
        en_db = 10 * torch.log10(energy + eps)
        if en_db < -80:
            return True
        return False


class Metadata_Record_Types(TypedDict):
    """Define the fields and their types in a record.
    Field names must match column names in CSV file header.
    """

    filename: str
    fold: int
    target: int
    category: str
    esc10: bool
    src_file: str
    take: str

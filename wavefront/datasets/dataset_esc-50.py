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
import pandas as pd

from wavefront.utilities.utilities import get_datasets_config
import wavefront.datasets.dataset


class DataSet(wavefront.datasets.dataset.DataSet):
    def __init__(self, modelconf, device, is_train: bool = True):
        """
        :param is_train: bool parameter, selects a portion for train and _another_ portion for test/validation
        """
        super().__init__(modelconf, device, is_train)
        config = get_datasets_config()
        cw_len = modelconf.windowing.cw_len
        cw_shift = modelconf.windowing.cw_shift
        self.esc_path_root = config.esc_50["path_root"]
        self.esc_path_audio = os.path.join(
            self.esc_path_root, config.esc_50["path_audio"]
        )
        self.esc_path_metadata = os.path.join(
            self.esc_path_root, config.esc_50["path_metadata"]
        )
        self.esc_file_matadata = os.path.join(
            self.esc_path_metadata, config.esc_50["filename_metadata"]
        )
        self.contains_silence = True

        self.metadata = self.get_audio_metadata()
        self.audio_list = self.get_audio_list()
        self.n_classes = self.metadata["target"].max() + 2

        w = torch.ones(self.n_classes, dtype=torch.float32)
        w[-1] = 0.242
        self.weights = w

        self.classes_indices = [str(idx) for idx in range(self.n_classes)]
        labels = self.metadata["target"].unique()
        labels.sort()
        self.classes_labels = [str(idx) for idx in labels]
        self.classes_labels.append(str(self.n_classes - 1))

        # split dataset in train and test
        # get one fold as "test" and the rest as "train"...
        self.audio_test_list = self.get_audio_of_specified_fold(2)
        self.audio_train_list = self.get_audio_of_specified_fold(2, is_fold=False)

        if is_train:
            audio_dict_list = self.load_audio_in_memory(self.audio_train_list)

        else:
            audio_dict_list = self.load_audio_in_memory(self.audio_test_list)

        self.n_features = int(self.sample_rate * cw_len / 1000.00)
        self.wshift = int(self.sample_rate * cw_shift / 1000.00)

        self.features_ld = audio_dict_list  # raw list of audio

        if not self.is_train:
            n_frames = int(
                np.floor(
                    (len(self.features_ld[0]["data"]) - self.n_features) / self.wshift
                )
            )
            self.frame_map = {}
            self.n_frames = (
                torch.ones(len(self.features_ld), dtype=torch.int) * n_frames
            )
            for i in range(len(self.features_ld)):
                for frame in range(n_frames):
                    self.frame_map[n_frames * i + frame] = i

        return

    def __getitem__(self, idx):
        tdata, tclass = super().__getitem__(idx)

        if self.is_augment:
            if self.check_silence_frame(tdata):
                tclass = self.n_classes - 1
        else:
            for i, frame in enumerate(tdata):
                if self.check_silence_frame(frame):
                    tclass[i] = self.n_classes - 1

        return tdata, tclass

    def get_audio_metadata(self):
        return pd.read_csv(self.esc_file_matadata)

    def get_audio_list(self):
        return sorted(os.listdir(self.esc_path_audio))

    def get_audio_of_specified_fold(self, fold_number: int, is_fold=True):
        if (fold_number > 5) or (fold_number < 1):
            raise ValueError("ESC contains 5 folds, with indices [1-5]")
        if is_fold:
            is_fold_mask = self.metadata["fold"] == fold_number
            return self.metadata[is_fold_mask]["filename"]
        else:
            is_notfold_mask = self.metadata["fold"] != fold_number
            return self.metadata[is_notfold_mask]["filename"]

    def get_class_from_audio_name(self, audio_filename):
        mask_audio_name = self.metadata["filename"] == audio_filename
        this_audio_target = self.metadata[mask_audio_name]["target"].tolist()[0]
        return this_audio_target

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

import wavefront.datasets.dataset
import os
import numpy as np
import itertools
import torchaudio

from wavefront.utilities.utilities import get_datasets_config, ReadList
from wavefront.utilities.utilities import audio2frames


class DataSet(wavefront.datasets.dataset.DataSet):
    def __init__(self, modelconf, device, is_train: bool = True):
        super().__init__(modelconf, device, is_train)
        self.name = "timit"
        config = get_datasets_config()
        self.esc_path_root = config.timit["path_root"]
        self.esc_path_audio_labels = config.timit["path_audio_labels"]
        self.esc_path_test_labels = config.timit["path_tests_labels"]
        self.esc_path_audio = self.esc_path_root
        self.esc_path_test = os.path.join(
            self.esc_path_root, config.timit["path_tests"]
        )
        self.esc_path_metadata = config.timit["path_metadata"]
        self.esc_file_matadata = os.path.join(
            self.esc_path_metadata, config.timit["filename_metadata"]
        )

        self.metadata = self.get_audio_metadata()
        self.n_classes = max(self.metadata.values()) + 1
        self.weights = None

        self.classes_indices = [str(idx) for idx in range(self.n_classes)]
        self.classes_labels = [str(idx) for idx in self.classes_indices]

        # split dataset in train and test
        # get one fold as "test" and the rest as "train"...
        self.audio_test_list = ReadList(self.esc_path_test_labels)
        self.audio_train_list = ReadList(
            self.esc_path_audio_labels
        )  # self.get_audio_of_specified_fold(1, is_fold=False)

        if is_train:
            audio_dict_list = self.load_audio_in_memory(self.audio_train_list)
        else:
            audio_dict_list = self.load_audio_in_memory(self.audio_test_list)

        # how many frames? how many features?

        cw_len = modelconf.windowing.cw_len
        cw_shift = modelconf.windowing.cw_shift
        self.n_features = int(self.sample_rate * cw_len / 1000.00)
        self.wshift = int(self.sample_rate * cw_shift / 1000.00)

        self.n_frames = 1
        self.features_ld = audio_dict_list  # raw list of audio

        if not self.is_train:
            self.n_frames = []
            self.frame_map = {}
            for i, el in enumerate(self.features_ld):
                n_frames = int(
                    np.floor((len(el["data"]) - self.n_features) / self.wshift) + 1
                )
                beg_frame = sum(self.n_frames)
                for j in range(n_frames):
                    self.frame_map[beg_frame + j] = i
                self.n_frames.append(n_frames)

        return

    def get_class_from_audio_name(self, audio_filename):
        return self.metadata[audio_filename]

    def get_audio_metadata(self):
        with open(self.esc_file_matadata, "rb") as fobj:
            lab_dict = np.load(fobj, allow_pickle=True).item()
        return lab_dict

    def get_audio_list(self):
        return sorted(ReadList(self.esc_path_audio_labels))

    def load_frames_in_memory(self, audio_list):
        audio_list_dictionary = []
        samplerate = None
        for file in audio_list:
            file_path = os.path.join(self.esc_path_audio, file)
            audio, samplerate = torchaudio.load(file_path)
            frames = audio2frames(
                audio.view(-1),
                lframe=self.n_features,
                lhop=self.wshift,
                pad=0,
                last="skip",
            )  # Assuming audio is mono
            thisframeclass = self.get_class_from_audio_name(file)
            this_audio_dict_list = [
                {"name": file, "order": i, "data": frame, "class": thisframeclass}
                for i, frame in enumerate(frames)
            ]

            audio_list_dictionary.append(this_audio_dict_list)

        audio_list_dictionary = list(
            itertools.chain.from_iterable(audio_list_dictionary)
        )
        self.sample_rate = samplerate
        self.n_channels = 1 if (audio.ndim == 1) else audio.shape[0]
        return audio_list_dictionary

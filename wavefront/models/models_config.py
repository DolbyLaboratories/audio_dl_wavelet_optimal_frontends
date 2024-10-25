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

from dataclasses import field
from wavefront.utilities.config import module_config


@module_config
class WaveletNetArchConfig:
    N_filt: int = 80  # FIXME: n_features
    use_laynorm_inp: bool = True
    use_batchnorm_inp: bool = False
    use_laynorm: bool = True
    use_batchnorm: bool = False
    act: str = "leaky_relu"
    drop: float = 0.0
    wlen: float = 29.0249  # FIXME: ADD MS
    wshift: int = 5
    wavelet_order: int = (
        16  # Order of interpolating wavelet. Supported orders are any even number
    )
    dual: bool = False
    learnable_wavelet: bool = True


@module_config
class MFCCNetArchConfig:
    cnn_N_filt: list = field(default_factory=list)
    cnn_len_filt: list = field(default_factory=list)
    cnn_use_laynorm_inp: bool = True
    cnn_use_batchnorm_inp: bool = False
    cnn_use_laynorm: list = field(default_factory=list)
    cnn_use_batchnorm: list = field(default_factory=list)
    cnn_act: list = field(default_factory=list)
    cnn_drop: list = field(default_factory=list)
    mfcc_wlen: int = 25
    mfcc_wshift: int = 10
    n_mel_fbank: int = 80

    def __post_init__(self):
        self.cnn_N_filt = [81, 32]
        self.cnn_len_filt = [5, 5]
        self.cnn_drop = [0.2, 0.2, 0.0]
        self.cnn_use_laynorm = [True, True, True]
        self.cnn_use_batchnorm = [False, False, False]
        self.cnn_act = ["leaky_relu", "leaky_relu", "leaky_relu"]


@module_config
class SincNetArchConfig:
    cnn_N_filt: list = field(default_factory=list)
    cnn_len_filt: list = field(default_factory=list)
    cnn_max_pool_len: list = field(default_factory=list)
    cnn_use_laynorm_inp: bool = True
    cnn_use_batchnorm_inp: bool = False
    cnn_use_laynorm: list = field(default_factory=list)
    cnn_use_batchnorm: list = field(default_factory=list)
    cnn_act: list = field(default_factory=list)
    cnn_drop: list = field(default_factory=list)

    def __post_init__(self):
        self.cnn_N_filt = [80, 32]
        self.cnn_len_filt = [251, 5, 5]
        self.cnn_max_pool_len = [3, 3, 3]
        self.cnn_drop = [0.2, 0.2, 0.0]
        self.cnn_use_laynorm = [True, True, True]
        self.cnn_use_batchnorm = [False, False, False]
        self.cnn_act = ["leaky_relu", "leaky_relu", "leaky_relu"]


@module_config
class CNNArchConfig:
    cnn_N_filt: list = field(default_factory=list)
    cnn_len_filt: list = field(default_factory=list)
    cnn_use_laynorm: list = field(default_factory=list)
    cnn_use_batchnorm: list = field(default_factory=list)
    cnn_act: list = field(default_factory=list)
    cnn_drop: list = field(default_factory=list)
    cnn_max_pool_len: list = field(default_factory=list)

    def __post_init__(self):
        self.cnn_N_filt = [32]
        self.cnn_len_filt = [3, 3]
        self.cnn_drop = [0.2, 0.0]
        self.cnn_use_laynorm = [False, False]
        self.cnn_use_batchnorm = [True, True]
        self.cnn_act = ["leaky_relu", "leaky_relu"]
        self.cnn_max_pool_len = [3, 3]


@module_config
class DNN1ArchConfig:
    lay: list = field(default_factory=list)
    drop: list = field(default_factory=list)
    use_laynorm_inp: bool = True
    use_batchnorm_inp: bool = False
    use_batchnorm: list = field(default_factory=list)
    use_laynorm: list = field(default_factory=list)
    act: list = field(default_factory=list)

    def __post_init__(self):
        self.lay = [512]
        self.drop = [0.2, 0.0, 0.0]
        self.use_batchnorm = [True, True, True]
        self.use_laynorm = [False, False, False]
        self.act = ["leaky_relu", "leaky_relu", "leaky_relu"]


@module_config
class DNN2ArchConfig:
    lay: list = field(default_factory=list)
    drop: list = field(default_factory=list)
    use_laynorm_inp: bool = True
    use_batchnorm_inp: bool = False
    use_batchnorm: list = field(default_factory=list)
    use_laynorm: list = field(default_factory=list)
    act: list = field(default_factory=list)

    def __post_init__(self):
        self.lay = [51]
        self.drop = [0.0]
        self.use_batchnorm = [False]
        self.use_laynorm = [False]
        self.act = ["linear"]


@module_config
class SincNetConfig:
    SincNetArch: SincNetArchConfig = field(default_factory=SincNetArchConfig)
    DNN1Arch: DNN1ArchConfig = field(default_factory=DNN1ArchConfig)
    DNN2Arch: DNN2ArchConfig = field(default_factory=DNN2ArchConfig)


@module_config
class MFCCConfig:
    MFCCNetArch: MFCCNetArchConfig = field(default_factory=MFCCNetArchConfig)
    DNN1Arch: DNN1ArchConfig = field(default_factory=DNN1ArchConfig)
    DNN2Arch: DNN2ArchConfig = field(default_factory=DNN2ArchConfig)


@module_config
class LiftingWaveletConfig:
    WaveletNetArch: WaveletNetArchConfig = field(default_factory=WaveletNetArchConfig)
    CNNArch: CNNArchConfig = field(default_factory=CNNArchConfig)
    DNN1Arch: DNN1ArchConfig = field(default_factory=DNN1ArchConfig)
    DNN2Arch: DNN2ArchConfig = field(default_factory=DNN2ArchConfig)


@module_config
class ASTConfig:
    CNNArch: CNNArchConfig = field(default_factory=CNNArchConfig)
    DNN1Arch: DNN1ArchConfig = field(default_factory=DNN1ArchConfig)
    DNN2Arch: DNN2ArchConfig = field(default_factory=DNN2ArchConfig)

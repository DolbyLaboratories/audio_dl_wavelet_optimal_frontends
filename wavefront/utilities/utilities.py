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
import yaml
from munch import Munch
import torch
import matplotlib.pyplot as plt
import numpy as np


def cast_types_in_dictionary(dict_, typed_dict) -> dict:
    """Convert values in given dictionary to corresponding types in TypedDict ."""
    # from: https://stackoverflow.com/questions/11665628/read-data-from-csv-file-and-transform-from-string-to-correct-data-type-includin/54794212#54794212
    fields = typed_dict.__annotations__
    return {
        name: [fields[name](val) for val in values] for name, values in dict_.items()
    }


def get_datasets_config():
    datasets_config_file = os.path.join("wavefront", "config_datasets.yaml")
    config = yaml.safe_load(open(datasets_config_file))
    config = Munch(config)  # if you find a better way to simplify this, go on
    return config


def audio_raw_to_stft(audio_raw, sample_rate, n_channels):
    frame_size = 2048
    overlap = 0

    return torch.stft(audio_raw, frame_size, return_complex=True)


def ReadList(list_file):
    f = open(list_file, "r")
    lines = f.readlines()
    list_sig = []
    for x in lines:
        list_sig.append(x.rstrip())
    f.close()
    return list_sig


def audio2frames(
    x, lframe, lhop, pad=0, last="zeropad"
):  # Assumes input as a single-dim tensor
    if pad > 0:
        z = torch.zeros(pad, device=x.device)
        x = torch.cat([z, x, z], 0)
    if last == "zeropad":
        aux = torch.cat([x, torch.zeros(lframe, device=x.device)], 0)
    elif last == "skip":
        aux = x.clone()
    else:
        raise NotImplementedError
    # t = torch.arange(lframe, device=x.device).view(1, -1) + torch.arange(0, len(aux) - lframe, lhop,
    #                                                                      device=x.device).view(-1, 1)
    t1 = torch.arange(0, lframe, device=x.device).view(1, -1)

    # Create the second range tensor ensuring the step size and range are valid
    end_value = len(aux) - lframe
    if lhop > 0 and end_value > 0:
        t2 = torch.arange(0, end_value, lhop, device=x.device).view(-1, 1)
    else:
        print(
            f"Check the values of lhop: {lhop} and len(aux): {len(aux)} - lframe: {lframe}"
        )
        raise ValueError(
            f"Check the values of lhop: {lhop} and len(aux): {len(aux)} - lframe: {lframe}"
        )
    t = t1 + t2

    return aux[t]


def frames2audio(x, lframe, lhop):
    xx = x.clone()
    xout = torch.zeros((len(x) - 1) * lhop + lframe, device=xx.device)
    overlap = lframe - lhop
    xout[:lframe] = xx[0]
    for i in range(1, len(x)):
        xout[overlap + i * lhop : lframe + i * lhop] = xx[i, -lhop:]
    xout = xout / torch.max(torch.abs(xout))
    return xout


def check_percentage_silence(x, lframe, lhop, fname):
    with open(fname, "w") as f:
        sr = []
        per_class_sil = {}
        pr = []
        for el in x:
            audio = el["data"]
            name = el["name"]
            target = name.split("-")[-1].split(".")[0]
            frames = audio2frames(audio, lframe, lhop, last="skip")
            n_sil_frames = 0
            for frame in frames:
                if check_silence_frame(frame):
                    n_sil_frames += 1
            percentage_silence = n_sil_frames / len(frames) if len(frames) > 0 else 0
            sr.append(percentage_silence)
            if target not in per_class_sil:
                per_class_sil[target] = []

            per_class_sil[target] = per_class_sil.get(target).append(
                1 - percentage_silence
            )
            f.write(f"{el['name']}: {percentage_silence:.3f}\n")

        f.write(f"Per class presence rate:\n")
        for key in per_class_sil.keys():
            presence_rate = (
                sum(per_class_sil[key])
                * 100
                / (len(per_class_sil) * len(per_class_sil[key]))
            )
            pr.append(presence_rate)
            f.write(f"{key}: {presence_rate:.2f}\n")
        avg_sil = sum(sr) * 100.0 / len(sr)
        f.write(f"Total average silence rate: {avg_sil:.2f}\n")
        print(sum(pr))


def check_silence_frame(audio_frame):
    eps = 1e-20
    energy = torch.sum(audio_frame**2) / audio_frame.numel()
    en_db = 10 * torch.log10(energy + eps)
    if en_db < -90:
        return True
    return False


def get_wavelet_plots(lowpass_filter, highpass_filter, fs):
    # Get the fft of the  filters

    frequency_response_lp = torch.fft.fft(lowpass_filter)
    frequency_response_hp = torch.fft.fft(highpass_filter)

    threshold = 1e-6
    frequency_response_hp[abs(frequency_response_hp) < threshold] = 0.0
    frequency_response_lp[abs(frequency_response_lp) < threshold] = 0.0
    frequency_response_lp = frequency_response_lp[: len(frequency_response_lp) // 2]
    frequency_response_hp = frequency_response_hp[: len(frequency_response_hp) // 2]
    # Compute the frequencies corresponding to the FFT output

    frequencies = (
        torch.arange(len(frequency_response_lp)) * fs / (len(frequency_response_lp) * 2)
    )

    # Compute the magnitude and phase of the frequency response
    magnitude_lp = torch.abs(frequency_response_lp)
    phase_lp = np.unwrap(torch.angle(frequency_response_lp).numpy())
    magnitude_hp = torch.abs(frequency_response_hp)
    phase_hp = np.unwrap(torch.angle(frequency_response_hp).numpy())

    # Plot the wavelet filters
    plt.figure(figsize=(14, 7))

    plt.subplot(2, 1, 1)
    plt.plot(
        lowpass_filter[
            int(len(highpass_filter) * 0.49) : int(len(highpass_filter) * 0.511)
        ],
        label="Lowpass Filter",
    )
    plt.title("Lowpass Filter (Scaling Function)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(
        highpass_filter[
            int(len(highpass_filter) * 0.49) : int(len(highpass_filter) * 0.511)
        ],
        label="Highpass Filter",
    )
    plt.title("Highpass Filter (Wavelet Function)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")

    plt.tight_layout()

    impulse_fig = plt.gcf()

    plt.figure(figsize=(14, 7))

    plt.subplot(2, 2, 1)
    plt.plot(frequencies, magnitude_lp, label="Lowpass Filter Magnitude")
    plt.title("Lowpass Filter(Scaling Function)\nFFT Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.subplot(2, 2, 2)
    plt.plot(frequencies, phase_lp, label="Lowpass Filter Phase")
    plt.title("Lowpass Filter(Scaling Function)\nFFT Unwrapped Phase")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase")

    plt.subplot(2, 2, 3)
    plt.plot(frequencies, magnitude_hp, label="Highpass Filter Magnitude")
    plt.title("Highpass Filter(Wavelet Function)\nFFT Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.subplot(2, 2, 4)
    plt.plot(frequencies, phase_hp, label="Highpass Filter Phase")
    plt.title("Highpass Filter(Wavelet Function)\nFFT Unwrapped Phase")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase")

    plt.tight_layout()

    fft_fig = plt.gcf()

    plt.figure(figsize=(14, 7))

    plt.subplot(2, 2, 1)
    plt.semilogx(
        frequencies,
        20 * torch.log10(magnitude_lp / max(magnitude_lp)),
        label="Lowpass Filter Magnitude",
    )
    plt.title("Lowpass Filter(Scaling Function)\nFFT Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(10, 22100)
    plt.ylim(-20, 1)

    plt.subplot(2, 2, 3)
    plt.semilogx(
        frequencies,
        20 * torch.log10(magnitude_hp / max(magnitude_hp)),
        label="Highpass Filter Magnitude",
    )
    plt.title("Highpass Filter(Wavelet Function)\nFFT Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(10, 22100)
    plt.ylim(-20, 1)

    plt.subplot(2, 2, 2)
    plt.semilogx(frequencies, 20 * np.log10(phase_lp), label="Lowpass Filter Magnitude")
    plt.title("Lowpass Filter(Scaling Function)\nFFT Unwrapped Phase")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (dB)")
    plt.xlim(10, 22100)

    plt.subplot(2, 2, 4)
    plt.semilogx(
        frequencies, 20 * np.log10(phase_hp), label="Highpass Filter Magnitude"
    )
    plt.title("Highpass Filter(Wavelet Function)\nFFT Unwrapped Phase")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (dB)")
    plt.xlim(10, 22100)

    plt.tight_layout()

    fft_semilogx_fig = plt.gcf()

    return impulse_fig, fft_fig, fft_semilogx_fig

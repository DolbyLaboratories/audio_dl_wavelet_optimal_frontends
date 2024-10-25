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

import matplotlib.pyplot as plt
from scipy.signal import freqz
import numpy as np


def cumulative_frequency_response_from_impulses(impulse_responses):
    w, h_cumulative = None, None
    for impulse in impulse_responses:
        w, h = freqz(impulse, worN=8000)
        if h_cumulative is None:
            h_cumulative = np.abs(h)
        else:
            h_cumulative += np.abs(h)
    return w, h_cumulative / np.max(h_cumulative)


def get_cumulative_frequency_response(impulse_responses, fs):
    n_filt = impulse_responses.shape[0]
    w, h_cumulative = cumulative_frequency_response_from_impulses(
        impulse_responses.view(n_filt, -1)
    )

    plt.figure(figsize=(10, 6))
    plt.semilogx(w * fs / (2 * np.pi), 20 * np.log10(h_cumulative), "b")
    plt.title("Cumulative Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.xlim(10, 10**4)
    plt.ylim(-20, 1)
    plt.grid()
    plt.tight_layout()

    return plt.gcf()  # Get current figure

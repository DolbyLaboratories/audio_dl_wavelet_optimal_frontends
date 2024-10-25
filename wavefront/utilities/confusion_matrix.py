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

import numpy as np
import matplotlib.pyplot as pyp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools


def get_confusion_matrix(
    y_true,
    y_pred,
    classes_indices,
    classes_labels,
    normalize=None,
):
    if all(isinstance(element, list) for element in y_true):
        y_true = list(itertools.chain.from_iterable(y_true))
        y_pred = list(itertools.chain.from_iterable(y_pred))

    cm_numpy = confusion_matrix(
        y_true,
        y_pred,
        labels=[int(idx) for idx in classes_indices],
        normalize=normalize,
    )
    cm_figure = ConfusionMatrixDisplay(
        confusion_matrix=cm_numpy[-15:, -15:], display_labels=classes_labels[-15:]
    )
    pyp.close()
    values_format = None if normalize is None else ".1f"
    return cm_figure.plot(colorbar=False, values_format=values_format).figure_, cm_numpy

    # confusion matrix
    # normalize : {'true', 'pred', 'all'}, default=None
    #     Normalizes confusion matrix over the true (rows), predicted (columns)
    #     conditions or all the population. If None, confusion matrix will not be
    #     normalized.


def get_distance_from_identity(cm_mat, n_classes):
    return np.sum(np.abs(cm_mat - np.identity(n_classes)))

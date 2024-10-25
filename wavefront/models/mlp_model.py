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

import string
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import itertools
from torch import nn

from wavefront.utilities.confusion_matrix import *


def act_fun(act_type):
    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.Softmax(dim=1)  # initially LogSoftMax

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!


class MLP(nn.Module):
    def __init__(self, options, input_dim, n_classes=None):
        super(MLP, self).__init__()

        self.input_dim = int(input_dim)
        self.lay = options.lay
        self.fc_drop = options.drop
        self.use_batchnorm = options.use_batchnorm
        self.use_laynorm = options.use_laynorm
        self.use_laynorm_inp = options.use_laynorm_inp
        self.use_batchnorm_inp = options.use_batchnorm_inp
        self.fc_act = options.act

        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        # input layer normalization
        if self.use_laynorm_inp:
            self.ln0 = nn.LayerNorm(self.input_dim)

        # input batch normalization
        if self.use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        if n_classes is not None:
            self.N_lay = 1
            self.lay = [n_classes]
        else:
            self.N_lay = len(self.lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_lay):
            # dropout
            self.drop.append(nn.Dropout(p=self.fc_drop[i]))

            # activation
            self.act.append(act_fun(self.fc_act[i]))

            add_bias = True

            if self.use_laynorm[i] or self.use_batchnorm[i]:
                # layer norm initialization
                self.ln.append(nn.LayerNorm(self.lay[i]))
                self.bn.append(nn.BatchNorm1d(self.lay[i], momentum=0.05))
                add_bias = False

            # Linear operations
            self.wx.append(nn.Linear(current_input, self.lay[i], bias=add_bias))

            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(
                torch.Tensor(self.lay[i], current_input).uniform_(
                    -np.sqrt(0.01 / (current_input + self.lay[i])),
                    np.sqrt(0.01 / (current_input + self.lay[i])),
                )
            )
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.lay[i]))

            current_input = self.lay[i]

    def forward(self, x):
        # Applying Layer/Batch Norm
        if bool(self.use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.use_batchnorm_inp):
            x = self.bn0((x))

        for i in range(self.N_lay):
            if self.act[i] != "linear":
                if self.use_laynorm[i]:
                    x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))

                if self.use_batchnorm[i]:
                    x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))

                if self.use_batchnorm[i] == False and self.use_laynorm[i] == False:
                    x = self.drop[i](self.act[i](self.wx[i](x)))

            else:
                if self.use_laynorm[i]:
                    x = self.drop[i](self.ln[i](self.wx[i](x)))

                if self.use_batchnorm[i]:
                    x = self.drop[i](self.bn[i](self.wx[i](x)))

                if self.use_batchnorm[i] == False and self.use_laynorm[i] == False:
                    x = self.drop[i](self.wx[i](x))

        return x


class CNN(nn.Module):
    def __init__(self, options, input_dim):
        super(CNN, self).__init__()

        self.cnn_N_filt = options.cnn_N_filt
        self.cnn_len_filt = options.cnn_len_filt
        self.cnn_max_pool_len = options.cnn_max_pool_len
        self.cnn_act = options.cnn_act
        self.cnn_drop = options.cnn_drop
        self.cnn_use_laynorm = options.cnn_use_laynorm
        self.cnn_use_batchnorm = options.cnn_use_batchnorm
        self.input_dim = input_dim
        self.N_cnn_lay = len(options.cnn_N_filt)
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        in_coefs, current_input = self.input_dim

        for i in range(self.N_cnn_lay):
            current_output = int(
                (current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]
            )

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))

            # layer norm initialization (why over two last dimensions???)
            self.ln.append(
                nn.LayerNorm(
                    [
                        self.cnn_N_filt[i],
                        current_output,
                    ]
                )
            )

            self.bn.append(
                nn.BatchNorm1d(
                    self.cnn_N_filt[i],
                    momentum=0.05,
                )
            )
            print(in_coefs, self.cnn_N_filt[i], self.cnn_len_filt[i])
            self.conv.append(
                nn.Conv1d(in_coefs, self.cnn_N_filt[i], self.cnn_len_filt[i])
            )
            in_coefs = self.cnn_N_filt[i]
            current_input = current_output

        self.out_dim = current_input * in_coefs

    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[1]

        for i in range(self.N_cnn_lay):
            if self.cnn_use_laynorm[i]:
                x = self.drop[i](
                    self.act[i](
                        self.ln[i](
                            F.max_pool1d(
                                torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i]
                            )
                        )
                    )
                )

            elif self.cnn_use_batchnorm[i]:
                x = self.drop[i](
                    self.act[i](
                        self.bn[i](
                            F.max_pool1d(
                                torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i]
                            )
                        )
                    )
                )
            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                x = self.drop[i](
                    self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))
                )

        x = x.reshape(batch, -1)

        return x


class CNN_2D(nn.Module):
    def __init__(self, options, input_dim):
        super(CNN_2D, self).__init__()

        self.cnn_N_filt = options.cnn_N_filt
        self.cnn_len_filt = options.cnn_len_filt
        self.cnn_max_pool_len = options.cnn_max_pool_len
        self.cnn_act = options.cnn_act
        self.cnn_drop = options.cnn_drop
        self.cnn_use_laynorm = options.cnn_use_laynorm
        self.cnn_use_batchnorm = options.cnn_use_batchnorm
        self.input_dim = input_dim
        self.N_cnn_lay = len(options.cnn_N_filt)
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        in_coefs, in_channs, current_input = self.input_dim

        for i in range(self.N_cnn_lay):
            current_output = int(current_input / self.cnn_max_pool_len[i])
            out_channs = int(in_channs / self.cnn_max_pool_len[i])
            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))

            # layer norm initialization (why over two last dimensions???)
            self.ln.append(
                nn.LayerNorm(
                    [
                        self.cnn_N_filt[i],
                        out_channs,
                        current_output,
                    ]
                )
            )

            self.bn.append(
                nn.BatchNorm2d(
                    # Also huge error commited by authors
                    self.cnn_N_filt[i],
                    momentum=0.05,
                )
            )

            self.conv.append(
                nn.Conv2d(
                    in_coefs,
                    self.cnn_N_filt[i],
                    kernel_size=(self.cnn_len_filt[i], self.cnn_len_filt[i]),
                    padding="same",
                )
            )
            in_coefs = self.cnn_N_filt[i]
            current_input = current_output
            in_channs = out_channs

        self.out_dim = out_channs * current_input * in_coefs

    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[1]

        for i in range(self.N_cnn_lay):
            if self.cnn_use_laynorm[i]:
                x = self.drop[i](
                    self.act[i](
                        self.ln[i](
                            F.max_pool2d(
                                torch.abs(self.conv[i](x)),
                                kernel_size=(
                                    self.cnn_max_pool_len[i],
                                    self.cnn_max_pool_len[i],
                                ),
                            )
                        )
                    )
                )

            elif self.cnn_use_batchnorm[i]:
                x = self.drop[i](
                    self.act[i](
                        self.bn[i](
                            F.max_pool2d(
                                torch.abs(self.conv[i](x)),
                                kernel_size=(
                                    self.cnn_max_pool_len[i],
                                    self.cnn_max_pool_len[i],
                                ),
                            )
                        )
                    )
                )
            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                x = self.drop[i](
                    self.act[i](
                        F.max_pool2d(
                            self.conv[i](x),
                            kernel_size=(
                                self.cnn_max_pool_len[i],
                                self.cnn_max_pool_len[i],
                            ),
                        )
                    )
                )

        x = x.reshape(batch, -1)

        return x


class Model(pl.LightningModule):
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
        super().__init__()
        self.save_hyperparameters()  # this is to be able to save and properly load checkpoints
        self.contains_silence = contains_silence

        self.weights = weights
        if (
            self.weights is not None
        ):  # we'll probably have to change to adapt to architecture
            self.weights = torch.Tensor(weights)
        self.learning_rate = opts.optimization.lr
        self.lr = self.learning_rate
        cw_len = opts.windowing.cw_len
        self.fs = sample_rate if sample_rate is not None else opts.windowing.fs
        self.wlen = int(self.fs * cw_len / 1000.00)

        self.y_true = []
        self.y_pred = []
        self.validation_step_loss = []
        self.training_step_loss = []
        self.training_step_acc = []
        self.training_step_weighted_acc = []
        self.pred_snt = []
        self.lab_snt = []

        # labels for confusion matrix
        self.classes_indices = (
            classes_indices
            if classes_indices
            else [str(idx) for idx in range(n_classes)]
        )
        mult = np.ceil(n_classes / len(string.ascii_letters)).astype(int)
        self.classes_labels = (
            classes_labels
            if classes_labels
            else [
                letter
                for idx, letter in enumerate(list(string.ascii_letters * mult))
                if idx < n_classes
            ]
        )

    def forward(self, x):  # this is to make prediction (or test)
        h = self.layers(x)
        probs = torch.softmax(h, dim=1)  # these are probabilities now!
        # predicted_class = torch.argmax(probs)
        return probs

    def step(self, batch, batch_idx):
        # call this from training and validation so you have the exact same function
        x, y = batch
        x = x.view(-1, self.wlen)  # like a reshape in np
        y = y.view(-1)  # like a reshape in np
        h = self.layers(x)
        loss = self.cost(h, y.long())
        return loss, x, y, h

    def step_valid_test(self, batch, batch_idx):
        # call this from training and validation so you have the exact same function
        loss, x, y, h = self.step(batch, batch_idx)

        probs = torch.softmax(h, dim=-1)  # these are probabilities now!
        if self.contains_silence:
            added_output = torch.sum(
                probs[y != self._hparams.n_classes - 1], dim=0
            )  # add probabilities for sentence classification
        else:
            added_output = torch.sum(probs, dim=0)
        probs_snt = torch.softmax(added_output, dim=0)
        predicted_classes = torch.argmax(probs, axis=1)
        pred_snt = torch.argmax(probs_snt)

        global_accuracy = sum(predicted_classes == y) / len(y)

        yn = y.cpu().numpy()
        pc = predicted_classes.cpu().numpy()
        class_accuracy = []
        for cl in range(self._hparams.n_classes):
            mask = yn == cl
            if any(mask):
                pcm = pc[mask]
                ynm = yn[mask]
                this_class_accuracy = sum(pcm == ynm) / (len(ynm) if len(ynm) else 1)
                class_accuracy.append(this_class_accuracy)

        weighted_class_accuracy = torch.mean(torch.Tensor(class_accuracy).float())

        return (
            global_accuracy,
            weighted_class_accuracy,
            loss,
            predicted_classes,
            pred_snt.cpu(),
        )

    def training_step(self, batch, batch_idx):
        loss, x, y, h = self.step(batch, batch_idx)
        self.training_step_loss.append(loss)
        self.log("train_loss", loss, sync_dist=True)

        predicted_classes = torch.argmax(h, axis=1)

        global_accuracy = sum(predicted_classes == y) / len(y)
        self.log("train_accuracy", global_accuracy, sync_dist=True)
        self.training_step_acc.append(global_accuracy)

        yn = y.cpu().numpy()
        pc = predicted_classes.cpu().numpy()
        class_accuracy = []
        for cl in range(self._hparams.n_classes):
            mask = yn == cl
            if any(mask):
                pcm = pc[mask]
                ynm = yn[mask]
                this_class_accuracy = sum(pcm == ynm) / (len(ynm) if len(ynm) else 1)
                class_accuracy.append(this_class_accuracy)

        weighted_class_accuracy = torch.mean(torch.Tensor(class_accuracy).float())
        self.log("train_weighted_accuracy", weighted_class_accuracy, sync_dist=True)
        self.training_step_weighted_acc.append(weighted_class_accuracy)

        return loss

    def on_train_epoch_end(self):
        loss = torch.mean(torch.stack(self.training_step_loss))
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        self.training_step_loss.clear()

        acc = torch.mean(torch.stack(self.training_step_acc))
        self.log("train_accuracy", acc, on_epoch=True, sync_dist=True)
        self.training_step_acc.clear()

        w_acc = torch.mean(torch.stack(self.training_step_weighted_acc))
        self.log("train_weighted_accuracy", w_acc, on_epoch=True, sync_dist=True)
        self.training_step_weighted_acc.clear()

    def log_y(self, batch, predicted_classes, pred_snt):
        x, y_true = batch
        y_true = y_true.view(-1).long()
        y_true = y_true.cpu().numpy()
        y_pred = predicted_classes.cpu().numpy()
        if self.contains_silence:
            y_true_snt = next(
                filter(lambda x: x != self._hparams.n_classes - 1, y_true), None
            )  # get first non silence label
        else:
            y_true_snt = y_true[0]
        # save data for Confusion Matrix plot
        self.y_true.append(y_true.tolist())
        self.y_pred.append(y_pred.tolist())
        self.pred_snt.append(pred_snt)
        self.lab_snt.append(y_true_snt)
        if len(self.y_true) == 73:
            print(y_true)
        return

    def reset_y(self):
        self.y_true, self.y_pred = [], []
        return

    def validation_step(self, batch, batch_idx):
        (
            global_accuracy,
            weighted_class_accuracy,
            loss,
            predicted_classes,
            pred_snt,
        ) = self.step_valid_test(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_accuracy", global_accuracy.to("cpu"), sync_dist=True)
        self.log(
            "val_weighted_accuracy", weighted_class_accuracy.to("cpu"), sync_dist=True
        )

        self.log_y(
            batch, predicted_classes, pred_snt
        )  # changed it to before calculating cm_mat

        self.validation_step_loss.append(loss)
        return loss

    def on_validation_epoch_end(self):
        loss = torch.mean(torch.stack(self.validation_step_loss).float()).cpu().tolist()
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

        cm_fig, _ = get_confusion_matrix(
            self.y_true, self.y_pred, self.classes_indices, self.classes_labels
        )
        self.logger.experiment.add_figure("validation", cm_fig, self.current_epoch)

        cm_norm_fig, cm_mat = get_confusion_matrix(
            self.y_true,
            self.y_pred,
            self.classes_indices,
            self.classes_labels,
            normalize="true",
        )
        self.logger.experiment.add_figure(
            "validation_norm_true", cm_norm_fig, self.current_epoch
        )

        distance_from_identity = get_distance_from_identity(
            cm_mat, self._hparams.n_classes
        )
        self.log(
            "validation_cm_from_id",
            distance_from_identity,
            on_epoch=True,
            sync_dist=True,
        )

        self.reset_y()
        self.validation_step_loss.clear()

        snt_acc = np.sum(i == j for i, j in zip(self.pred_snt, self.lab_snt)) / len(
            self.lab_snt
        )

        class_accuracy = []
        yn = self.lab_snt
        pc = self.pred_snt
        for cl in range(self._hparams.n_classes):
            mask = [el == cl for el in yn]
            if any(mask):
                pcm = list(itertools.compress(pc, mask))
                ynm = list(itertools.compress(yn, mask))
                this_class_accuracy = sum(i == j for i, j in zip(pcm, ynm)) / (
                    len(ynm) if len(ynm) else 1
                )
                class_accuracy.append(this_class_accuracy)

        weighted_class_accuracy = torch.mean(torch.Tensor(class_accuracy).float())

        self.log(
            "Sentence Classification Accuracy",
            snt_acc,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "Sentence Weighted Classification Accuracy",
            weighted_class_accuracy,
            on_epoch=True,
            sync_dist=True,
        )

        cm_norm_fig, _ = get_confusion_matrix(
            self.lab_snt,
            self.pred_snt,
            self.classes_indices,
            self.classes_labels,
        )
        self.logger.experiment.add_figure(
            "validation_sentence", cm_norm_fig, self.current_epoch
        )

        cm_norm_fig, _ = get_confusion_matrix(
            self.lab_snt,
            self.pred_snt,
            self.classes_indices,
            self.classes_labels,
            normalize="true",
        )
        self.logger.experiment.add_figure(
            "validation_norm_sentence", cm_norm_fig, self.current_epoch
        )

        self.pred_snt.clear()
        self.lab_snt.clear()

    def test_step(self, batch, batch_idx):
        (
            global_accuracy,
            weighted_class_accuracy,
            loss,
            predicted_classes,
            pred_snt,
        ) = self.step_valid_test(batch, batch_idx)
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_accuracy", global_accuracy.to("cpu"), sync_dist=True)
        self.log(
            "test_weighted_accuracy", weighted_class_accuracy.to("cpu"), sync_dist=True
        )
        self.log_y(batch, predicted_classes, pred_snt)

        cm_norm_fig, cm_mat = get_confusion_matrix(
            self.y_true,
            self.y_pred,
            self.classes_indices,
            self.classes_labels,
            normalize="true",
        )
        distance_from_identity = get_distance_from_identity(
            cm_mat, self._hparams.n_classes
        )
        self.log("test_cm_from_id", distance_from_identity, sync_dist=True)

        return loss

    def test_epoch_end(self, outputs):
        loss = torch.mean(torch.stack(outputs))
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

        cm_fig, _ = get_confusion_matrix(
            self.y_true,
            self.y_pred,
            self.classes_indices,
            self.classes_labels,
        )
        self.logger.experiment.add_figure("test", cm_fig, self.current_epoch)

        cm_norm_fig, _ = get_confusion_matrix(
            self.y_true,
            self.y_pred,
            self.classes_indices,
            self.classes_labels,
            normalize="all",
        )
        self.logger.experiment.add_figure(
            "test_norm_all", cm_norm_fig, self.current_epoch
        )

        cm_norm_fig, cm_mat = get_confusion_matrix(
            self.y_true,
            self.y_pred,
            self.classes_indices,
            self.classes_labels,
            normalize="true",
        )
        self.logger.experiment.add_figure(
            "test_norm_true", cm_norm_fig, self.current_epoch
        )

        cm_norm_fig, _ = get_confusion_matrix(
            self.y_true,
            self.y_pred,
            self.classes_indices,
            self.classes_labels,
            normalize="pred",
        )
        self.logger.experiment.add_figure(
            "test_norm_pred", cm_norm_fig, self.current_epoch
        )

        distance_from_identity = get_distance_from_identity(
            cm_mat, self._hparams.n_classes
        )
        self.log("test_cm_from_id", distance_from_identity, sync_dist=True)

        self.reset_y()

        snt_acc = np.sum(i == j for i, j in zip(self.pred_snt, self.lab_snt)) / len(
            self.lab_snt
        )

        class_accuracy = []
        yn = self.lab_snt
        pc = self.pred_snt
        for cl in range(self._hparams.n_classes):
            mask = [el == cl for el in yn]
            if any(mask):
                pcm = list(itertools.compress(pc, mask))
                ynm = list(itertools.compress(yn, mask))
                this_class_accuracy = sum(i == j for i, j in zip(pcm, ynm)) / (
                    len(ynm) if len(ynm) else 1
                )
                class_accuracy.append(this_class_accuracy)

        weighted_class_accuracy = torch.mean(torch.Tensor(class_accuracy).float())

        self.log(
            "Test Sentence Classification Accuracy",
            snt_acc,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "Test Sentence Weighted Classification Accuracy",
            weighted_class_accuracy,
            on_epoch=True,
            sync_dist=True,
        )

        self.pred_snt.clear()
        self.lab_snt.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=(self.lr or self.learning_rate)
        )
        return optimizer

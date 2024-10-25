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
import importlib
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# logging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def train(config):
    version = "AST_augm_full"
    version = "testvenv"
    batch_size = config.optimization.batch_size  # v0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    module_name = "wavefront.datasets.dataset_%s" % config.dataset_name
    ds = importlib.import_module(module_name, __name__)

    # load dataset depending on dataset_name
    dataset_train = ds.DataSet(config, device)
    dataset_valid = ds.DataSet(config, device, is_train=False)

    n_classes = dataset_train.n_classes
    sample_rate = dataset_train.sample_rate
    contains_silence = dataset_train.contains_silence
    loss_func_weights = dataset_train.weights
    pl.seed_everything(config.optimization.seed)

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8,
    )

    model_name = "wavefront.models.model_%s" % config.model_name
    model = importlib.import_module(model_name, __name__)

    model = model.Model(
        n_classes,
        opts=config,
        sample_rate=sample_rate,
        classes_indices=dataset_train.classes_indices,
        classes_labels=dataset_train.classes_labels,
        contains_silence=contains_silence,
        weights=loss_func_weights,
    )

    output_model_folder = os.path.join("models")
    if not os.path.isdir(output_model_folder):
        os.mkdir(output_model_folder)
    output_model_filename = "mlp_v%s" % version
    checkpoint_callback = ModelCheckpoint(
        monitor="val_weighted_accuracy",
        mode="max",
        dirpath=output_model_folder,
        filename=output_model_filename,
    )

    output_model_filename = "mlp_snt_v%s" % version
    checkpoint_callback_snt = ModelCheckpoint(
        monitor="Sentence Classification Accuracy",
        mode="max",
        dirpath=output_model_folder,
        filename=output_model_filename,
    )

    output_model_filename = "mlp_end_v%s" % version
    checkpoint_callback_last = ModelCheckpoint(
        dirpath=output_model_folder,
        filename=output_model_filename,
    )

    # RUN TRAINER
    logger = TensorBoardLogger(
        save_dir=os.getcwd(), version=str(version), name="lightning_logs"
    )
    gpu_list = [7]
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpu_list,
        logger=logger,
        strategy=pl.strategies.ddp.DDPStrategy(
            find_unused_parameters=True
        ),  # multi gpu support
        sync_batchnorm=True if len(gpu_list) > 1 else False,
        benchmark=True,
        deterministic=False,
        max_epochs=config.optimization.N_epochs,
        callbacks=[
            checkpoint_callback,
            checkpoint_callback_snt,
            checkpoint_callback_last,
        ],
        log_every_n_steps=10,
    )

    # VALIDATION dataset
    loader_valid = DataLoader(
        dataset_valid, batch_size=1, drop_last=True, shuffle=False, num_workers=8
    )

    print(f"Version: {version}")
    trainer.fit(model, loader_train, val_dataloaders=loader_valid)

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

# Load model directly
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    ASTConfig,
    ASTModel,
)
import torch.nn as nn
import torch

from wavefront.models.mlp_model import CNN, MLP
from wavefront.models.mlp_model import Model as Md


class ASTpretrained(nn.Module):
    def __init__(self, sr, out_coefs=80):
        super(ASTpretrained, self).__init__()

        self.out_dim = out_coefs
        # int(self.model.config.hidden_size//3))
        self.sr = sr

        self.extractor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        self.model = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593", torch_dtype=torch.float16
        )  # config=custom_config
        ast_out_dim = self.model.config.hidden_size
        self.out_dim = [ast_out_dim, 1214]

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        inputs = self.extractor(
            x.cpu().numpy(),
            return_tensors="pt",
            sampling_rate=self.sr,
            max_length=x.shape[-1],
        )
        # del x
        inputs.input_values = inputs.input_values.to(torch.float16)

        with torch.no_grad():
            outputs = self.model(**inputs.to(torch.float16).to(self.model.device))
            # del inputs
            torch.cuda.empty_cache()

        # out = F.max_pool1d(outputs.last_hidden_state, 3).to(torch.float32)  # (B,seq_len, hidden_state)
        out = outputs.last_hidden_state.to(torch.float32).permute(0, 2, 1)
        # self.fc(outputs.last_hidden_state.to(torch.float32)).permute(0, 2, 1)
        return out  # .reshape(-1, self.out_dim*1214) # (B,hidden_state, seq_len)


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
        self.frontend_is_frozen = False
        self.freeze_after_n_epochs = 1500

        cnn_arch = opts.net.CNNArch
        dnn1_arch = opts.net.DNN1Arch
        dnn2_arch = opts.net.DNN2Arch

        seq_len = int(opts.windowing.cw_len * sample_rate / 1000)

        ast = ASTpretrained(sample_rate)
        cnn = CNN(cnn_arch, input_dim=ast.out_dim)
        dnn1 = MLP(dnn1_arch, input_dim=cnn.out_dim)
        dnn2 = MLP(dnn2_arch, input_dim=dnn1_arch.lay[-1], n_classes=n_classes)

        self.layers = nn.Sequential(ast, cnn, dnn1, dnn2)
        # cross entropy loss
        self.cost = nn.CrossEntropyLoss(weight=loss_weights)

        self.automatic_optimization = (
            False  # Add separate optimizers for the frontend and rest of the layers
        )

    def forward(self, x):  # this is to make prediction (or test)
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)

        opt_frontend, opt_backend = self.optimizers()
        opt_frontend.zero_grad()
        opt_backend.zero_grad()

        self.manual_backward(loss)

        opt_frontend.step()
        opt_backend.step()

        return loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        # Specific configuration for ReduceLROnPlateau scheduler that monitor val_loss !!!
        sch1, sch2 = self.lr_schedulers()

        sch1.step(self.trainer.callback_metrics["val_loss"])
        sch2.step(self.trainer.callback_metrics["val_loss"])

    def configure_optimizers(self):
        optimizer_frontend = torch.optim.AdamW(
            self.layers[0].parameters(), lr=self.lr * 2, amsgrad=True
        )
        optimizer_backend = torch.optim.AdamW(
            self.layers[1:].parameters(), lr=self.lr, amsgrad=True
        )

        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_frontend, factor=0.5
        )
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_backend, factor=0.5
        )

        return (
            {
                "optimizer": optimizer_frontend,
                "lr_scheduler": {
                    "scheduler": scheduler1,
                    "monitor": "val_loss",
                },
            },
            {
                "optimizer": optimizer_backend,
                "lr_scheduler": {
                    "scheduler": scheduler2,
                    "monitor": "val_loss",
                },
            },
        )

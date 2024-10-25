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
import pytorch_lightning as pl

from wavefront.wavefront_main import *
from wavefront.models.model_SincNet import SincNetModel
from wavefront.models.model_MFCC import MFCCModel
from wavefront.models.model_LiftingWavelet import LiftingWaveletModel
from wavefront.models.model_AST import ASTPretrainedModel

# Define new hyperparameters
new_hparams = {
    'input_dim': 784,  # Must match the original model architecture
    'hidden_dim': 128,
    'output_dim': 10,
    'lr': 0.0001  # New learning rate
}
checkpoint =
# Load the model from the checkpoint but with updated hparams
model = LiftingWaveletModel.load_from_checkpoint(
    checkpoint_path="model_checkpoint.ckpt",
    hparams=new_hparams  # Override with new hyperparameters
)

# Since `configure_optimizers` uses the `hparams` to create the optimizer, the new learning rate will be used.

# Define the trainer without using `resume_from_checkpoint`
trainer = pl.Trainer(max_epochs=20)

# Resume training
trainer.fit(model)

if __name__ == "__main__":
    import argparse
    import sys
    import os


    parser = argparse.ArgumentParser(description="Train and Predict Wavefront")
    # and or add config file to add hyper parameters for specific dataset and model learning, can be copy paste from run_from_cli_example
    # parser = argparse.ArgumentParser(
    #     description="""
    # When run without the '-q' flag this will print the fully qualified configuration. You can
    # override that configuration by creating a *.yaml file and specifying it using the config_file
    # argument. Note that if no config file is specified but a file named config.yaml exists this will be
    # used by default. You can also override individual configurations on the command line using syntax
    # such as dataset_name=timit. Command line overrides take precedence over entries in the *.yaml
    # file and any configurations not specified in the *.yaml file will use defaults from the module
    # configuration types.
    #     """
    # )

    parser.add_argument(
        "-c", "--config_file", default='config_SincNet_TIMIT.yaml', help="yaml configuration file location"
    )
    parser.add_argument("-q", "--quiet", default=False, action="store_true")
    [opts, extra_args] = parser.parse_known_args(sys.argv[1:])

    ConfigClass = ConfigClass
    if opts.config_file is not None:
        file = opts.config_file
        with open(file, "r") as fobj:
            config = load_from_yaml(ConfigClass, fobj.read())
    else:
        config = ConfigClass()

    # This line allows to override parts of the config on the command line
    # ie chain.sample_rate=32000
    config = apply_cmdline_overrides(config, extra_args)
    if not opts.quiet:
        # Print out the full config, so you know what was run
        print("Running with full configuration:")
        print(dump_to_yaml(config))

    checkpoint =
    # Load the model from the checkpoint but with updated hparams
    model = LiftingWaveletModel.load_from_checkpoint(
        checkpoint_path="model_checkpoint.ckpt",
        hparams=new_hparams  # Override with new hyperparameters
    )

    # Since `configure_optimizers` uses the `hparams` to create the optimizer, the new learning rate will be used.

    # Define the trainer without using `resume_from_checkpoint`
    trainer = pl.Trainer(max_epochs=20)

    # Resume training
    trainer.fit(model)
    train(config)

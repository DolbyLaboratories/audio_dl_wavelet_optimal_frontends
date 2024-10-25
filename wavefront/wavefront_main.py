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

from wavefront.wavefront_train import train
from wavefront.utilities.config import (
    module_config,
    load_from_yaml,
    dump_to_yaml,
    apply_cmdline_overrides,
)
import wavefront.models.models_config as models_config


@module_config
class Windowing:
    fs: int = 16000
    cw_len: int = 200
    cw_shift: int = 10
    resample: int = 16000


@module_config
class Optimization:
    lr: float = 0.001
    batch_size: int = 128
    N_epochs: int = 360
    N_batches: int = 800
    N_eval_epoch: int = 8
    seed: int = 1234


@module_config
class ConfigClass:
    dataset_name: str = "timit"
    model_name: str = "SincNet"
    config_yaml: str = ""
    windowing: Windowing = field(default_factory=Windowing)
    optimization: Optimization = field(default_factory=Optimization)
    net: models_config.ASTConfig = field(default_factory=models_config.ASTConfig)


# when calling this: OMP_NUM_THREADS=1 python wavefront_main.py -d "whatevs_dataset" -m "whatevs_model" -etc
if __name__ == "__main__":
    import argparse
    import sys
    import os
    import importlib

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
        "-c",
        "--config_file",
        default="config_SincNet_TIMIT.yaml",
        help="yaml configuration file location",
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

    model_name = "%sConfig" % config.model_name
    model_net = getattr(models_config, model_name)
    model_net = model_net()
    config.net = model_net
    config = apply_cmdline_overrides(config, extra_args)

    if not opts.quiet:
        # Print out the full config, so you know what was run
        print("Running with full configuration:")
        print(dump_to_yaml(config))

    train(config)

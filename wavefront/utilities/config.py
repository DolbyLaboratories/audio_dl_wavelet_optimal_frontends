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
from typing import Dict
from marshmallow_dataclass import dataclass
from marshmallow.validate import Equal


def module_config(cls=None, *args, **kwargs):
    """Returns the same class as was passed in, with the following
    modifications.

    The class will be made into a marshmallow dataclass which adds
    the ability to perform serialisation, deserialisation and
    validation (but adds some constraints, see marshmallow and
    marshmallow_dataclass documentation for more details).

    In addition a field called _module will be added and defaults to
    containing the name of the input class. It also has a validator
    defined that ensures that the name of this field equals the name
    of the class. The effect is to add an entry to each module in a
    configuration hierarchy that explicitly states the name of the
    class it should deserialise to. If a field is defined that is a
    Union of such classes this ensures that the correct class is created,
    even if the configured fields are valid for multiple classes in the
    Union. ie:

    .. code-block:: python

        @module_config(frozen=True)
        class ModuleA:
            param: int = 0

        @module_config(frozen=True)
        class ModuleB:
            param: int = 0

        @module_config(frozen=True)
        class HigherModule:
            submodule: Union[ModuleA, ModuleB] = field(default_factory=ModuleB)

    In this case creating a HigherModule will instantiate a ModuleB config as
    the submodule field. When serialised it will save the name 'ModuleB' so that
    upon deserialisation the same submodule will be instantiated. Without this
    addition marshmallow would attempt to create ModuleA first (as it is first
    in the Union list) and succeed because the param names are compatible.
    """

    def _process_class(cls, *args, **kwargs):
        # Create the new _module field that contains the class name and validates to always be equal to the class name
        # note: does not show up in a repr of the configuration because that would be redundant
        cls._module = field(
            repr=False,
            default=cls.__name__,
            metadata=dict(validate=Equal(cls.__name__)),
        )
        # Add the new _module field to the class annotations so that marshmallow will pick up the type annotation
        annotations = cls.__dict__.get("__annotations__", {})
        annotations["_module"] = str
        # Hand the new class over to the marshmallow_dataclass.dataclass decorator
        cls = dataclass(cls, *args, **kwargs)
        return cls

    def wrap(cls):
        return _process_class(cls, *args, **kwargs)

    # See if we're being called as @module_config or @module_config().
    if cls is None:
        # We're called with parens.
        return wrap

    # We're called as @module_config without parens.
    return wrap(cls)


def apply_cmdline_overrides(cfg, overrides):
    for extra in overrides:
        # Convert to a dict so we can manipulate it
        cfg_dict = cfg.__class__.Schema().dump(cfg)
        key, val = extra.split("=")
        sub = cfg_dict
        while "." in key:
            step, _, key = key.partition(".")
            pre = sub
            sub = sub[step]
        if key == "_module":
            # Need to fall back to defaults then for this module's sub parameters, so overwrite
            # the entire dictionary rather than replacing just the one key
            pre[step] = dict(_module=val)
        else:
            # Otherwise just replace that one key
            sub[key] = val
        # Convert back into config objects so the validation and defaults etc get applied
        cfg = cfg.__class__.Schema().load(cfg_dict)
    return cfg


def dump_to_dict(cfg_obj):
    return cfg_obj.__class__.Schema().dump(cfg_obj)


def load_from_dict(cfg_class, d):
    return cfg_class.Schema().load(d)


def dump_to_yaml(cfg_obj):
    import yaml

    return yaml.dump(dump_to_dict(cfg_obj))


def load_from_yaml(cfg_class, yaml_str: str, overrides: Dict = None):
    """The object loaded from yaml will be updated with the fields in overrides before instantiating the config class"""
    import yaml

    d = yaml.load(yaml_str, yaml.Loader)
    if overrides is not None:
        d.update(overrides)
    return load_from_dict(cfg_class, d)

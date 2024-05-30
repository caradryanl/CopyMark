import dataclasses
import functools
import importlib
import inspect
import json
import os
import re
from collections import OrderedDict
from pathlib import PosixPath
from typing import Any, Dict, Tuple, Union
import logging, tqdm

from diffusers.pipelines.pipeline_loading_utils import (
    ALL_IMPORTABLE_CLASSES,
    CONNECTED_PIPES_KEYS,
    CUSTOM_PIPELINE_FILE_NAME,
    LOADABLE_CLASSES,
    _fetch_class_library_tuple,
    _get_pipeline_class,
    _unwrap_model,
    is_safetensors_compatible,
    load_sub_model,
    maybe_raise_or_warn,
    variant_compatible_siblings,
    warn_deprecated_model_variant,
)


# borrow from diffusers.configuration_utils.py
def dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)

def test(**kwargs):
    pretrained_model_name_or_path = '../models/diffusers/stable-diffusion-v1-5/'

    from_flax = kwargs.pop("from_flax", False)
    torch_dtype = kwargs.pop("torch_dtype", None)
    provider = kwargs.pop("provider", None)
    sess_options = kwargs.pop("sess_options", None)
    device_map = kwargs.pop("device_map", None)
    max_memory = kwargs.pop("max_memory", None)
    offload_folder = kwargs.pop("offload_folder", None)
    offload_state_dict = kwargs.pop("offload_state_dict", False)
    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", True)
    variant = kwargs.pop("variant", None)

    config = dict_from_json_file(None, json_file=pretrained_model_name_or_path+'model_index.json')
    init_dict = {k: v for k, v in config.items() if not k.startswith("_") and not "safety_checker" in k}
    print(init_dict)

    init_kwargs = {}
    pipeline_class = None
    model_variants = {}
    for name, (library_name, class_name) in tqdm.tqdm(init_dict.items(), desc="Loading pipeline components..."):

        # 6.2 Define all importable classes
        from diffusers import pipelines
        is_pipeline_module = hasattr(pipelines, library_name)
        importable_classes = ALL_IMPORTABLE_CLASSES
        loaded_sub_model = None
        

        # 6.3 Use passed sub model or load class_name from library_name
        # load sub model
        loaded_sub_model = load_sub_model(
            library_name=library_name,
            class_name=class_name,
            importable_classes=importable_classes,
            pipelines=pipelines,
            is_pipeline_module=is_pipeline_module,
            pipeline_class=pipeline_class,
            torch_dtype=torch_dtype,
            provider=provider,
            sess_options=sess_options,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            model_variants=model_variants,
            name=name,
            from_flax=from_flax,
            variant=variant,
            low_cpu_mem_usage=low_cpu_mem_usage,
            cached_folder=pretrained_model_name_or_path,
        )
        print(
            f"Loaded {name} as {class_name} from `{name}` subfolder of {pretrained_model_name_or_path}."
        )

        init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

if __name__ == '__main__':
    test()
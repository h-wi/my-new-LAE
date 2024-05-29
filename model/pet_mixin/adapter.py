# -*- coding: utf-8 -*-

from typing import Dict, List

import torch
import torch.nn as nn

from ..pet import Adapter, STAdapter, Conv2dAdapter, LinearLoRA, KVLoRA, Conv2dLoRA

NAME_SEP = "/"


def normalize_name(name):
    return name.replace(".", NAME_SEP)


def denormlize_name(name):
    return name.replace(NAME_SEP, ".")


def get_submodule(
    module: nn.Module,
    name: str,
    default: nn.Module = nn.Identity(),
):
    names = name.split(NAME_SEP)
    while names:
        module = getattr(module, names.pop(0), default)
    return module


class AdapterMixin:
    adapters: nn.ModuleDict

    def attach_adapter(self, **kwargs: Dict[str, nn.Module]):
        if not isinstance(getattr(self, "adapters", None), nn.ModuleDict): # block에 Adapter가 없으면..
            self.adapters = nn.ModuleDict()
        for name, adapter in kwargs.items():
            name = normalize_name(name)
            self.adapters.add_module(name, adapter)

    def detach_adapter(self, *names: List[str]):
        adapters = {}
        if not hasattr(self, "adapters"):
            return adapters

        names = names if names else map(denormlize_name, self.adapters.keys())
        for name in names:
            adapters[name] = self.adapters.pop(normalize_name(name))
        return adapters

    def adapt_module(self, name: str, input: torch.Tensor, **kwargs):
        name = normalize_name(name)
        module = get_submodule(self, name)
        # import pdb;
        # pdb.set_trace()
        if not isinstance(module, (Adapter, STAdapter, Conv2dAdapter)):
            assert kwargs == {}, f"Unknown kwargs: {kwargs.keys()}"
        if hasattr(self, "adapters") and name in self.adapters: # self = block instance여야됨
            return self.adapters[name](module, input, **kwargs)
        return module(input, **kwargs)

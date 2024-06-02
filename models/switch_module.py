from .rotated_modules import warped_modules, WaRPModule
import torch.nn as nn

def switch_module(module):
    new_children = {}

    for name, submodule in module.named_children():
        if isinstance(submodule, nn.Conv2d) and hasattr(submodule, 'is_warp_conv'):

            switched = warped_modules[type(submodule)](submodule)
            new_children[name] = switched

        switch_module(submodule)

    for name, switched in new_children.items():
        setattr(module, name, switched)

    return module.cuda()


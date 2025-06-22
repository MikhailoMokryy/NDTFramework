from typing import Union
from framework.nbdt_wan.framework_model import FrameworkHardNBDT, FrameworkNBDT
from framework.ndf_kontschieder.framework_model import FrameworkNDF, FrameworkNDT
from framework.resnet.framework_model import FrameworkResNet18

models = dict(
    {
        "ndf": FrameworkNDF,
        "ndt": FrameworkNDT,
        "nbdt": FrameworkNBDT,
        "hard_nbdt": FrameworkHardNBDT,
        "resnet18": FrameworkResNet18,
    }
)

ModelsType = type[
    Union[
        FrameworkNDF,
        FrameworkNDT,
        FrameworkNBDT,
        FrameworkHardNBDT,
        FrameworkResNet18,
    ]
]

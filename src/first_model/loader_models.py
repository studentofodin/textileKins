from pathlib import Path

import dill as pickle
import pandas as pd

from utils.interfaces import ModelInterface
from utils.loader_files import read_yaml


def loader(model_props: dict) -> ModelInterface:
    config = read_yaml(Path("config.yaml"))
    with open(Path(model_props["model_path"]), "rb") as file:
        pickle_obj = pickle.load(file)
    model_class = model_props["model_class"]
    if model_class == "SVGP":
        from utils.adapters import AdapterSVGP
        mdl = AdapterSVGP(pickle_obj, model_props, rescale_y=True)
    elif model_class == "GPy_GPR":
        from utils.adapters import AdapterGPy
        mdl = AdapterGPy(pickle_obj, model_props, rescale_y=True)
    else:
        raise(TypeError(f"The model class {model_class} is not yet supported"))
    return mdl

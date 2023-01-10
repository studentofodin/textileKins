import pathlib as pl
import yaml
import dill

from src.base_classes.model_interface import *
from src.base_classes.model_wrapper import ModelWrapper as mw

parent_dir = pl.Path(__file__).parent
print(parent_dir)

model_name = 'unevenness_card_web'

yaml_path = parent_dir / 'models' / (model_name + '.yaml')
pkl_path = parent_dir / 'models' / (model_name + '.pkl')

with open(yaml_path, 'r') as stream:
    try:
        model_props = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

model_props['model_path'] = pkl_path

model_interface = mw.load_model(model_props)



import pathlib as pl
import yaml

from src.base_classes.model_wrapper import ModelWrapper

parent_dir = pl.Path(__file__).parent
print(parent_dir)

model_name = 'unevenness_card_web'
models_dir = parent_dir / 'models'

with open(models_dir / (model_name + '.yaml'), 'r') as stream:
    try:
        model_props = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

model_props = [model_props, model_props]
model_props_dirs = [models_dir, models_dir]
model_wrapper = ModelWrapper(model_props, model_props_dirs)

pass



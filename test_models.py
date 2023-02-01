import pathlib as pl
import yaml
import numpy as np

from src.base_classes.model_wrapper import ModelWrapper

parent_dir = pl.Path(__file__).parent

model_names = ['unevenness_card_web', 'min_area_weight']
model_dir = parent_dir / 'models'
model_props = list()
for mn in model_names:
    with open(model_dir / (mn + '.yaml'), 'r') as stream:
        try:
            model_props.append(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
model_wrapper = ModelWrapper(model_props, model_dir)

# keys = ["Ishikawa_WeightPerAreaCardDelivery", "Ishikawa_CardMassThroughputSetpoint", "Ishikawa_LayersCount",
#         "Ishikawa_DraftRatioNeedleloom1Intake", "Ishikawa_DraftRatioNeedleloom", "v_Vorreisser", "v_Arbeiter_HT",
#         "v_Wender_HT", "v_Arbeiter_VR", "v_Wender_VR"]
# values = np.ones((len(keys)))
#
# inputs = dict(zip(keys, values))

inputs = {"Ishikawa_LayersCount":1, "Ishikawa_DraftRatioNeedleloom1Intake":1, "Ishikawa_DraftRatioNeedleloom":1,
          "Ishikawa_WeightPerAreaCardDelivery":5, "Ishikawa_CardMassThroughputSetpoint":1, "v_Vorreisser":5,
          "v_Arbeiter_HT":6, "v_Wender_HT":8, "v_Arbeiter_VR":1, "v_Wender_VR":5, "v_Abnehmer":0, "mean_cylinder_worker":1}

print(model_wrapper.get_outputs(inputs))
print(model_wrapper.means)
print(model_wrapper.vars)


pass



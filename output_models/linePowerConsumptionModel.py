import gpytorch
import numpy as np
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import AdditiveKernel, RBFKernel, ScaleKernel

CARD_DELIVERY_WIDTH = 3  # m
KG_H_TO_G_MIN = 100/6


def prcnt_to_mult(prcnt: float) -> float:
    return (prcnt / 100) + 1


def unpack_dict(X: dict, training_inputs: list[str]) -> np.array:
    for key in X.keys():
        X[key] = np.array(X[key]).reshape(-1, 1)
    X_unpacked = []
    for f in training_inputs:
        if f == "D_009_NM2_AuszGeschwS_m_min":
            X_unpacked.append(X["LineSpeed"])
        elif f == "D_036_K_Durchsatz_Ist_kg_h":
            X_unpacked.append(X["CardMassThroughputSetpoint"])
        elif f == "M_010_FS_Zufuehr_m_min":
            X_unpacked.append(X["FeederDeliverySpeed"])
        elif f == "CL01_BeltSpeedActual":
            cardDeliversSpeed = \
                X["CardMassThroughputSetpoint"] * \
                KG_H_TO_G_MIN / \
                X["CardDeliveryWeightPerArea"] / \
                CARD_DELIVERY_WIDTH
            X_unpacked.append(cardDeliversSpeed)
        elif f == "M_015_NM1_Vorschub_mm_H":
            X_unpacked.append(X["Needleloom1FeedPerStroke"])
        elif f == "M_005_NM2_Vorschub_mm_H":
            # simulator does not use this parameter, so set it to median value
            X_unpacked.append(np.ones_like(X["Needleloom1FeedPerStroke"])*X["Needleloom2FeedPerStroke"])
    return np.concatenate(X_unpacked, axis=1)


likelihood = gpytorch.likelihoods.GaussianLikelihood()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood_mdl):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood_mdl)

        kernel = AdditiveKernel(
            ScaleKernel(
                RBFKernel(ard_num_dims=3, active_dims=(0, 1, 2), lengthscale_constraint=GreaterThan(2.0)),
            ),
            ScaleKernel(
                RBFKernel(ard_num_dims=3, active_dims=(0, 1, 3), lengthscale_constraint=GreaterThan(1.0)),
            ),
            ScaleKernel(
                RBFKernel(ard_num_dims=4, active_dims=(0, 1, 4, 5), lengthscale_constraint=GreaterThan(3.0)),
            ),
        )

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

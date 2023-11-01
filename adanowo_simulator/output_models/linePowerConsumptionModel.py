import gpytorch
import numpy as np
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, ScaleKernel, PolynomialKernel

CARD_DELIVERY_WIDTH = 3  # m
KG_H_TO_G_MIN = 100/6


def prcnt_to_mult(prcnt: float) -> float:
    return (prcnt / 100) + 1


def unpack_dict(X: dict, training_inputs: list[str]) -> np.array:
    X = X.copy()
    for key in X.keys():
        X[key] = np.array(X[key]).reshape(-1, 1)
    X_unpacked = []
    for f in training_inputs:
        if f == "D_XXX_K_DurchsatzTheor_kg_h":
            X_unpacked.append(X["MassThroughput"])
        elif f == "D_011_NM2_AuszGeschw_m_min":
            X_unpacked.append(X["ProductionSpeedSetpoint"])
        elif f == "M_015_NM1_Vorschub_mm_H":
            X_unpacked.append(X["Needleloom1FeedPerStroke"])
    return np.concatenate(X_unpacked, axis=1)


likelihood = gpytorch.likelihoods.GaussianLikelihood()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood_mdl):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood_mdl)

        kernel = ScaleKernel(
            RBFKernel(ard_num_dims=1, active_dims=(0,), lengthscale_constraint=GreaterThan(2.0))
        ) + ScaleKernel(
            RBFKernel(ard_num_dims=2, active_dims=(0, 2), lengthscale_constraint=GreaterThan(2.0))
        )

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

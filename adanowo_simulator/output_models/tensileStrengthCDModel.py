import gpytorch
import numpy as np
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import PolynomialKernel, RBFKernel, ScaleKernel


def unpack_dict(X: dict, training_inputs: list[str]) -> np.array:
    X = X.copy()
    for key in X.keys():
        X[key] = np.array(X[key]).reshape(-1, 1)
    X_unpacked = []
    for f in training_inputs:
        if f == "CL01_LayersCalculatorLayers":
            X_unpacked.append(X["Cross-lapperLayersCount"].round())
        elif f == "M_031_K_AbliefGew_g_m2":
            X_unpacked.append(X["CardDeliveryWeightPerArea"])
        elif f == "M_015_NM1_Vorschub_mm_H":
            X_unpacked.append(X["Needleloom1FeedPerStroke"])
        elif f == "M_007_NM1_AuszVerzug_Proznt":
            X_unpacked.append(X["Needleloom1DraftRatio"])
        elif f == "D_018_SW_Gesamtverzug_Perc":
            X_unpacked.append(X["DrawFrameDraftRatio"])
        elif f == "Fibre_A":
            X_unpacked.append(np.ones_like(X["FibreA"]))
    return np.concatenate(X_unpacked, axis=1)


likelihood = gpytorch.likelihoods.GaussianLikelihood()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood_mdl):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood_mdl)

        kernel = ScaleKernel(
            RBFKernel(ard_num_dims=1, active_dims=(0,), lengthscale_constraint=GreaterThan(3.0)) *
            RBFKernel(ard_num_dims=1, active_dims=(1,), lengthscale_constraint=GreaterThan(7.0)) *
            RBFKernel(ard_num_dims=1, active_dims=(2,), lengthscale_constraint=GreaterThan(0.5)) *
            RBFKernel(ard_num_dims=1, active_dims=(3,), lengthscale_constraint=GreaterThan(3.0)) *
            RBFKernel(ard_num_dims=1, active_dims=(4,), lengthscale_constraint=GreaterThan(0.5)) *
            RBFKernel(ard_num_dims=1, active_dims=(5,), lengthscale_constraint=GreaterThan(0.0))
        )

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

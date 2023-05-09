import gpytorch
import numpy as np
from gpytorch.kernels import (PolynomialKernel, ScaleKernel)


def unpack_dict(X: dict, training_features: list[str]) -> np.array:
    for key in X.keys():
        X[key] = np.array(X[key]).reshape(-1, 1)
    X_unpacked = []

    def prcnt_to_mult(prcnt: float) -> float:
        return (prcnt/100) + 1

    for f in training_features:
        if f == "weight_per_area_theoretical":
            weight_per_area_theoretical = \
                X["CardDeliveryWeightPerArea"] * \
                X["Cross-lapperLayersCount"].round()*2 / \
                prcnt_to_mult(X["Needleloom1DraftRatioIntake"]) / \
                prcnt_to_mult(X["Needleloom1DraftRatio"]) / \
                prcnt_to_mult(X["Cross-lapperProfiling"]*-1) / \
                prcnt_to_mult(X["DrawFrameDraftRatio"]) - \
                X["SmileEffectStrength"]
            X_unpacked.append(weight_per_area_theoretical)
    return np.concatenate(X_unpacked, axis=1)


likelihood = gpytorch.likelihoods.GaussianLikelihood()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood_mdl):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood_mdl)

        kernel = ScaleKernel(
                PolynomialKernel(power=1, ard_num_dims=1, active_dims=(0,))
        )

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

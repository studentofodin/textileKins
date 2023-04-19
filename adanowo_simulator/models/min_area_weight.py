import numpy as np
import gpytorch
from gpytorch.kernels import ScaleKernel, PolynomialKernel, RBFKernel, AdditiveKernel, ProductKernel
from gpytorch.constraints import Interval


def unpack_dict(X: dict, training_features: list[str]) -> np.array:
    for key in X.keys():
        X[key] = np.array(X[key]).reshape(-1, 1)
    X_unpacked = []

    def prcnt_to_mult(prcnt: float) -> float:
        return (prcnt/100) + 1

    for f in training_features:
        if f == "weight_per_area_theoretical":
            weight_per_area_theoretical = \
                X["Ishikawa_WeightPerAreaCardDelivery"] * \
                X["Ishikawa_LayersCount"] / \
                prcnt_to_mult(X["Ishikawa_DraftRatioNeedleloom1Intake"]) / \
                prcnt_to_mult(X["Ishikawa_DraftRatioNeedleloom"])
            X_unpacked.append(weight_per_area_theoretical)
    return np.concatenate(X_unpacked, axis=1)


likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(0.148, 0.149))
# noise is constrained because we assume three samples are taken each time, decreasing the expected sample variance by
# 1/3rd. Original noise was 0.447.


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood_mdl, lengthscale_constraint):
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

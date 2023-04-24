import numpy as np
import gpytorch
from gpytorch.kernels import ScaleKernel, PolynomialKernel, RBFKernel, AdditiveKernel, ProductKernel
from gpytorch.constraints import GreaterThan


def unpack_dict(X: dict, training_features: list[str]) -> np.array:
    for key in X.keys():
        X[key] = np.array(X[key]).reshape(-1, 1)
    X_unpacked = []
    for f in training_features:
        if f == "FG_soll":
            X_unpacked.append(X["Ishikawa_WeightPerAreaCardDelivery"])
        elif f == "mean_mass_cylinders":
            velocities = np.concatenate(
                (
                    X["v_Vorreisser"],
                    X["v_Arbeiter_HT"],
                    X["v_Wender_HT"],
                    X["v_Arbeiter_VR"],
                    X["v_Wender_VR"]
                ), axis=1
            )
            mean_masses = np.divide(X["Ishikawa_CardMassThroughputSetpoint"], velocities)
            X_unpacked.append(np.mean(mean_masses, axis=1).reshape(-1, 1))
        elif f == "Diff_ArbeiterZuWender":
            X_unpacked.append(X["v_Arbeiter_HT"] - X["v_Wender_HT"])
    return np.concatenate(X_unpacked, axis=1)


likelihood = gpytorch.likelihoods.GaussianLikelihood()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood_mdl, lengthscale_constraint):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood_mdl)

        kernel = AdditiveKernel(
            ScaleKernel(
                ProductKernel(PolynomialKernel(power=1, ard_num_dims=1, active_dims=(0,)),
                              RBFKernel(ard_num_dims=1, active_dims=(0,)))
            ),
            ScaleKernel(RBFKernel(ard_num_dims=3, lengthscale_constraint=GreaterThan(lengthscale_constraint)))
        )

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

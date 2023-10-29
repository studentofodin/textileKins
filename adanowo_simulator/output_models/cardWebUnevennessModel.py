import gpytorch
import numpy as np
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import PolynomialKernel, RBFKernel, ScaleKernel

SCALE_AREA_WEIGHT = 0.158
SCALE_THROUGHPUT = 0.030


def unpack_dict(X: dict, training_features: list[str]) -> np.array:
    X = X.copy()
    for key in X.keys():
        X[key] = np.array(X[key]).reshape(-1, 1)
    X_unpacked = []
    for f in training_features:
        if f == "FG_soll":
            X_unpacked.append(X["CardDeliveryWeightPerArea"] * SCALE_AREA_WEIGHT)
        elif f == "mean_mass_cylinders":
            velocities = np.concatenate(
                (
                    X["v_PreRoll"],
                    X["v_MainCylinder"],
                    X["v_WorkerMain"],
                    X["v_StripperMain"],
                    X["v_WorkerPre"],
                    X["v_StripperPre"]
                ), axis=1
            )
            mean_masses = np.divide(X["MassThroughput"] * SCALE_THROUGHPUT, velocities)
            X_unpacked.append(np.mean(mean_masses, axis=1).reshape(-1, 1))
        elif f == "Diff_ArbeiterZuWender":
            X_unpacked.append(X["v_WorkerMain"] - X["v_StripperMain"])
    return np.concatenate(X_unpacked, axis=1)


likelihood = gpytorch.likelihoods.GaussianLikelihood()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood_mdl):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood_mdl)

        kernel = ScaleKernel(
           PolynomialKernel(power=1, ard_num_dims=1, active_dims=(0,)) *
           RBFKernel(ard_num_dims=1, active_dims=(0,))
        ) + \
            ScaleKernel(RBFKernel(ard_num_dims=3, lengthscale_constraint=GreaterThan(0.4)))

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

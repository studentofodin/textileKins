import gpytorch
import numpy as np
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import AdditiveKernel, PolynomialKernel, ProductKernel, RBFKernel, ScaleKernel


def unpack_dict(X: dict, training_features: list[str]) -> np.array:
    for key in X.keys():
        X[key] = np.array(X[key]).reshape(-1, 1)
    X_unpacked = []
    for f in training_features:
        if f == "CL01_LayersCalculatorLayers":
            X_unpacked.append(X["Cross-lapperLayersCount"].round())
        elif f == "M_031_K_AbliefGew_g_m2":
            X_unpacked.append(X["CardDeliveryWeightPerArea"])
        elif f == "M_015_NM1_Vorschub_mm_H":
            X_unpacked.append(X["Needleloom1FeedPerStroke"])
        elif f == "M_007_NM1_AuszVerzug_Proznt":
            X_unpacked.append(X["Needleloom1DraftRatio"])
        elif f == "M_017_NM2_Einsttiefe_mm":
            X_unpacked.append(X["Needleloom2NeedlepunchDepth"])
        elif f == "D_018_SW_Gesamtverzug_Perc":
            X_unpacked.append(X["DrawFrameDraftRatio"])
        elif f == "Product_UNICO-30":
            X_unpacked.append(np.ones_like(X["ProductB"]))
        elif f == "M_010_NM2_AuszVerzug_Proz":
            X_unpacked.append(X["Needleloom2DraftRatio"])

    return np.concatenate(X_unpacked, axis=1)


likelihood = gpytorch.likelihoods.GaussianLikelihood()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood_mdl):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood_mdl)

        kernel = AdditiveKernel(
            ScaleKernel(
                ProductKernel(
                    PolynomialKernel(power=1, active_dims=(0,), offset_constraint=Interval(0, 1)),
                    PolynomialKernel(power=1, active_dims=(1,), offset_constraint=Interval(0, 3)),
                    RBFKernel(ard_num_dims=1, active_dims=(2,), lengthscale_constraint=GreaterThan(10.0)),
                    RBFKernel(ard_num_dims=1, active_dims=(3,), lengthscale_constraint=GreaterThan(5.0)),
                    RBFKernel(ard_num_dims=1, active_dims=(4,), lengthscale_constraint=GreaterThan(5.0)),
                    RBFKernel(ard_num_dims=1, active_dims=(5,), lengthscale_constraint=GreaterThan(13.0)),
                    RBFKernel(ard_num_dims=1, active_dims=(6,), lengthscale_constraint=GreaterThan(2.0)),
                    RBFKernel(ard_num_dims=1, active_dims=(7,), lengthscale_constraint=GreaterThan(5.0))
                )
            ),
            ScaleKernel(
                    PolynomialKernel(power=0, active_dims=(0,))
            )
        )

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

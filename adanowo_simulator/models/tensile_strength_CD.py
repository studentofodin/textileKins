import numpy as np
import gpytorch
from gpytorch.kernels import ScaleKernel, PolynomialKernel, RBFKernel, AdditiveKernel, ProductKernel, RQKernel
from gpytorch.constraints import GreaterThan, Interval


def unpack_dict(X: dict, training_features: list[str]) -> np.array:
    for key in X.keys():
        X[key] = np.array(X[key]).reshape(-1, 1)
    X_unpacked = []
    for f in training_features:
        if f == "CL01_LayersCalculatorLayers":
            X_unpacked.append(X["Ishikawa_LayersCount"])
        elif f == "M_031_K_AbliefGew_g_m2":
            X_unpacked.append(X["Ishikawa_WeightPerAreaCardDelivery"])
        elif f == "M_015_NM1_Vorschub_mm_H":
            X_unpacked.append(X["M_015_NM1_Vorschub_mm_H"])
        elif f == "M_007_NM1_AuszVerzug_Proznt":
            X_unpacked.append(X["Ishikawa_DraftRatioNeedleloom"])
        elif f == "M_011_NM1_EinzVerzug_Proznt":
            X_unpacked.append(X["Ishikawa_DraftRatioNeedleloom1Intake"])
        elif f == "D_018_SW_Gesamtverzug_Perc":
            X_unpacked.append(X["D_018_SW_Gesamtverzug_Perc"])
        elif f == "Product_UNICO-30":
            X_unpacked.append(np.ones_like(X["D_018_SW_Gesamtverzug_Perc"]))

    return np.concatenate(X_unpacked, axis=1)


likelihood = gpytorch.likelihoods.GaussianLikelihood()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood_mdl, lengthscale_constraint):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood_mdl)

        # kernel = AdditiveKernel(
        #     ProductKernel(
        #         ScaleKernel(
        #             PolynomialKernel(power=1, active_dims=(0,), offset_constraint=Interval(-5, 10)),
        #         ),
        #         PolynomialKernel(power=1, active_dims=(1,), offset_constraint=Interval(-5, 10)),
        #         RQKernel(ard_num_dims=2, active_dims=(0, 1)),
        #         ScaleKernel(
        #             RQKernel(ard_num_dims=1, active_dims=(2,),
        #                      lengthscale_constraint=GreaterThan(lengthscale_constraint)),
        #         ),
        #         ScaleKernel(
        #             RQKernel(ard_num_dims=3, active_dims=(3, 4, 5),
        #                      lengthscale_constraint=GreaterThan(lengthscale_constraint)),
        #         ),
        #         ScaleKernel(
        #             RBFKernel(ard_num_dims=1, active_dims=(6,)),
        #             ),
        #         ),
        #     ScaleKernel(
        #         RQKernel(ard_num_dims=7, lengthscale_constraint=GreaterThan(lengthscale_constraint)),
        #         outputscale_constraint=Interval(0, 1)
        #     ),
        # )
        kernel = ProductKernel(
            ScaleKernel(
                PolynomialKernel(power=1, active_dims=(0,), offset_constraint=Interval(0, 1)),
            ),
            ScaleKernel(
                PolynomialKernel(power=1, active_dims=(1,), offset_constraint=Interval(0, 3)),
            ),
            ScaleKernel(
                RBFKernel(ard_num_dims=1, active_dims=(2,),
                          lengthscale_constraint=GreaterThan(10)),
            ),
            ScaleKernel(
                RQKernel(ard_num_dims=1, active_dims=(3,),
                         lengthscale_constraint=GreaterThan(45.0)),
            ),
            ScaleKernel(
                RQKernel(ard_num_dims=1, active_dims=(5,),
                         lengthscale_constraint=GreaterThan(45.0)),
            ),
            ScaleKernel(
                RQKernel(ard_num_dims=1, active_dims=(6,),
                         lengthscale_constraint=GreaterThan(90.0)),
            ),
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

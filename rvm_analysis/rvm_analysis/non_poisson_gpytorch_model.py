import numpy as np
import torch
import gpytorch
import pyro
from pyro.distributions import TorchDistribution
from torch.distributions import constraints
import matplotlib.pyplot as plt
import math
from quantities import s
import tqdm
import os
import json

from rvm_analysis.colours import get_cell_colour
from rvm_analysis.save_tools import save_in_folder
from rvm_analysis.simulations import GeneralizedPoisson
from typing import Union
from pathlib import Path

# class GeneralizedPoisson(TorchDistribution):
#     """
#     Generalized Poisson distribution with mean and dispersion.
#     p(y) ∝ [(λ1 * (λ1 + λ2 * y)^(y - 1)) / y!] * exp(-(λ1 + λ2 * y))
#     Support: y in {0, 1, 2, ...}
#     λ1 > 0, and λ2 in (-1, 1) such that λ1 + λ2*y > 0
#     """
#     arg_constraints = {'rate': constraints.positive, 'dispersion': constraints.interval(-1.0, 1.0)}
#     support = constraints.nonnegative_integer
#     has_enumerate_support = True

#     def __init__(self, rate, dispersion, validate_args=None):
#         self.rate = rate
#         self.dispersion = dispersion
#         super().__init__(batch_shape=torch.broadcast_shapes(rate.shape, dispersion.shape),
#                          validate_args=validate_args)

#     def sample(self, sample_shape=torch.Size()):
#         raise NotImplementedError("Sampling not implemented for GeneralizedPoisson")

#     def log_prob(self, value):
#         if self._validate_args:
#             self._validate_sample(value)

#         # Ensure positivity
#         eps = 1e-6
#         rate = self.rate.clamp(min=eps)
#         dispersion = self.dispersion.clamp(min=-1 + eps, max=1 - eps)
#         val = value.float()

#         log_pmf = (
#             val * torch.log(rate + dispersion * val)
#             - (rate + dispersion * val)
#             - torch.lgamma(val + 1)
#         )
#         log_pmf[val == 0] = -rate[val == 0]  # handle y = 0 separately

#         return log_pmf


class GeneralizedPoissonGPModel(gpytorch.models.ApproximateGP):
    """
    GP model with a Generalized Poisson likelihood.
    """
    def __init__(self, num_inducing=64, T_max=1, name_prefix="genpoisson_gp"):
        self.name_prefix = name_prefix

        inducing_points = torch.linspace(0, T_max, num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points,
            gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        rbf_kernel = gpytorch.kernels.RBFKernel()
        periodic_kernel = gpytorch.kernels.PeriodicKernel()
        periodic_kernel.period_length = 0.1
        self.covar_module = gpytorch.kernels.ScaleKernel(periodic_kernel)

        # Dispersion parameter for the generalized Poisson (trainable)
        # self.dispersion = torch.nn.Parameter(torch.tensor(0.01))  # initial dispersion

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, x, y):
        function_dist = self.pyro_guide(x)
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            pyro.sample(self.name_prefix + ".f(x)", function_dist)

    def model(self, x, y):
        pyro.module(self.name_prefix + ".gp", self)

        function_dist = self.pyro_model(x)
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)
            rate_samples = function_samples.exp()

            # Add custom generalized Poisson likelihood
            return pyro.sample(
                self.name_prefix + ".y",
                # GeneralizedPoisson(rate_samples, self.dispersion),
                pyro.distributions.Poisson(rate_samples),
                obs=y
            )    
from torch.distributions import Beta
from gpytorch.priors import Prior

class BetaPrior(Prior):
    """A prior for a beta distribution, compatible with GPy."""
    def __init__(self, alpha: float, beta: float, validate_args: bool = False):
        # Call the base class constructor
        super().__init__()
        self._alpha = alpha
        self._beta = beta
        self._dist = Beta(concentration1=alpha, concentration0=beta, validate_args=validate_args)

    def log_prob(self, value):
        return self._dist.log_prob(value)

    def rsample(self, sample_shape=torch.Size()):
        return self._dist.rsample(sample_shape)

    def expand(self, batch_shape):
        new = BetaPrior(self._alpha, self._beta)
        new._dist = self._dist.expand(batch_shape)
        return new
    

class DiagonalKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False  # no lengthscale here

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Learn a single variance parameter
        self.register_parameter(name="raw_variance", parameter=torch.nn.Parameter(torch.tensor(0.0)))
        # Constraint to keep variance positive
        self.register_constraint("raw_variance", gpytorch.constraints.Positive())

    @property
    def variance(self):
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value):
        self._set_variance(value)

    def forward(self, x1, x2, diag=False, **params):
        # If computing diagonal elements only:
        if diag:
            # Return vector of variances for each point
            return self.variance.expand(x1.size(0))
        
        # Otherwise compute full covariance matrix, which is diagonal:
        # Create an identity matrix scaled by variance
        # Note: batch size handling omitted for simplicity
        size1 = x1.size(0)
        size2 = x2.size(0)
        if size1 != size2:
            # If x1 and x2 have different sizes, covariance off-diagonal zeros anyway
            # so return zeros matrix of (size1 x size2)
            return x1.new_zeros(size1, size2)
        else:
            # Return diagonal matrix with variance on diagonal
            return self.variance * torch.eye(size1, device=x1.device, dtype=x1.dtype)
        


class PeriodicGPModel(gpytorch.models.ApproximateGP):
    """ This is the final version of my Periodic GP, used to train the Periodic Model."""

    def __init__(self,T_min,T_max,num_inducing):
        inducing_points = torch.linspace(T_min, T_max, num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing),
            learn_inducing_locations=False
        )
        # variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        # variational_strategy = gpytorch.variational.VariationalStrategy(
        #     self, inducing_points, variational_distribution, learn_inducing_locations=True
        # )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        per_kern = gpytorch.kernels.PeriodicKernel()
        
        per_kern.register_prior("lengthscale_prior",gpytorch.priors.NormalPrior(loc=0.1,scale=0.1),"raw_lengthscale")
        # rbf_kern = gpytorch.kernels.RBFKernel()
        # rbf_kern.register_constraint("raw_lengthscale",gpytorch.constraints.GreaterThan(torch.tensor(3000.0)))

        # rbf_short_kern = gpytorch.kernels.RBFKernel()
        # rbf_short_kern.register_constraint("raw_lengthscale",gpytorch.constraints.Interval(10.0,100.0))
        # self.k_short = gpytorch.kernels.ScaleKernel(rbf_short_kern,outputscale_prior=gpytorch.priors.NormalPrior(20.0,30.0))
        # rbf_short_kern.register_constraint("raw_lengthscale",gpytorch.constraints.Interval(torch.tensor(1),torch.tensor(10)))
        self.per_scale = gpytorch.kernels.ScaleKernel(per_kern)
        self.covar_module = self.per_scale 

        # self.covar_module.register_prior("outputscale_prior",gpytorch.priors.NormalPrior(loc=signal_std,scale=),"raw_lengthscale")
        per_kern.lengthscale = 0.1
        per_kern.period_length = 300
        # Control kernel usage at test time
        # self.use_short_kernel = True

    @property
    def main_scale(self):
        return self.per_scale.outputscale 
    @main_scale.setter
    def main_scale(self, value):
        self.per_scale.outputscale  = value
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

def nonlinearity(nonlinear_func):
    """ Useful so that I can choose different link functions but only change one line of code."""
    return lambda f: nonlinear_func(f)


#Custom Likelihood: Poisson + Uniform Mixture
class PoissonUniformMixtureLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    """ This is the final version of my Poisson Periodic likelihood, used to train the Periodic Model."""
    def __init__(self, uniform_max,link_function,link_function_scale=10.0,model_mean=False):
        super().__init__()
        self.model_mean = model_mean
        self.uniform_max = uniform_max
        # self.corruption_logit = torch.nn.Parameter(torch.tensor(-4.0))  # logit of corruption probability
        self.register_parameter("raw_alpha",parameter=torch.nn.Parameter(torch.tensor(0.0,requires_grad=True),requires_grad=True))
        self.register_constraint("raw_alpha", gpytorch.constraints.LessThan(1.0))
        self.register_prior("alpha_prior",gpytorch.priors.NormalPrior(loc=0.0, scale=0.5),lambda module: module.alpha)
        self.initialize(alpha=0.0)
        self.link_function=link_function
        self.link_function_scale=link_function_scale
        

        # Define corruption_prob directly in (0, 1)
        self.register_parameter(
            name="raw_corruption_prob",
            parameter=torch.nn.Parameter(torch.tensor(0.01).logit(),requires_grad=False)  # Start at ~0.02
        )
        #! These are not used because the requires_grad attribute is False above
        self.register_constraint("raw_corruption_prob", gpytorch.constraints.Interval(0.0, 1.0))
        self.register_prior("corruption_prob_prior",BetaPrior(alpha=1.0, beta=3.0),"corruption_prob")

        uniform_log_prob_scalar = torch.log(torch.tensor(1.0 / (self.uniform_max + 1)))
        self.register_buffer("uniform_log_prob_scalar", uniform_log_prob_scalar)

    @property
    def corruption_prob(self):
        return torch.sigmoid(self.raw_corruption_prob)
    
    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)#*torch.tensor(0.8)

    @alpha.setter
    def alpha(self, value):
        self._set_alpha(value)

    def _set_alpha(self, value):
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    def forward(self, function_samples, **kwargs):
        raise NotImplementedError("Use log_marginal instead")
    
    def expected_log_prob(self, target, function_dist, **kwargs):
        
        log_1minus_corr_prob = torch.log(1 - self.corruption_prob)
        log_corr_prob = torch.log(self.corruption_prob)
        
        # Draw multiple samples from the function distribution
        f_samples = function_dist.rsample(torch.Size([30]))  # shape: (20, batch_size, ...)
        # Compute rate and Poisson log-probabilities for all samples
        rate = self.link_function_scale* self.link_function(f_samples)  # shape: (20, batch_size, ...) #!Add a factor of 10 to keep values reasonable
        if self.model_mean:
            poisson_log_prob = GeneralizedPoisson(mu=rate * (1-self.alpha), alpha=self.alpha).log_prob(target) #!
        else:
            poisson_log_prob = GeneralizedPoisson(mu=rate, alpha=self.alpha).log_prob(target) #!
        # target is broadcasted: shape (batch_size, ...) -> (20, batch_size, ...)
        
        # Uniform log-prob needs to be broadcasted to shape (20, batch_size, ...)
        uniform_log_prob = self.uniform_log_prob_scalar.expand(target.shape)
        uniform_log_prob = uniform_log_prob.expand_as(poisson_log_prob) #.unsqueeze(0).
        
        # Combine with corruption probabilities
        mix_log_prob = torch.logsumexp(torch.stack([
            log_1minus_corr_prob + poisson_log_prob,
            log_corr_prob + uniform_log_prob
        ], dim=0), dim=0)  # shape: (20, batch_size, ...)
        
        # Expectation over Monte Carlo samples
        expected_log_prob = mix_log_prob.mean(dim=0)  # shape: (batch_size, ...)
        
        return expected_log_prob

class MeanGPModel(gpytorch.models.ApproximateGP):
    def __init__(self,T_max,num_inducing):
        inducing_points = torch.linspace(0, T_max, num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing),
            learn_inducing_locations=False
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = DiagonalKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

def set_mean_and_scale(model,likelihood,link_function_scale,train_y,has_scale=True):
    """Sets the initial mean and scale of the GP model."""
    model.mean_module.constant = (train_y.max()+train_y.min()) * (1-likelihood.alpha.item())/2 / link_function_scale
    if has_scale:
        model.main_scale = train_y.var()/100 #(train_y.max() - train_y.min()) ** 2 /2 #* (1-likelihood.alpha.item())**2
        print("initial scale:", model.main_scale, "sqrt",torch.sqrt(model.main_scale))


def create_model(T_min,T_max, max_count,link_function,link_function_scale,num_data,num_inducing=200,model_mean=False):
    """Creates a periodic GP model with a PoissonUniformMixtureLikelihood. """
    model = PeriodicGPModel(T_min=T_min,T_max=T_max,num_inducing=num_inducing)
    likelihood = PoissonUniformMixtureLikelihood(uniform_max=int(max_count),
                                                 link_function=link_function,
                                                 link_function_scale=link_function_scale,model_mean=model_mean)

    # Use the variational ELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)
    return model, likelihood, mll

def create_mean_model(T_max, max_count,link_function,link_function_scale,num_data,num_inducing=200,model_mean=False):
    """ Creates a mean GP model with a PoissonUniformMixtureLikelihood."""
    model = MeanGPModel(T_max=T_max,num_inducing=num_inducing)
    likelihood = PoissonUniformMixtureLikelihood(uniform_max=int(max_count)
                                                 ,link_function=link_function,
                                                 link_function_scale=link_function_scale,model_mean=model_mean)

    # Use the variational ELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)
    return model, likelihood, mll




def plot_training_results(train_x, train_y, test_x, test_y,
                          latent_test_func_quantiles, latent_train_func_quantiles,
                          cell_name, index,quantiles_with_noise,quantiles_with_noise_training,
                          fontsize=7,cut_time_neutral=660*s,cut_time=1200*s,save_path="./results/gen_pois_no_exp/"):
    """
    Plots the training results for a cell over test and training data.

    Parameters:
    - train_x, train_y: training data
    - test_x, test_y: test data
    - latent_percentiles: (lower, mean, upper) for test predictions
    - latent_training_percentiles: (lower, mean, upper) for train predictions
    - y_sim: sample from the model (unused now)
    - cell_name, index: identifiers for the plot (unused here)
    - fontsize: unified font size for labels and legend
    """
    lower, mean, upper = latent_test_func_quantiles
    lower_train, mean_train, upper_train = latent_train_func_quantiles
    lower_poisson_test, mean_poisson_test, upper_poisson_test = quantiles_with_noise
    lower_poisson_train, mean_poisson_train, upper_poisson_train = quantiles_with_noise_training

    plt.rcParams['axes.linewidth'] = 0.5
    fig, ax = plt.subplots(figsize=(7, 1),constrained_layout=True)
    ax: plt.Axes
    # GP predictions with uncertainty for test data
    ax.plot(test_x, mean, color='black',linestyle=(0,(1,1)),linewidth=0.7)
    # ax.plot(test_x, mean_poisson_test, label='GP Likelihood test prediction', color='red')
    # ax.plot(train_x, mean_poisson_train, label='GP Likelihood train prediction', color='blue')
    ax.fill_between(
        test_x,
        lower,
        upper,
        color='slategray', alpha=0.4
    )

    # GP predictions with uncertainty for train data
    ax.plot(train_x, mean_train, label='Predicted Mean', color='black',linewidth=0.7)
    ax.fill_between(
        train_x,
        lower_train,
        upper_train,
        color='slategray', alpha=0.4,label="95% Mean CI"
    )

    ax.fill_between(
        test_x,
        lower_poisson_test,
        upper_poisson_test,
        color='gray', alpha=0.2,label="95% Observation CI"
    )

    ax.fill_between(
        train_x,
        lower_poisson_train,
        upper_poisson_train,
        color='gray', alpha=0.2
    )

    # True functions
    ax.scatter(test_x, test_y, alpha=0.8, color=get_cell_colour(names=cell_name),marker='x',s=3)
    ax.scatter(train_x, train_y, alpha=0.8, color=get_cell_colour(names=cell_name),s=3)

    # Style adjustments
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # ax.tick_params(labelsize=fontsize)
    ax.tick_params(which='both',left=True, bottom=True, labelbottom=True,labelsize=fontsize,width=0.5,length=2)

    # ax.legend(fontsize=fontsize,loc='upper right',ncol=3,bbox_to_anchor=(1.0, 1.2))
    ax.set_xlabel("Time(s)", fontsize=fontsize,labelpad=1)
    ax.set_ylabel("Spike Count", fontsize=fontsize,labelpad=1)
    ax.grid(False)
    print(cut_time_neutral)
    ax.vlines(cut_time if cell_name in ["ON","OFF","NEUTRAL_EXTRA"] else cut_time_neutral,*ax.get_ylim(), color='black',linewidth=1.0)
    ax.set_xlim(-5,torch.max(test_x).item() + 5)
    
    save_in_folder(f"{cell_name}_GP_fit",basePath=save_path,svg=True,save=False)



def train_model(model, likelihood, mll,train_x,train_y,training_iterations=1000,lr=0.1,plot=True,mean_model=False):        
    """Trains the Poisson GP model."""
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)

    iterator:tqdm.notebook.tqdm = tqdm.notebook.tqdm(range(training_iterations))
    loss_history = []
    outputscales = []

    for i in iterator:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y) #- torch.pow(likelihood.alpha,2)
        loss.backward()
        loss_history.append(float(loss.detach().float()))
        if not mean_model:
            outputscales.append(float(model.main_scale.item()))
        optimizer.step()
        if mean_model:
            postfix = {"loss": f"{loss.item():.4f}"}
        else:
            postfix = {"loss": f"{loss.item():.4f}","outputscale": f"{model.main_scale.item():.4f}"}
        iterator.set_postfix(postfix)

        # Plot the loss
    fig, ax = plt.subplots(1,2,figsize=(25, 2))
    ax[0].plot(loss_history, label="ELBO Loss")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("SVI Training Loss")
    ax[0].legend()
    if not mean_model:
        ax[1].plot(outputscales,label="output scale")
        ax[1].set_title("Kernel outputscale")
        ax[1].legend()
    else:
        ax[1].remove()
    plt.show()

    return loss_history

def import_params(fitted_param_dir: str,param_dict_name:str = "params.json"):
    """Imports the fitted GP parameters"""
    with open(os.path.join(fitted_param_dir,"params.json"),'r') as f:
        params = json.load(f)
    return params


from rvm_analysis.simulations import generalized_poisson_sample


def get_gen_poiss_samples(model,link_function,link_function_scale,test_x,train_x,w,model_mean=False,num_samples=200):
    """
    Returns, in order:
    1) Test x function 2.5 percent, mean, 97.5 percent
    2) Test x sample 2.5 percent, mean, 97.5 percent
    3) Train x function 2.5 percent, mean, 97.5 percent
    4) Test x sample 2.5 percent, mean, 97.5 percent
    """
    model.eval()
    with torch.no_grad():
        outputs = [model(test_x),model(train_x)] #* Do test then train
        quantile_lists: list[list] = []
        for output in outputs:
            rng = np.random.default_rng()

            # Step 1: Sample f(x)
            num_samples = num_samples
            # f_samples = output.rsample(torch.Size([num_samples]))  # shape: [S, N]
            f_samples = output.sample(torch.Size([num_samples]))  # shape: [S, N]

            # Step 2: Transform to lambda
            # lambda_samples = torch.exp(f_samples)#!.exp()  # shape: [S, N]
            lambda_samples = link_function_scale * link_function(f_samples)#!.exp()  # shape: [S, N]
            if model_mean:
                mean_samples = lambda_samples
            else:
                mean_samples = lambda_samples / (1-w)

            lower, upper = [np.percentile(mean_samples, p, axis=0) for p in [2.5,97.5]]

            quantile_lists.append([lower,mean_samples.mean(axis=0),upper])

            k_vals = np.arange(0, 2000)

            # Initialize empty count samples: shape [S, N]
            count_samples = np.zeros((num_samples, lambda_samples.shape[1]))

            for i in range(num_samples):
                for j in range(lambda_samples.shape[1]):
                    if model_mean == True:
                        param_samples = lambda_samples[i, j].item() * (1-w)
                    else:
                        param_samples = lambda_samples[i,j].item()
                    count_samples[i, j] = generalized_poisson_sample(param_samples, #!w #! #!* (1-w)
                                                                     alpha=w,#w, #! w
                                                                     k_vals=k_vals, rng=rng)

            # Step 4: Compute percentiles (97.5% credible interval)
            lower, upper = [np.percentile(count_samples, p, axis=0) for p in [2.5,97.5]]
            quantile_lists.append([lower,count_samples.mean(axis=0),upper])
    return quantile_lists




def get_summary_parameters_and_fits(model,likelihood,model_id,cell_name,model_index,results=None):
    """
    Gets summaries of all of the GP parameters, and their respective fits. Returns the parameter stats.
    """
    param_subset = {
        "lengthscale": model.covar_module.base_kernel.lengthscale.item(),
        "period_length": model.covar_module.base_kernel.period_length.item(),
        "outputscale": model.covar_module.outputscale.item(),
        "w": likelihood.alpha.item(),
        "noise_prob": likelihood.corruption_prob.item()
    }

    for key, value in param_subset.items():
        print(key, f"{value:.4f}",end=" ")


    stats = {
        'model_id': model_id,
        'cell_name': cell_name,
        'id': model_index,
        # 'r2': r2,
        # 'pseudo-r2': pseudo_r2,
        # 'pseudo-r2-train': pseudo_r2_train,
        # 'pseudo-r2-adjusted': pseudo_r2_adjusted,
        # 'pseudo-r2-train-adjusted': pseudo_r2_train_adjusted,
        **param_subset
    }
    if results is not None:
        stats = {
            **stats,
            **results
        }
    return stats


def evaluate_model(model,link_function,link_function_scale,likelihood,test_x,train_x,test_y,train_y,
                   model_index,cell_name,plot=True,model_mean=False,plot_CI_samples=200):
    """ Generates samples from the model at the required points, for the training and test data."""

    with torch.no_grad(),gpytorch.settings.fast_pred_var():

        latent_percents,quantiles_with_noise,latent_percents_train,quantiles_with_noise_training = get_gen_poiss_samples(
        model,link_function,link_function_scale,test_x,train_x,likelihood.alpha.numpy(),model_mean=model_mean,num_samples=plot_CI_samples)

        return latent_percents,quantiles_with_noise,latent_percents_train,quantiles_with_noise_training
        # if plot:
        #     plot_training_results_resids(
        #         train_x, train_y, test_x, test_y,
        #         latent_percents,
        #         latent_percents_train,
        #         cell_name,model_index, quantiles_with_noise
        #         ,quantiles_with_noise_training,fontsize=7,
        #     )

def save_model_params(model,likelihood,model_param_path: Union[str,Path],lik_param_path: Union[str,Path],save: bool=False):
    """ Saves the parameters of a model and likelihood. Save is whether to actually save anything or not."""
    if save:
        torch.save(model.state_dict(), model_param_path)         
        torch.save(likelihood.state_dict(), lik_param_path)
        print(f"Saved model parameters to {model_param_path}")





def compare_log_liks(model, likelihood, mean_model, mean_liks, train_x, train_y, test_x, test_y):
    """
    Compare two models in terms of expected log predictive densities (ELPD) on train and test data.
    Also prints a pseudo-R2 metric for each.

    Args:
        model: Trained GP model
        likelihood: GP likelihood used with the model
        mean_model: Baseline (mean-only) GP model
        mean_liks: Likelihood used with the mean_model
        train_x, train_y: Training data (tensors)
        test_x, test_y: Test data (tensors)
    """
    model.eval()
    likelihood.eval()
    mean_model.eval()
    mean_liks.eval()

    with torch.no_grad():
        # Get latent distributions (model outputs before likelihood)
        output_train = model(train_x)
        output_test = model(test_x)

        mean_output_train = mean_model(train_x)
        mean_output_test = mean_model(test_x)

        # Compute expected log predictive densities (ELPDs) using expected_log_prob
        elpd_train = likelihood.expected_log_prob(train_y, output_train)
        mean_elpd_train = mean_liks.expected_log_prob(train_y, mean_output_train)

        elpd_test = likelihood.expected_log_prob(test_y, output_test)
        mean_elpd_test = mean_liks.expected_log_prob(test_y, mean_output_test)

        # Sum over data points to get total ELPD
        total_elpd_train = elpd_train.sum()
        total_mean_elpd_train = mean_elpd_train.sum()

        total_elpd_test = elpd_test.sum()
        total_mean_elpd_test = mean_elpd_test.sum()

        # Compute pseudo-R2 (relative improvement over mean model)
        pseudo_r2_train = 1 - total_elpd_train / total_mean_elpd_train
        pseudo_r2_test = 1 - total_elpd_test / total_mean_elpd_test

        # Print results
        results = {
            "Train Model ELPD": total_elpd_train.item(),
            "Train Mean Model ELPD": total_mean_elpd_train.item(),
            "Train Pseudo-R2": pseudo_r2_train.item(),
            "Test Model ELPD": total_elpd_test.item(),
            "Test Mean Model ELPD": total_mean_elpd_test.item(),
            "Test Pseudo-R2": pseudo_r2_test.item(),
        }
        print("=== TRAIN DATA ===","=== TEST DATA ===")
        print(f"Pseudo-R2: {pseudo_r2_train.item():.4f}",f"   {pseudo_r2_test.item():.4f}")

        return results
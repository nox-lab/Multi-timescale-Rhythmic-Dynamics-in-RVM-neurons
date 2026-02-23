import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import torch
from torch.distributions import Distribution
from torch.distributions import constraints
from gpytorch.likelihoods import Likelihood
from torch import nn
import scipy
import torch
from torch.distributions import constraints
from torch.distributions import Distribution, constraints

def simulate_cmp_timeseries(
    timesteps=200,
    lam_base=100,
    lam_amp=40,
    lam_freq=1.0,
    nu=2.0,
    k_max=300,
    plot=True,
    seed=None,
    T=1200
):
    """
    Simulate CMP-distributed count time series with sinusoidal rate.

    Parameters:
        timesteps (int): Number of time points.
        lam_base (float): Baseline λ.
        lam_amp (float): Amplitude of sinusoidal λ.
        lam_freq (float): Frequency of sinusoidal λ.
        nu (float): CMP dispersion parameter (>1 = underdispersion).
        k_max (int): Max value to consider for truncated PMF.
        plot (bool): Whether to plot the time series.
        seed (int or None): Random seed.

    Returns:
        t (np.ndarray): Time points.
        lam_t (np.ndarray): True λ(t).
        samples (np.ndarray): CMP samples at each time.
    """
    rng = default_rng(seed)
    t = np.linspace(0, T, timesteps)
    lam_t = np.exp(lam_base + lam_amp * np.sin(t * lam_freq* 2 * np.pi))

    k_vals = np.arange(0, k_max)

    def cmp_sample(lam_val):
        log_unnorm = k_vals * np.log(lam_val) - nu * scipy.special.gammaln(k_vals + 1)
        unnorm = np.exp(log_unnorm - np.max(log_unnorm))  # stabilize
        probs = unnorm / np.sum(unnorm)
        cdf = np.cumsum(probs)
        u = rng.uniform()
        return k_vals[np.searchsorted(cdf, u)]

    samples = np.array([cmp_sample(lam) for lam in lam_t])

    if plot:
        plt.figure(figsize=(10, 4))
        # plt.plot(t, lam_t, label="λ(t) - true rate", color='blue')
        plt.scatter(t, samples, label="CMP samples", color='orange', alpha=0.7)
        plt.title(f"CMP Time Series (ν={nu})")
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        

    return t, lam_t, samples

def cmp_sample(lam_val,nu,k_vals,rng):
    log_unnorm = k_vals * np.log(lam_val) - nu * scipy.special.gammaln(k_vals + 1)
    unnorm = np.exp(log_unnorm - np.max(log_unnorm))  # stabilize
    probs = unnorm / np.sum(unnorm)
    cdf = np.cumsum(probs)
    u = rng.uniform()
    if np.sum(probs) < 0.999:
        print("Warning: probability mass truncated; consider increasing range of k_vals.")
    return k_vals[np.searchsorted(cdf, u)]

def simulate_cmp_from_rate(lam_t,
    nu=2.0,
    k_max=300,
    seed=None,
):
    """
    Simulate CMP-distributed count time series with sinusoidal rate.

    Parameters:
        lam_t (array): The base rate to sample from
        nu (float): CMP dispersion parameter (>1 = underdispersion).
        k_max (int): Max value to consider for truncated PMF.
        seed (int or None): Random seed.

    Returns:
        samples (np.ndarray): CMP samples at each time.
    """
    rng = default_rng(seed)

    k_vals = np.arange(0, k_max)

    samples = np.array([cmp_sample(lam,nu,k_vals,rng) for lam in lam_t])

    return samples

import torch
from torch.distributions import constraints, Distribution

import torch
from torch.distributions import constraints, Distribution

class ConwayMaxwellPoisson0Corrected(Distribution):
    arg_constraints = {'rate': constraints.nonnegative, 'nu': constraints.positive}
    support = constraints.nonnegative_integer

    def __init__(self, rate, nu, validate_args=None):
        self.rate = torch.as_tensor(rate,dtype=torch.float64)
        self.nu = torch.as_tensor(nu,dtype=torch.float64)
        batch_shape = self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        rate = self.rate
        nu = self.nu
        value = torch.as_tensor(value, dtype=rate.dtype)

        rate, nu, value = torch.broadcast_tensors(rate, nu, value)

        mask_point_mass = (rate == 0) & (value == 0)
        mask_zero_prob = (rate == 0) & (value > 0)
        mask_normal = rate > 0

        log_pmf = torch.full_like(rate, float('-inf'), dtype=rate.dtype)  # ensure dtype matches rate

        if mask_normal.any():
            log_rate = torch.log(rate[mask_normal])
            val_factorial = torch.special.gammaln(value[mask_normal] + 1)
            log_Z = self._log_Z(rate[mask_normal], nu[mask_normal])

            # Cast all to rate.dtype to avoid dtype mismatch
            log_rate = log_rate.to(rate.dtype)
            val_factorial = val_factorial.to(rate.dtype)
            log_Z = log_Z.to(rate.dtype)

            log_pmf[mask_normal] = value[mask_normal] * log_rate - nu[mask_normal] * val_factorial - log_Z

        log_pmf[mask_point_mass] = torch.tensor(0.0, dtype=rate.dtype)  # cast here too

        return log_pmf

    def _log_Z(self, rate, nu, max_sum=100):
        Z = torch.zeros_like(rate, dtype=rate.dtype)
        for k in range(max_sum):
            k_tensor = torch.full_like(rate, k, dtype=rate.dtype)
            term = k_tensor * torch.log(rate) - nu * torch.special.gammaln(k_tensor + 1)
            Z += torch.exp(term)
        return torch.log(Z)

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError("Sampling not implemented for CMP.")

    @property
    def mean(self):
        raise NotImplementedError("Mean is not available in closed form for CMP.")

    @property
    def variance(self):
        raise NotImplementedError("Variance is not available in closed form for CMP.")



class ConwayMaxwellPoisson(Distribution):
    arg_constraints = {'rate': constraints.nonnegative, 'nu': constraints.positive}
    support = constraints.nonnegative_integer

    def __init__(self, rate, nu, validate_args=None):
        # self.rate = rate
        # self.nu = nu
        self.rate = torch.as_tensor(rate)
        self.nu = torch.as_tensor(nu)
        batch_shape = self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        rate, nu = self.rate, self.nu
        val_factorial = torch.special.gammaln(value + 1)
        log_pmf = value * torch.log(rate) - nu * val_factorial - self._log_Z(rate, nu)
        return log_pmf

    def _log_Z(self, rate, nu, max_sum=500):
        """Log-normalizer approximation using truncated sum."""
        Z = torch.zeros_like(rate)
        for k in range(max_sum):
            term = (k * torch.log(rate) - nu * torch.special.gammaln(torch.tensor(k + 1.0, dtype=rate.dtype)))
            Z += torch.exp(term)
        return torch.log(Z)

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError("Sampling not implemented for CMP.")

    @property
    def mean(self):
        raise NotImplementedError("Mean is not available in closed form for CMP.")

    @property
    def variance(self):
        raise NotImplementedError("Variance is not available in closed form for CMP.")


class CMPUnderdispersedLikelihood(Likelihood):
    def __init__(self, init_nu=2.0):
        super().__init__()
        # nu > 1 => underdispersion; nu = 1 => Poisson; nu < 1 => overdispersion
        self.nu = nn.Parameter(torch.tensor(init_nu))

    def forward(self, function_samples, **kwargs):
        """ Function samples should already be exponentiated!!"""
        # rate = torch.exp(function_samples)
        return ConwayMaxwellPoisson(rate=function_samples, nu=self.nu.clamp(min=0.1, max=10.0))

    def expected_log_prob(self, target, function_dist, **kwargs):
        if not function_dist.has_rsample:
            raise RuntimeError("Expected rsample-able distribution for function_dist")

        samples = function_dist.rsample(sample_shape=torch.Size([10]))
        rate = torch.exp(samples)
        nu_clamped = self.nu.clamp(min=0.1, max=10.0)
        log_probs = ConwayMaxwellPoisson(rate=rate, nu=nu_clamped).log_prob(target)
        return log_probs.mean(0)
    

class GeneralizedPoisson(Distribution):
    """A fully pytorch implementation of the generalised poisson distribution."""
    arg_constraints = {'mu': constraints.nonnegative, 'alpha': constraints.real}
    support = constraints.nonnegative_integer
    has_rsample = False

    def __init__(self, mu, alpha, validate_args=None):
        self.mu = torch.as_tensor(mu,dtype=torch.float64)
        self.alpha = torch.as_tensor(alpha,dtype=torch.float64)
        if validate_args:
            if not torch.all(self.mu > 0):
                raise ValueError("mu must be > 0")
        batch_shape = torch.broadcast_shapes(self.mu.shape, self.alpha.shape)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.mu.dtype, device=self.mu.device)

        mu, alpha = torch.broadcast_tensors(self.mu, self.alpha)
        mu, alpha, value = torch.broadcast_tensors(mu, alpha, value)

        adjusted = mu + alpha * value
        log_pmf = torch.full_like(mu, float('-inf'), dtype=mu.dtype)

        # Case: mu == 0 & value == 0 => log_prob = 0
        mask_point_mass = (mu == 0) & (value == 0)
        log_pmf[mask_point_mass] = torch.tensor(0.0, dtype=mu.dtype, device=mu.device)

        # Case: mu == 0 & value > 0 => log_prob = -inf (already the default)
        # mask_zero_prob = (mu == 0) & (value > 0)

        # Case: mu > 0 & adjusted > 0
        mask_normal = (mu > 0) & (adjusted > 0)
        if mask_normal.any():
            log_fact = torch.lgamma(value[mask_normal] + 1)
            log_pmf[mask_normal] = (
                torch.log(mu[mask_normal]) +
                (value[mask_normal] - 1) * torch.log(adjusted[mask_normal]) -
                adjusted[mask_normal] -
                log_fact
            )

        return log_pmf
 

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError("Sampling not implemented for Generalized Poisson.")

    @property
    def mean(self):
        mu, alpha = self.mu, self.alpha
        return mu / (1 - alpha)

    @property
    def variance(self):
        mu, alpha = self.mu, self.alpha
        return mu / ((1 - alpha) ** 3)



def get_cmp_samples(model,test_x,train_x):
    model.eval()
    with torch.no_grad():
        outputs = [model(test_x),model(train_x)]
        nu = model.nu.item()
        print(model.nu.item())
        quantile_lists: list[list] = []
        for output in outputs:
            rng = default_rng(0)

            # Step 1: Sample f(x)
            num_samples = 500
            f_samples = output.rsample(torch.Size([num_samples]))  # shape: [S, N]

            # Step 2: Transform to lambda
            lambda_samples = f_samples.exp()  # shape: [S, N]
            lower, median, upper = [np.percentile(lambda_samples, p, axis=0) for p in [2.5,50,97.5]]

            quantile_lists.append([lower,median,upper])

            k_vals = np.arange(0, 1000)

            # Initialize empty count samples: shape [S, N]
            count_samples = np.zeros((num_samples, lambda_samples.shape[1]))

            for i in range(num_samples):
                for j in range(lambda_samples.shape[1]):
                    count_samples[i, j] = cmp_sample(lambda_samples[i, j].item(), nu=nu, k_vals=k_vals, rng=rng)

            # Step 4: Compute percentiles (95% credible interval)
            lower, median, upper = [np.percentile(count_samples, p, axis=0) for p in [2.5,50,97.5]]
            quantile_lists.append([lower,median,upper])
    return quantile_lists


import numpy as np
import scipy.special

def generalized_poisson_sample(mu_val, alpha, k_vals, rng):
    """
    Sample from the Generalized Poisson distribution using vectorized PMF and inverse CDF.

    Args:
        mu_val (float): Mean-like parameter (μ > 0)
        theta (float): Dispersion parameter
        k_vals (np.ndarray): Array of possible k values to evaluate PMF
        rng (np.random.Generator): Random number generator

    Returns:
        int: A sampled count value
    """
    # Guard for valid domain
    support_mask = mu_val + alpha * k_vals > 0
    k_vals = k_vals[support_mask]

    # Compute stabilized log-PMF (unnormalized)
    with np.errstate(divide='ignore', invalid='ignore'):
        adjusted_means = mu_val + alpha * k_vals
        log_unnorm = (
            np.log(mu_val)
            + (k_vals - 1) * np.log(adjusted_means)
            - (adjusted_means)
            - scipy.special.gammaln(k_vals + 1)
        )

    # Stabilize and normalize
    log_unnorm -= np.max(log_unnorm)
    unnorm = np.exp(log_unnorm)
    probs = unnorm / np.sum(unnorm)

    if np.sum(probs) < 0.999:
        print("Warning: probability mass truncated; consider increasing range of k_vals.")

    # Inverse CDF sampling
    cdf = np.cumsum(probs)
    u = rng.uniform()
    return k_vals[np.searchsorted(cdf, u)]

def get_gen_poiss_samples(model,test_x,train_x):
    model.eval()
    with torch.no_grad():
        outputs = [model(test_x),model(train_x)]
        w = model.w.item()
        quantile_lists: list[list] = []
        for output in outputs:
            rng = default_rng(0)

            # Step 1: Sample f(x)
            num_samples = 500
            f_samples = output.rsample(torch.Size([num_samples]))  # shape: [S, N]

            # Step 2: Transform to lambda
            lambda_samples = f_samples.exp()  # shape: [S, N]
            lower, median, upper = [np.percentile(lambda_samples, p, axis=0) for p in [2.5,50,97.5]]

            quantile_lists.append([lower,median,upper])

            k_vals = np.arange(0, 1000)

            # Initialize empty count samples: shape [S, N]
            count_samples = np.zeros((num_samples, lambda_samples.shape[1]))

            for i in range(num_samples):
                for j in range(lambda_samples.shape[1]):
                    count_samples[i, j] = generalized_poisson_sample(lambda_samples[i, j].item() * (1-w),
                                                                     alpha=w,
                                                                     k_vals=k_vals, rng=rng)

            # Step 4: Compute percentiles (95% credible interval)
            lower, median, upper = [np.percentile(count_samples, p, axis=0) for p in [2.5,50,97.5]]
            quantile_lists.append([lower,median,upper])
    return quantile_lists


from torch.distributions import constraints
from pyro.distributions import TorchDistribution
from pyro.distributions.util import broadcast_shape

class RobustGeneralizedPoisson(TorchDistribution):
    """ A combination of a GenPoisson and a Uniform distribution."""
    arg_constraints = {
        "mu": constraints.positive,
        "alpha": constraints.real,
        "uniform_max": constraints.integer_interval(0, 1000),
        "corruption_prob": constraints.unit_interval,
    }
    support = constraints.nonnegative_integer
    has_rsample = False

    def __init__(self, mu, alpha, uniform_max, corruption_prob, validate_args=None):
        self.mu = mu
        self.alpha = alpha
        self.uniform_max = uniform_max
        self.corruption_prob = corruption_prob

        self.gp_dist = GeneralizedPoisson(mu, alpha)
        self.uniform_log_prob = torch.log(
            torch.tensor(1.0 / (uniform_max + 1), device=mu.device)
        )

        batch_shape = broadcast_shape(mu.shape, alpha.shape)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        log_gp = self.gp_dist.log_prob(value)
        log_uniform = self.uniform_log_prob.expand_as(log_gp)

        mix_log_prob = torch.logsumexp(torch.stack([
            torch.log1p(-self.corruption_prob) + log_gp,
            torch.log(self.corruption_prob) + log_uniform
        ], dim=0), dim=0)

        return mix_log_prob

    def sample(self, sample_shape=torch.Size()):
        # Naive sample from mixture
        bern = torch.bernoulli(self.corruption_prob.expand(self.batch_shape))
        use_uniform = bern.bool()

        y_gp = self.gp_dist.sample(sample_shape)
        y_uniform = torch.randint(
            low=0, high=self.uniform_max + 1,
            size=sample_shape + self.batch_shape,
            device=self.mu.device
        )

        return torch.where(use_uniform, y_uniform, y_gp)

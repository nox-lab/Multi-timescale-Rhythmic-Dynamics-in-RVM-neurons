# pyright: reportOperatorIssue=false

import pickle
import pymc as pm
import arviz as az
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Protocol, cast
import matplotlib.pyplot as plt
from rvm_analysis.plotting import BrokenAxes
from rvm_analysis.utils import get_cell_colour
from pymc import math as pmmath

class PriorInferenceData(Protocol):
    prior: xr.Dataset


class PosteriorInferenceData(Protocol):
    posterior: xr.Dataset


def compute_predictions_over_new_data(model,idata,X_test):
    """
    Computes model predictions over new input data, storing the result as count_pred_new in idata_predictions.
    """
    with model:

        pm.set_data({"X": X_test})
        rate = model["rate"]

        # Define new Poisson node for predictive count samples
        if "count_pred_new" not in model.named_vars:
            count_pred_new = pm.Poisson("count_pred_new", mu=rate)


        idata_predictions = pm.sample_posterior_predictive(
            idata,var_names=["rate",
                            "count_pred_new"],predictions=True)
    return idata_predictions


def create_X_and_Y_training_data(counts,skip=1,plot=True):
    """
    Creates the X and Y training data by taking the count at each time,
    optionally skipping times for testing purposes.
    """
    X_data = counts['time'][::skip]
    Y_data = counts['count'][::skip]

    if plot:
        bax =BrokenAxes(width_ratios=(1,1))

        bax.scatter(X_data,Y_data,alpha=0.3)
        plt.show()
    
    return X_data,Y_data


def store_model_predictions(save_path: Path,data):
    """
    Stores the data stored in ```data```, which should be a dictionary, by pickling to ```save_path```. Originally, data was:
        ```data = {
        "counts":counts ,
        "X_test":X_test ,
        "count_pred_mean":count_pred_mean ,
        "lambda_preds":lambda_preds ,
        "lambda_mean":lambda_mean ,
        "preds_numpy":preds_numpy,
        "t_switch": idata.posterior['t_switch']

        }```
    """
    file = open(save_path, 'wb')

    # dump information to that file
    pickle.dump(data, file)

    # close the file
    file.close()


def plot_prior_predictive_checks(model,X_test,n_sample_paths=10):
    """Calculates and plots prior predictive checks on the rate."""
    with model:
        pm.set_data({"X": X_test})
        idata_prior = cast(PriorInferenceData,pm.sample_prior_predictive(samples=n_sample_paths,var_names=['rate']))
        
    y = idata_prior.prior['rate'].mean(("chain"))

    _, ax = plt.subplots()

    ax.plot(X_test, y.T, c="k", alpha=0.4)

    ax.set_xlabel("Predictor")
    ax.set_ylabel("Mean Outcome")
    ax.set_title("Prior predictive checks")
    plt.show()


def build_double_exponential_on_model(X_data,Y_data,p):
    with pm.Model() as on_model:

        # The Xdata
        X = pm.MutableData("X",X_data)


        # Priors
        t_switch = pm.Uniform("t_switch", lower=p["PRIORS"]["T_LOWER"],upper=p["PRIORS"]["T_UPPER"])
        log_base_rate_pre = pm.Normal('log_base_rate_pre',
                                    mu=p["PRIORS"]["LOG_BASE_RATE_PRE"]["MU"],
                                    sigma=p["PRIORS"]["LOG_BASE_RATE_PRE"]["SIGMA"])  # prior for baseline rate
        log_base_rate_post = pm.Normal('log_base_rate_post',
                                    mu=p["PRIORS"]["LOG_BASE_RATE_POST"]["MU"],
                                    sigma=p["PRIORS"]["LOG_BASE_RATE_POST"]["SIGMA"])  # prior for baseline rate
        

        # beta_1_pre = pm.HalfNormal('beta_1_pre', sigma=p["PRIORS"]["SIGMA_BASE_RATE"])  # prior for coefficients
        beta_1_pre = pm.LogNormal('beta_1_pre', mu=np.log(0.1),sigma=0.5)  # prior for coefficients
        # beta_1_post = -1 * pm.HalfNormal('beta_1_post', sigma=p["PRIORS"]["SIGMA_BASE_RATE"])  # prior for coefficients
        # beta_slow_post = -1 * pm.HalfNormal('beta_slow_post', sigma=p["PRIORS"]["SIGMA_BASE_RATE_POST"])  # prior for coefficients
        beta_1_post = -pm.LogNormal(
            "beta_1_post",
            mu=np.log(0.1),      # center around 0.1
            sigma=0.5            # fairly wide but not insane
        )

        beta_slow_post = -pm.LogNormal(
            "beta_slow_post",
            mu=np.log(1/50),    # center around 0.005
            sigma=0.5            # tighter so it doesn't overlap much with fast
        )

        log_k_pre = pm.Normal('log_k_pre',
                            mu=p["PRIORS"]["LOG_K_PRE"]["MU"],
                            sigma=p["PRIORS"]["LOG_K_PRE"]["SIGMA"])
        log_k_post = pm.Normal('log_k_post',
                            mu=p["PRIORS"]["LOG_K_POST"]["MU"],
                            sigma=p["PRIORS"]["LOG_K_POST"]["SIGMA"])

        
        k_slow = pm.Deterministic("k_slow", pmmath.exp(log_k_pre) + pmmath.exp(log_base_rate_pre) - pmmath.exp(log_base_rate_post) - pmmath.exp(log_k_post))
        
        slow_rate = k_slow * pmmath.exp(pmmath.dot(X - t_switch,beta_slow_post))

        # Rate function
        condition = X <= t_switch
        rate = pm.Deterministic("rate",pmmath.switch(
            condition,
            pmmath.exp(log_base_rate_pre) + pmmath.exp(log_k_pre) * pmmath.exp(pmmath.dot(X - t_switch,beta_1_pre)),
            slow_rate + pmmath.exp(log_base_rate_post) + pmmath.exp(log_k_post) * pmmath.exp(pmmath.dot(X - t_switch,beta_1_post))
        ))

        # Likelihood
        counts_obs = pm.Poisson("counts_obs", mu=rate, observed=Y_data)

    return on_model

def build_double_exponential_on_model_reversed_sampling(X_data,Y_data,p):
    with pm.Model() as on_model:

        # The Xdata
        X = pm.MutableData("X",X_data)


        # Priors
        t_switch = pm.Uniform("t_switch", lower=p["PRIORS"]["T_LOWER"],upper=p["PRIORS"]["T_UPPER"])
        log_base_rate_post = pm.Normal('log_base_rate_post',
                                    mu=p["PRIORS"]["LOG_BASE_RATE_POST"]["MU"],
                                    sigma=p["PRIORS"]["LOG_BASE_RATE_POST"]["SIGMA"])  # prior for baseline rate
        

        beta_1_pre = pm.HalfNormal('beta_1_pre', sigma=p["PRIORS"]["SIGMA_BASE_RATE"])  # prior for coefficients
        # beta_1_post = -1 * pm.HalfNormal('beta_1_post', sigma=p["PRIORS"]["SIGMA_BASE_RATE"])  # prior for coefficients
        # beta_slow_post = -1 * pm.HalfNormal('beta_slow_post', sigma=p["PRIORS"]["SIGMA_BASE_RATE"]/4)  # prior for coefficients
        beta_1_post = -pm.LogNormal(
            "beta_1_post",
            mu=np.log(1),      # center around 0.1
            sigma=0.5            # fairly wide but not insane
        )

        beta_slow_post = -pm.LogNormal(
            "beta_slow_post",
            mu=np.log(1/200),    # center around 0.005
            sigma=0.3            # tighter so it doesn't overlap much with fast
        )

        log_k_post = pm.Normal('log_k_post',
                            mu=p["PRIORS"]["LOG_K_PRE"]["MU"],
                            sigma=p["PRIORS"]["LOG_K_PRE"]["SIGMA"])
        
        log_k_slow = pm.Normal('log_k_slow',
                            mu=p["PRIORS"]["LOG_K_PRE"]["MU"],
                            sigma=p["PRIORS"]["LOG_K_PRE"]["SIGMA"])

        log_k_pre = pm.Normal('log_k_pre',
                            mu=p["PRIORS"]["LOG_K_PRE"]["MU"],
                            sigma=p["PRIORS"]["LOG_K_PRE"]["SIGMA"])
        
        slow_rate = pmmath.exp(log_k_slow) * pmmath.exp(pmmath.dot(X - t_switch,beta_slow_post))

        # Using a stick breaking construction for the first two coefficients.
        # We want k_pre + base_rate_pre =  pmmath.exp(log_k_slow) + pmmath.exp(log_base_rate_post) + pmmath.exp(log_k_post)
        # So we sample uniformly from 0 to 1, and set the two rates equal to that.
        split_var = pm.Beta('split_var',alpha=0.5,beta=5)
        peak_height = pm.Deterministic('peak_height',pmmath.exp(log_k_slow) + pmmath.exp(log_base_rate_post) + pmmath.exp(log_k_post))

        base_rate_pre = pm.Deterministic("base_rate_pre",peak_height * split_var)

        log_base_rate_pre = pm.Deterministic("log_base_rate_pre",pmmath.log(base_rate_pre))

        k_pre = pm.Deterministic("k_pre",peak_height - base_rate_pre)

        # log_base_rate_pre = pm.Normal('log_base_rate_pre',
        #                             mu=p["PRIORS"]["LOG_BASE_RATE_PRE"]["MU"],
        #                             sigma=p["PRIORS"]["LOG_BASE_RATE_PRE"]["SIGMA"])  # prior for baseline rate
        # k_pre = pm.Deterministic("k_pre", pmmath.exp(log_k_slow) + pmmath.exp(log_base_rate_post) - pmmath.exp(log_base_rate_pre) + pmmath.exp(log_k_post))

        # Rate function
        condition = X <= t_switch
        rate = pm.Deterministic("rate",pmmath.switch(
            condition,
            base_rate_pre + k_pre * pmmath.exp(pmmath.dot(X - t_switch,beta_1_pre)),
            slow_rate + pmmath.exp(log_base_rate_post) + pmmath.exp(log_k_post) * pmmath.exp(pmmath.dot(X - t_switch,beta_1_post))
        ))

        # Likelihood
        counts_obs = pm.Poisson("counts_obs", mu=rate, observed=Y_data)

    return on_model



def build_single_exponential_on_model(X_data,Y_data,p):
    with pm.Model() as on_model:

        # The Xdata
        X = pm.MutableData("X",X_data)


        # Priors
        t_switch = pm.Uniform("t_switch", lower=p["PRIORS"]["T_LOWER"],upper=p["PRIORS"]["T_UPPER"])
        log_base_rate_pre = pm.Normal('log_base_rate_pre',
                                    mu=p["PRIORS"]["LOG_BASE_RATE_PRE"]["MU"],
                                    sigma=p["PRIORS"]["LOG_BASE_RATE_PRE"]["SIGMA"])  # prior for baseline rate
        log_base_rate_post = pm.Normal('log_base_rate_post',
                                    mu=p["PRIORS"]["LOG_BASE_RATE_POST"]["MU"],
                                    sigma=p["PRIORS"]["LOG_BASE_RATE_POST"]["SIGMA"])  # prior for baseline rate
        

        # beta_1_pre = pm.HalfNormal('beta_1_pre', sigma=p["PRIORS"]["SIGMA_BASE_RATE"])  # prior for coefficients
        # beta_1_post = -1 * pm.HalfNormal('beta_1_post', sigma=p["PRIORS"]["SIGMA_BASE_RATE"])  # prior for coefficients

        # beta_1_pre = pm.HalfNormal('beta_1_pre', sigma=p["PRIORS"]["SIGMA_BASE_RATE"])  # prior for coefficients
        beta_1_pre = pm.LogNormal('beta_1_pre', mu=np.log(0.1),sigma=0.5)  # prior for coefficients
        # beta_1_post = -1 * pm.HalfNormal('beta_1_post', sigma=p["PRIORS"]["SIGMA_BASE_RATE"])  # prior for coefficients
        # beta_slow_post = -1 * pm.HalfNormal('beta_slow_post', sigma=p["PRIORS"]["SIGMA_BASE_RATE_POST"])  # prior for coefficients
        # beta_1_post = -pm.LogNormal(
        #     "beta_1_post",
        #     mu=np.log(0.1),      # center around 0.1
        #     sigma=0.5            # fairly wide but not insane
        # )

        beta_1_post = -pm.LogNormal(
            "beta_1_post",
            mu=np.log(1/50),    # center around 0.005
            sigma=0.5            # tighter so it doesn't overlap much with fast
        )
        log_k_pre = pm.Normal('log_k_pre',
                            mu=p["PRIORS"]["LOG_K_PRE"]["MU"],
                            sigma=p["PRIORS"]["LOG_K_PRE"]["SIGMA"])
        k_post = pm.Deterministic('k_post',
                                    pmmath.exp(log_base_rate_pre) + pmmath.exp(log_k_pre) - pmmath.exp(log_base_rate_post)
                                    )
        
        # Rate function
        condition = X <= t_switch
        rate = pm.Deterministic("rate",pmmath.switch(
            condition,
            pmmath.exp(log_base_rate_pre) + pmmath.exp(log_k_pre) * pmmath.exp(pmmath.dot(X - t_switch,beta_1_pre)),
            pmmath.exp(log_base_rate_post) + k_post * pmmath.exp(pmmath.dot(X - t_switch,beta_1_post))
        ))

        # Likelihood
        counts_obs = pm.Poisson("counts_obs", mu=rate, observed=Y_data)
    return on_model


def build_linear_slope_recovery_off_model(X_data,Y_data,p):
    with pm.Model() as off_model:

        X = pm.MutableData("X", X_data)

        # Priors
        b = pm.HalfNormal("b", sigma=p["PRIORS"]["SIGMA_B"])   # Lower sigmoid asymptote
        a = pm.Deterministic("a",b+pm.HalfNormal("a_bar", sigma=p["PRIORS"]["SIGMA_A"]))   # Final asymptote
        c = pm.Deterministic("c",b+pm.HalfNormal("c_bar", sigma=p["PRIORS"]["SIGMA_A"]))   # Final asymptote
        log_k = pm.Normal("log_k", mu=0, sigma=p["PRIORS"]["K"])
        log_k2 = pm.Normal("log_k2", mu=0,sigma=p["PRIORS"]["K2"])     
        k3 = pm.HalfNormal("k3", sigma=p["PRIORS"]["K3"])  
        t = pm.Uniform("t_switch", lower=p["PRIORS"]["T_LOWER"],upper=p["PRIORS"]["T_UPPER"])
        u0 = pm.Uniform("u0",lower=b,upper=a)

        # Calculate x0 to ensure sigmoid passes through (t, u0)
        x0 = t - (1 / pmmath.exp(log_k)) * pmmath.log((a - b) / (u0 - b) - 1)

        # Left side: decreasing sigmoid
        sigmoid_part = b + (a - b) / (1 + pmmath.exp(pmmath.exp(log_k) * (X - x0)))

        # Right side: exponential increasing to a
        exp_part = u0 + (c - u0) * (1 - pmmath.exp(-pmmath.exp(log_k2) * (X - t))) + k3 * (X-t)

        # Piecewise function
        rate = pm.Deterministic("rate",pmmath.switch(X < t, sigmoid_part, exp_part))

        # Observation model
        obs = pm.Poisson('obs', mu=rate, observed=Y_data)

    return off_model


def build_single_exponential_off_model(X_data,Y_data,p):
    with pm.Model() as off_model:

        X = pm.MutableData("X", X_data)

        # Priors
        b = pm.HalfNormal("b", sigma=p["PRIORS"]["SIGMA_B"])   # Lower sigmoid asymptote
        a = pm.Deterministic("a",b+pm.HalfNormal("a_bar", sigma=p["PRIORS"]["SIGMA_A"]))   # Final asymptote
        k = pm.Normal("k", mu=p["PRIORS"]["K"]["MU"], sigma=p["PRIORS"]["K"]["SIGMA"])       # Steepness
        k2 = pm.Normal("k2", mu=p["PRIORS"]["K2"]["MU"],sigma=p["PRIORS"]["K2"]["SIGMA"])       # Steepness
        t = pm.Uniform("t_switch", lower=p["PRIORS"]["T_LOWER"],upper=p["PRIORS"]["T_UPPER"])
        u0 = pm.Uniform("u0",lower=b,upper=a)

        # Calculate x0 to ensure sigmoid passes through (t, u0)
        x0 = t - (1 / pmmath.exp(k)) * pmmath.log((a - b) / (u0 - b) - 1)

        # Left side: decreasing sigmoid
        sigmoid_part = b + (a - b) / (1 + pmmath.exp(pmmath.exp(k) * (X - x0)))

        # Right side: exponential increasing to a
        exp_part = u0 + (a - u0) * (1 - pmmath.exp(-pmmath.exp(k2) * (X - t)))

        # Piecewise function
        rate = pm.Deterministic("rate",pmmath.switch(X < t, sigmoid_part, exp_part))

        # Observation model
        obs = pm.Poisson('obs', mu=rate, observed=Y_data)

        return off_model


def build_double_exponential_off_model(X_data,Y_data,p):
    with pm.Model() as off_model:

        X = pm.MutableData("X", X_data)

        # Priors
        b = pm.HalfNormal("b", sigma=p["PRIORS"]["SIGMA_B"])   # Lower sigmoid asymptote
        a = pm.Deterministic("a",b+pm.HalfNormal("a_bar", sigma=p["PRIORS"]["SIGMA_A"]))   # Final asymptote
        k = pm.Normal("log_k", mu=p["PRIORS"]["K"]["MU"], sigma=p["PRIORS"]["K"]["SIGMA"])       # Steepness
        k2 = pm.Normal("log_k2", mu=p["PRIORS"]["K2"]["MU"],sigma=p["PRIORS"]["K2"]["SIGMA"])       # Steepness
        k3 = pm.Normal("log_k3", mu=p["PRIORS"]["K3"]["MU"],sigma=p["PRIORS"]["K3"]["SIGMA"])       # Steepness
        t = pm.Uniform("t_switch", lower=p["PRIORS"]["T_LOWER"],upper=p["PRIORS"]["T_UPPER"])
        u0 = pm.Uniform("u0",lower=b,upper=a)
        c = pm.Deterministic("c",u0+pm.HalfNormal("c_bar", sigma=p["PRIORS"]["SIGMA_A"]))   # Final asymptote
        d = pm.Deterministic("d",u0+pm.HalfNormal("d_bar", sigma=p["PRIORS"]["SIGMA_A"]))   # Final asymptote

        # Calculate x0 to ensure sigmoid passes through (t, u0)
        x0 = t - (1 / pmmath.exp(k)) * pmmath.log((a - b) / (u0 - b) - 1) 

        # Left side: decreasing sigmoid
        sigmoid_part = b + (a - b) / (1 + pmmath.exp(pmmath.exp(k) * (X - x0)))

        # Right side: exponentials increasing to (c-u0) + (d-u0) = c + d - 2u0
        exp_part_1 = (c - u0) * (1 - pmmath.exp(-pmmath.exp(k2) * (X - t)))
        exp_part_2 = (d - u0) * (1 - pmmath.exp(-pmmath.exp(k3) * (X - t)))

        # Piecewise function
        rate = pm.Deterministic("rate",pmmath.switch(X < t, sigmoid_part, u0 + exp_part_1+exp_part_2))

        # Observation model
        obs = pm.Poisson('obs', mu=rate, observed=Y_data)
        return off_model



def build_erf_off_model(X_data,Y_data,p):
    with pm.Model() as off_model:

        X = pm.MutableData("X", X_data)

        # Priors
        erfoffset = pm.HalfNormal("erfoffset", sigma=10)
        erfscale = pm.HalfNormal("erfscale",10)
        erfslope = pm.HalfNormal("erfslope",4)
        k2 = pm.Normal("log_k2", mu=p["PRIORS"]["K2"]["MU"],sigma=p["PRIORS"]["K2"]["SIGMA"])       # Steepness
        k3 = pm.Normal("log_k3", mu=p["PRIORS"]["K3"]["MU"],sigma=p["PRIORS"]["K3"]["SIGMA"])       # Steepness
        t = pm.Uniform("t_switch", lower=-4,upper=4)
        u0 = pm.Uniform("u0", lower=erfoffset, upper=erfoffset+2*erfscale)
        c = pm.Deterministic("c",u0+pm.HalfNormal("c_bar", sigma=p["PRIORS"]["SIGMA_A"]))   # Final asymptote
        d = pm.Deterministic("d",u0+pm.HalfNormal("d_bar", sigma=p["PRIORS"]["SIGMA_A"]))   # Final asymptote

        # Calculate x0 to ensure sigmoid passes through (t, u0)
        # x0 = t - (1 / pmmath.exp(k)) * pmmath.log((a - b) / (u0 - b) - 1) 
        x0 = t - pmmath.erfinv(1 - (u0 - erfoffset) / erfscale) / erfslope
        #x0  = τ=t0​−k1​erf−1(1+k1​k2​−u0​​)
        # Left side: decreasing sigmoid
        # sigmoid_part = b + (a - b) / (1 + pmmath.exp(pmmath.exp(k) * (X - x0)))

        erf = erfoffset + erfscale * (1- pmmath.erf(erfslope*(X-x0)))

        # Right side: exponentials increasing to (c-u0) + (d-u0) = c + d - 2u0
        exp_part_1 = (c - u0) * (1 - pmmath.exp(-pmmath.exp(k2) * (X - t)))
        exp_part_2 = (d - u0) * (1 - pmmath.exp(-pmmath.exp(k3) * (X - t)))

        # Piecewise function
        rate = pm.Deterministic("rate",pmmath.switch(X < t, erf, u0 + exp_part_1+exp_part_2))

        # Observation model
        obs = pm.Poisson('obs', mu=rate, observed=Y_data)
        return off_model

def build_linear_erf_off_model(X_data,Y_data,p):
    with pm.Model() as off_model:

        X = pm.MutableData("X", X_data)

        # Priors
        erfoffset = pm.HalfNormal("erfoffset", sigma=10)
        erfscale = pm.HalfNormal("erfscale",10)
        erfslope = pm.HalfNormal("erfslope",4)
        k2 = pm.Normal("log_k2", mu=p["PRIORS"]["K2"]["MU"],sigma=p["PRIORS"]["K2"]["SIGMA"])       # Steepness
        k3 = pm.HalfNormal("k3", sigma=p["PRIORS"]["K3"])  
        t = pm.Uniform("t_switch", lower=-4,upper=4)
        u0 = pm.Uniform("u0", lower=erfoffset, upper=erfoffset+2*erfscale)
        c = pm.Deterministic("c",u0+pm.HalfNormal("c_bar", sigma=p["PRIORS"]["SIGMA_A"]))   # Final asymptote
        # d = pm.Deterministic("d",u0+pm.HalfNormal("d_bar", sigma=p["PRIORS"]["SIGMA_A"]))   # Final asymptote

        # Calculate x0 to ensure sigmoid passes through (t, u0)
        # x0 = t - (1 / pmmath.exp(k)) * pmmath.log((a - b) / (u0 - b) - 1) 
        x0 = t - pmmath.erfinv(1 - (u0 - erfoffset) / erfscale) / erfslope
        #x0  = τ=t0​−k1​erf−1(1+k1​k2​−u0​​)
        # Left side: decreasing sigmoid
        # sigmoid_part = b + (a - b) / (1 + pmmath.exp(pmmath.exp(k) * (X - x0)))

        erf = erfoffset + erfscale * (1- pmmath.erf(erfslope*(X-x0)))

        # Right side: exponentials increasing to (c-u0) + (d-u0) = c + d - 2u0
        exp_part_1 = (c - u0) * (1 - pmmath.exp(-pmmath.exp(k2) * (X - t)))

        exp_part = u0 + exp_part_1 + k3 * (X-t)


        # Piecewise function
        rate = pm.Deterministic("rate",pmmath.switch(X < t, erf, exp_part))

        # Observation model
        obs = pm.Poisson('obs', mu=rate, observed=Y_data)
        return off_model


def calculate_rate_percentages_of_peak(ds,times=None,percentages=[0.1,0.5,0.9],cell_type='ON'):
    """
    Calculates the location and value of percentages of the relevant base rate (pre or post withdrawal
    for comparison against previous data looking at when the ON cell burst occurs.
    """
    # Ensure ds exists
    assert "rate" in ds, "Dataset must contain 'rate' variable"

    chain_dim = "chain"
    draw_dim = "draw"
    time_index = "rate_dim_2"

    # Prepare rate array as (n_samples, T)
    n_chains = ds.sizes[chain_dim]
    n_draws = ds.sizes[draw_dim]
    T = ds.sizes[time_index]
    n_samples = n_chains * n_draws

    # Stack chain+draw into a single sample dimension, then transpose to (sample, time)
    rate_xr = ds["rate"].stack(sample=(chain_dim, draw_dim)).transpose("sample", time_index)
    rate_np = rate_xr.values  # shape: (n_samples, T)

    if times is not None:
        time_coord = times
    else:
        time_coord = np.arange(T)

    # Find peak indices and thresholds
    peak_idx = np.argmax(rate_np, axis=1)
    base_vals_rise = rate_np[:,0]
    base_vals_fall = rate_np[:,-1]
    peak_vals = rate_np[np.arange(n_samples), peak_idx]
    thresholds_rise = {}
    thresholds_fall = {}
    for percent in percentages:
        thresholds_rise[percent] = percent * (peak_vals - base_vals_rise)
        thresholds_fall[percent] = percent * (peak_vals - base_vals_fall)
    # threshold = 0.9 * peak_vals                         # (n_samples,)

    plt.title("Histogram of peak times")
    plt.hist(time_coord[peak_idx],bins=100)
    plt.show()

    # Plot thresholds as posterior
    fig, ax = plt.subplots(3, 2, figsize=(6, 5),dpi=600)
    plt.suptitle("Posterior of thresholds per percentage")
    for i, percent in enumerate(percentages):
        az.plot_posterior(
            {
                f"rise_{percent}": thresholds_rise[percent] + base_vals_rise,
                f"fall_{percent}": thresholds_fall[percent] + base_vals_fall
            },
            ax=ax[i,:],
            hdi_prob=0.97
        )
    plt.tight_layout()
    plt.show()

    # find first index before peak where rate is greater than threshold 
    idx_grid = np.arange(T)

    rise_times = {}
    fall_times = {}
    for percent in percentages:
        def compute_rise_time_samples(thresholds_rise,percent,base_vals_rise,idx_grid):
            thresholds_percent_rise = thresholds_rise[percent] + base_vals_rise

            # mask for indices before peak
            mask_before_peak = idx_grid[None, :] <= peak_idx[:, None]
            above_thr = rate_np >= thresholds_percent_rise[:, None]
            candidate_rise = above_thr & mask_before_peak

            has_rise = candidate_rise.any(axis=1)
            # argmax gives first True; if no True, argmax returns 0 so mask those out
            first_rise_idx = np.argmax(candidate_rise, axis=1)
            first_rise_idx[~has_rise] = -1 # mark missing as -1
            
            # convert -1 -> np.nan and indices -> actual times
            rise_time = np.full(n_samples, np.nan)

            valid_rise_mask = first_rise_idx >= 0
            rise_time[valid_rise_mask] = time_coord[first_rise_idx[valid_rise_mask]]

            # reshape back to (chain, draw)
            rise_time_cd = rise_time.reshape(n_chains, n_draws)

            return rise_time_cd, rise_time

        def compute_fall_time_samples(thresholds_fall, percent,base_vals_fall,idx_grid):
            thresholds_percent_fall = thresholds_fall[percent] + base_vals_fall
            # find first index after peak where rate <= threshold (fall)
            mask_after_peak = idx_grid[None, :] >= peak_idx[:, None]
            below_thr = rate_np <= thresholds_percent_fall[:, None]
            candidate_fall = below_thr & mask_after_peak

            has_fall = candidate_fall.any(axis=1)
            first_fall_rel_idx = np.argmax(candidate_fall, axis=1) # relative to start of array
            first_fall_rel_idx[~has_fall] = -1

            # absolute fall index = peak_idx + relative index (only when found)
            first_fall_idx = np.full(n_samples, -1, dtype=int)
            valid_fall_mask = first_fall_rel_idx >= 0
            first_fall_idx[valid_fall_mask] =  first_fall_rel_idx[valid_fall_mask]
            first_fall_idx[first_fall_idx >= T] = -1

            # convert -1 -> np.nan and indices -> actual times
            fall_time = np.full(n_samples, np.nan)

            valid_fall_mask = first_fall_idx >= 0
            fall_time[valid_fall_mask] = time_coord[first_fall_idx[valid_fall_mask]]
            fall_time_cd = fall_time.reshape(n_chains, n_draws)

            return fall_time_cd, fall_time

        rise_time_cd, rise_time = compute_rise_time_samples(thresholds_rise,percent,base_vals_rise,idx_grid)
        fall_time_cd, fall_time = compute_fall_time_samples(thresholds_fall,percent,base_vals_fall,idx_grid)


        rise_times[percent] = rise_time
        fall_times[percent] = fall_time

        # InferenceData with these derived posterior variables
        idata_times = cast(PosteriorInferenceData,az.from_dict(
            posterior={
                f"Time to rise to {percent*100}%": rise_time_cd,
                f"Time to fall to {percent*100}%": fall_time_cd
            }
        ))

        # mean, median, and HDI
        print("Summary (mean, median, 95% HDI):\n")
        summary = az.summary(idata_times, hdi_prob=0.97)
        print(summary)

        # numpy summary:
        mean_rise = np.nanmean(rise_time)
        median_rise = np.nanmedian(rise_time)
        hdi_rise = az.hdi(rise_time[~np.isnan(rise_time)], hdi_prob=0.97)

        mean_fall = np.nanmean(fall_time)
        median_fall = np.nanmedian(fall_time)
        hdi_fall = az.hdi(fall_time[~np.isnan(fall_time)], hdi_prob=0.97)

        print("\nExplicit numeric results:")
        print(f"Rise to {percent}% - mean: {mean_rise:.3f}, median: {median_rise:.3f}, 95% HDI: [{hdi_rise[0]:.3f}, {hdi_rise[1]:.3f}]")
        print(f"Fall to {percent}% - mean: {mean_fall:.3f}, median: {median_fall:.3f}, 95% HDI: [{hdi_fall[0]:.3f}, {hdi_fall[1]:.3f}]")

        # Show full posterior distribution of times
        fig, axes = plt.subplots(1, 2, figsize=(9, 2.5),dpi=300)
        for name, ax in zip([f"Time to rise to {percent*100}%", f"Time to fall to {percent*100}%"], axes.flatten()[:2]):
            values = idata_times.posterior[name].values.flatten()

            # Compute mask correctly with parentheses
            range_mask = (time_coord >= np.nanmin(values) - 0.1) & (time_coord <= np.nanmax(values) + 0.1)
            az.plot_posterior(
                idata_times, var_names=name, kind='hist', bins=time_coord[range_mask],
                ax=ax, hdi_prob=0.97, textsize=10,
                round_to=2 if name == f"Time to rise to {percent*100}%" else 4,color=get_cell_colour(names=cell_type)
            )
            ax.set_xlabel("Time (s)",fontsize=10)



def calculate_rate_percentages_of_peak_OFF(ds,times=None,percentages=[0.1,0.5,0.9],cell_type='OFF'):
    """
    Calculates the location and value of percentages of the relevant base rate (pre or post withdrawal
    for comparison against previous data looking at when the OFF cell pause occurs.
    """
    # Ensure ds exists
    assert "rate" in ds, "Dataset must contain 'rate' variable"

    chain_dim = "chain"
    draw_dim = "draw"
    time_index = "rate_dim_2"

    # Prepare rate array as (n_samples, T)
    n_chains = ds.sizes[chain_dim]
    n_draws = ds.sizes[draw_dim]
    T = ds.sizes[time_index]
    n_samples = n_chains * n_draws

    # Stack chain+draw into a single sample dimension, then transpose to (sample, time)
    rate_xr = ds["rate"].stack(sample=(chain_dim, draw_dim)).transpose("sample", time_index)
    rate_np = rate_xr.values

    if times is not None:
        time_coord = times
    else:
        # numeric indices if no explicit time coordinate
        time_coord = np.arange(T)

    # Find dip indices and thresholds
    dip_idx = np.argmin(rate_np, axis=1)
    base_vals_pre_nadir = rate_np[:,0]
    base_vals_post_nadir = rate_np[:,-1]
    nadir_vals = rate_np[np.arange(n_samples), dip_idx]
    threshold_dip = {}
    threshold_recovery = {}
    for percent in percentages:
        threshold_dip[percent] = percent * (base_vals_pre_nadir -nadir_vals)
        threshold_recovery[percent] = percent * (base_vals_post_nadir -nadir_vals)

    plt.title("Histogram of nadir times")
    plt.hist(time_coord[dip_idx],bins=100)
    plt.show()

    # Plot thresholds as posterior
    fig, ax = plt.subplots(3, 2, figsize=(6, 5))
    plt.suptitle("Posterior of thresholds per percentage")
    for i, percent in enumerate(percentages):
        az.plot_posterior(
            {
                f"rise_{percent}": threshold_dip[percent] + base_vals_pre_nadir,
                f"fall_{percent}": threshold_recovery[percent] + base_vals_post_nadir
            },
            ax=ax[i,:],
            hdi_prob=0.97
        )
    plt.tight_layout()
    plt.show()

    # Find first index before peak where rate is greater than threshold
    idx_grid = np.arange(T)

    rise_times = {}
    fall_times = {}
    for percent in percentages:
        def compute_rise_time_samples(thresholds_rise,percent,base_vals_rise,idx_grid):
            thresholds_percent_rise = base_vals_rise - thresholds_rise[percent]

            # mask for indices before (and including) peak
            mask_before_peak = idx_grid[None, :] >= dip_idx[:, None]
            above_thr = rate_np >= thresholds_percent_rise[:, None]
            candidate_rise = above_thr & mask_before_peak

            has_rise = candidate_rise.any(axis=1)
            # argmax gives first True; if no True, argmax returns 0 so we must mask those out
            first_rise_idx = np.argmax(candidate_rise, axis=1)
            first_rise_idx[~has_rise] = -1
            
            rise_time = np.full(n_samples, np.nan)

            valid_rise_mask = first_rise_idx >= 0
            rise_time[valid_rise_mask] = time_coord[first_rise_idx[valid_rise_mask]]

            rise_time_cd = rise_time.reshape(n_chains, n_draws)

            return rise_time_cd, rise_time

        def compute_fall_time_samples(thresholds_fall, percent,base_vals_fall,idx_grid):
            thresholds_percent_fall = base_vals_fall - thresholds_fall[percent]
            # 4) Vectorized: find first index after (including) peak where rate <= threshold (fall)
            mask_pre_nadir = idx_grid[None, :] <= dip_idx[:, None]
            below_thr = rate_np <= thresholds_percent_fall[:, None]
            candidate_fall = below_thr & mask_pre_nadir

            has_fall = candidate_fall.any(axis=1)
            first_fall_rel_idx = np.argmax(candidate_fall, axis=1)
            first_fall_rel_idx[~has_fall] = -1

            # Absolute fall index = peak_idx + relative index (only when found)
            first_fall_idx = np.full(n_samples, -1, dtype=int)
            valid_fall_mask = first_fall_rel_idx >= 0
            first_fall_idx[valid_fall_mask] =  first_fall_rel_idx[valid_fall_mask]
            first_fall_idx[first_fall_idx >= T] = -1

            # Convert -1 -> np.nan and indices -> actual times
            fall_time = np.full(n_samples, np.nan)


            valid_fall_mask = first_fall_idx >= 0
            fall_time[valid_fall_mask] = time_coord[first_fall_idx[valid_fall_mask]]

            # Reshape back to (chain, draw) so arviz can consume it easily
            fall_time_cd = fall_time.reshape(n_chains, n_draws)

            return fall_time_cd, fall_time

        fall_time_cd, fall_time = compute_fall_time_samples(threshold_dip,percent,base_vals_pre_nadir,idx_grid)
        rise_time_cd, rise_time = compute_rise_time_samples(threshold_recovery,percent,base_vals_post_nadir,idx_grid)


        fall_times[percent] = fall_time
        rise_times[percent] = rise_time

        idata_times = cast(PosteriorInferenceData,az.from_dict(
            posterior={
                f"Time to fall by {percent*100}%": fall_time_cd,
                f"Time to rise to {percent*100}%": rise_time_cd,
            }
        ))

        summarize_rise_and_fall_stats(idata_times,rise_time,fall_time,percent,time_coord,hdi_prob=0.97,cell_type=cell_type)


def sample_model(model,X_data,full_save_path:Path,params,sample=False,chains=4,target_accept=0.8):
    """
    Draws samples from the model if sample is true and saves those samples,
    else recovers the idata from a save path.
    """
    with model:
        pm.set_data({"X": X_data})

        if sample:
            idata = pm.sample(draws=params["DRAWS"], tune=params["TUNE"],
                            return_inferencedata=True,chains=chains, target_accept = target_accept
                            )
            if not full_save_path.parent.exists:
                full_save_path.parent.mkdir(exist_ok=False)
            idata.to_netcdf(str(full_save_path))
        else:
            idata = az.from_netcdf(str(full_save_path))
    
    return idata


def compute_and_save_log_likelihood(model,idata,save_path,recompute=False):
    """
    Computes and saves the log likelihood of the model.
    """
    if recompute:
        with model:
            try:
                pm.compute_log_likelihood(idata)
            except Exception as e:
                print(e)

            idata_save = idata.copy()
            idata_save.to_netcdf(save_path)
            on_model_loo = az.loo(idata)
            print(on_model_loo)


def build_mean_model(X_data,Y_data,p):
    """
    Builds the model which just assumes a constant rate.
    """
    with pm.Model() as model:

        X = pm.MutableData("X",X_data)

        base_rate = pm.HalfNormal('base_rate', sigma=p["PRIORS"]["SIGMA_BASE_RATE"])
        constant_factor = pmmath.ones_like(X)
        rate = pm.Deterministic("rate",base_rate*constant_factor)

        # Likelihood
        counts_obs = pm.Poisson("counts_obs", mu=rate, observed=Y_data)

    return model

def summarize_rise_and_fall_stats(idata_times,rise_time,fall_time,percent,time_coord,hdi_prob=0.97,cell_type='OFF'):
    # mean, median, and HDI
    print(f"Summary (mean, median, {hdi_prob*100}% HDI):\n")
    summary = az.summary(idata_times, hdi_prob=hdi_prob)
    print(summary)

    mean_rise = np.nanmean(rise_time)
    median_rise = np.nanmedian(rise_time)
    hdi_rise = az.hdi(rise_time[~np.isnan(rise_time)], hdi_prob=hdi_prob)

    mean_fall = np.nanmean(fall_time)
    median_fall = np.nanmedian(fall_time)
    hdi_fall = az.hdi(fall_time[~np.isnan(fall_time)], hdi_prob=hdi_prob)

    print("\nExplicit numeric results:")
    print(f"Rise to {percent}% - mean: {mean_rise:.3f}, median: {median_rise:.3f}, {hdi_prob*100}% HDI: [{hdi_rise[0]:.3f}, {hdi_rise[1]:.3f}]")
    print(f"Fall to {percent}% - mean: {mean_fall:.3f}, median: {median_fall:.3f}, {hdi_prob*100}% HDI: [{hdi_fall[0]:.3f}, {hdi_fall[1]:.3f}]")

    # Show full posterior distribution of times
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.5),dpi=300)
    for name, ax in zip([f"Time to fall by {percent*100}%",f"Time to rise to {percent*100}%"], axes.flatten()[:2]):
        # Extract posterior values for this variable
        values = idata_times.posterior[name].values.flatten()

        # Compute mask correctly with parentheses
        range_mask = (time_coord >= np.nanmin(values) - 0.1) & (time_coord <= np.nanmax(values) + 0.1)
        az.plot_posterior(
            idata_times, var_names=name, kind='hist', bins=time_coord[range_mask],
            ax=ax, hdi_prob=hdi_prob,textsize=10,
            round_to=2 if name == f"Time to rise to {percent*100}%" else 4 ,color=get_cell_colour(names=cell_type)
        )
        ax.set_xlabel("Time (s)",fontsize=10)

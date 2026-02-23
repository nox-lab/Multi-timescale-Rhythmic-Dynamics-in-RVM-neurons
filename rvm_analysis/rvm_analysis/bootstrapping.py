from deprecated import deprecated
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.stats import norm
import numpy as np

def generate_bootstrap_samples(samples,block_ids,bootstrap_blocks):
    """ Generates a single mean bootstrap estimate of a statistic using block ids."""
    bootstrap_samples = []

    for sample_id in bootstrap_blocks:
        selected_samples = samples[block_ids == sample_id]
        bootstrap_samples.append(selected_samples)
    bootstrap_mean = np.mean(np.concatenate(bootstrap_samples, axis=0),axis=0)
    return bootstrap_mean, bootstrap_samples

def get_maximum_power_peak(psd,freqs,mask):
    """ Gets the index and frequency of the peak of maximum power in a signal."""
    freqs_masked = freqs[mask]
    psd_masked = psd[mask]
    peaks, _ = find_peaks(psd_masked)
    if len(peaks) == 0:
        raise ValueError("No peaks to find max power of")
    peak_idx = peaks[np.argmax(psd_masked[peaks])]
    peak_frequency = freqs_masked[peak_idx]

    return peak_idx, peak_frequency

def summary_psd_stat_no_mean(psd,freqs,plot=False,cutoff_freqs:tuple = (0.0001,0.1)):
    """
    Computes the mean PSD, peak frequency, and fwhm of a set of power
    spectra and returns them.

    NOTE: Computes the mean psd first!
    """
    # Compute the mean PSD
    d_freq = (freqs[1] - freqs[0]).magnitude

    # Find the peaks
    mask = (freqs >= cutoff_freqs[0]) & (freqs <= cutoff_freqs[1])
    try:
        peak_idx, peak_frequency = get_maximum_power_peak(psd,freqs,mask)
    except ValueError as e:
        print("No peak found")
        return psd,None,None,None,None
    widths, width_heights, left_ips, right_ips = peak_widths(psd, [peak_idx], rel_height=0.5)
    width = widths[0]

    # Compute the FWHM
    fwhm = width * d_freq

    # Convert the ips points to frequencies
    left_ips_freq, right_ips_freq = left_ips[0] * d_freq, right_ips[0] * d_freq

    if plot:
        fig = plt.figure(figsize=(10,2))
        plt.plot(freqs,psd)
        plt.xscale("log")
        plt.yscale("log")
        plt.vlines(peak_frequency,0,np.max(psd),label="peak freq")
        plt.axvspan(cutoff_freqs[0],cutoff_freqs[1],0,float(np.max(psd)),label="peak freq",alpha=0.3,color='red')
        plt.show()

    return psd, peak_frequency, fwhm, left_ips_freq, right_ips_freq   

def get_summary_psd_stats(psds,freqs):
    """
    Computes the mean PSD, peak frequency, and fwhm of a set of power
    spectra and returns them.

    NOTE: Computes the mean psd first!
    """
    # Compute the mean PSD
    mean_psd = np.mean(psds, axis=0)

    d_freq = (freqs[1] - freqs[0]).magnitude

    mean_psd, peak_frequency, fwhm, left_ips_freq, right_ips_freq = summary_psd_stat_no_mean(
        mean_psd,freqs
        )

    return mean_psd, peak_frequency, fwhm, left_ips_freq, right_ips_freq


def get_confidence_bounds(values,axis=0,lower_bound=2.5, upper_bound=97.5):
    """ Returns confidence bounds for the array `values`."""
    lower = np.percentile(values, lower_bound,axis=axis)
    upper = np.percentile(values, upper_bound,axis=axis)
    return lower, upper


import numpy as np

def basic_bootstrap_ci(boot_dist, original_stat, alpha=0.05, axis=0):
    """
    Compute a basic percentile bootstrap confidence interval.

    Parameters
    ----------
    boot_dist : np.ndarray
        Bootstrap distribution of statistics (n_bootstraps, n_features).
    original_stat : float
        The statistic computed on the original data (same shape as single bootstrap sample).
    alpha : float
        Significance level, e.g. 0.95
    axis : int
        Axis over which to compute the percentiles.

    Returns
    -------
    ci_lower : np.ndarray
    ci_upper : np.ndarray
        Lower and upper bounds of the basic bootstrap CI.
    """
    boot_dist = np.asarray(boot_dist)
    original_stat = np.asarray(original_stat)

    lower_pct = 100 * (1 - alpha / 2)
    upper_pct = 100 * (alpha / 2)

    lo = np.percentile(boot_dist, lower_pct, axis=axis)
    hi = np.percentile(boot_dist, upper_pct, axis=axis)

    ci_lower = 2 * original_stat - lo
    ci_upper = 2 * original_stat - hi

    return ci_lower, ci_upper



def bca_interval(boot_dist, original_stat, jackknife_stats, alpha=0.05):
    """
    Bias-Corrected and Accelerated (BCa) confidence intervals.

    Parameters
    ----------
    boot_dist : np.ndarray
        Array of bootstrap statistics. Shape (n_bootstraps,) or (n_bootstraps, n_features).
    original_stat : float or np.ndarray
        The statistic computed on the original (non-bootstrapped) data.
    jackknife_stats : np.ndarray
        Array of jackknife statistics. Shape (n_jackknife,) or (n_jackknife, n_features).
    alpha : float
        Significance level.

    Returns
    -------
    ci_lower : np.ndarray
    ci_upper : np.ndarray
        Lower and upper bounds of the BCa confidence interval.
    """
    boot_dist = np.asarray(boot_dist)
    jackknife_stats = np.asarray(jackknife_stats)
    original_stat = np.asarray(original_stat)

    # Ensure shape consistency
    if boot_dist.ndim == 1:
        boot_dist = boot_dist[:, np.newaxis]
    if jackknife_stats.ndim == 1:
        jackknife_stats = jackknife_stats[:, np.newaxis]
    if original_stat.ndim == 0:
        original_stat = np.array([original_stat])

    n_boot = boot_dist.shape[0]
    n_feats = boot_dist.shape[1]

    # Bias correction z0
    z0 = norm.ppf(np.mean(boot_dist < original_stat, axis=0))

    # Acceleration a
    jack_mean = np.mean(jackknife_stats, axis=0)
    numer = np.sum((jack_mean - jackknife_stats) ** 3, axis=0)
    denom = 6.0 * (np.sum((jack_mean - jackknife_stats) ** 2, axis=0) ** 1.5)
    a = numer / denom

    # z-scores for the confidence level
    z_low = norm.ppf(alpha / 2)
    z_high = norm.ppf(1 - alpha / 2)

    # Adjusted percentiles
    pct_low = norm.cdf(z0 + (z0 + z_low) / (1 - a * (z0 + z_low)))
    pct_high = norm.cdf(z0 + (z0 + z_high) / (1 - a * (z0 + z_high)))

    # # Get percentiles from sorted bootstrap distribution
    # sorted_boot = np.sort(boot_dist, axis=0)
    # idx_low = (pct_low * (n_boot - 1)).astype(int)
    # idx_high = (pct_high * (n_boot - 1)).astype(int)

    # # Clip indices in case of rounding issues
    # idx_low = np.clip(idx_low, 0, n_boot - 1)
    # idx_high = np.clip(idx_high, 0, n_boot - 1)

    # # Get CI bounds
    # ci_lower = sorted_boot[idx_low, np.arange(n_feats)]
    # ci_upper = sorted_boot[idx_high, np.arange(n_feats)]

    # Indices for the percentiles
    idx_low = (pct_low * (n_boot - 1)).astype(int)
    idx_high = (pct_high * (n_boot - 1)).astype(int)

    # Clip indices
    idx_low = np.clip(idx_low, 0, n_boot - 1)
    idx_high = np.clip(idx_high, 0, n_boot - 1)

    # Get CI bounds using np.partition to avoid full sort
    ci_lower = np.empty(n_feats)
    ci_upper = np.empty(n_feats)

    for i in range(n_feats):
        col = boot_dist[:, i]
        ci_lower[i] = np.partition(col, idx_low[i])[idx_low[i]]
        ci_upper[i] = np.partition(col, idx_high[i])[idx_high[i]]

    if ci_lower.size == 1:
        return ci_lower.item(), ci_upper.item()
    return ci_lower, ci_upper

@deprecated
def compute_block_bootstrap(spectra: list[tuple],block_ids,color,n_bootstrap = 100):
    """
    DEPRECATED
    Basic bootstrap underestimates the variance due to correlated samples of frequency
    from the same animal. Instead of this, here we resample from clusters instead
    of from cells.
    All PSDS should be normalised before inclusion.

    Args
    ---
    spectra: list of (freq, psd) pairs:
    [(freq, psd1), (freq, psd2), ..., (freq, psdN)]
    freq should be a the same for all psds, so is an irrelevant argument.
    We will use the block_ids list get the block ids
    and do the block bootstrapping. 
    This is because the when we choose a cell type do the frequencies from,
    we lose a few blocks.

    Returns
    ---
    None currently (but plots the estimate)
    """

    # Extract frequency and stack PSDs
    #! Freq should be the same for all psds
    freqs = spectra[0][0]
    psds = np.array([psd for _, psd in spectra])  # shape: (N, num_freqs)

    # Compute mean PSD
    mean_psd = np.mean(psds, axis=0)

    # First we compute the bootstrap block indices
    # We sample N (same number of animals) blocks with replacement
    # Then we take all the psds from that block.
    unique_blocks=np.unique(block_ids)
    N_blocks = len(unique_blocks)
    print("unique_blocks: ", unique_blocks)

    # Bootstrap resampling
    rng = np.random.default_rng(seed=42)  # reproducible
    boot_means = np.zeros((n_bootstrap, len(freqs)))
    sample_Ns = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):

        # Sample with replacement
        sample_indices = rng.integers(0, N_blocks, size=N_blocks)

        #Get the sample psds
        sample_psds = []
        for sample_id in sample_indices:
            psds_for_sample = psds[block_ids == sample_id]
            sample_psds.append(psds_for_sample)
        
        # sample_psds = psds[np.isin(block_ids,sample_indices)]
        sample_psds = np.concatenate(sample_psds, axis=0)
        sample_Ns[i] = len(sample_psds)
        boot_means[i] = np.mean(sample_psds, axis=0)

        #* Find the highest magnitude peak for each spiketrain
        peaks, properties = find_peaks(boot_means[i],height=0)


    # Compute 95% confidence interval
    lower = np.percentile(boot_means, 2.5, axis=0)
    upper = np.percentile(boot_means, 97.5, axis=0)

    peaks, properties  = find_peaks(mean_psd)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, mean_psd, label='Average PSD', color=color)
    plt.scatter(freqs[peaks],mean_psd[peaks],marker='x',color='red')
    plt.fill_between(freqs, lower, upper, color='gray', alpha=0.3, label='95% CI (bootstrap)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized PSD')
    plt.title('Average Normalized PSD with 95% Bootstrap CI')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.show()
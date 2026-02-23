import numpy as np
from scipy.signal.windows import dpss
from scipy.fft import rfft, irfft, rfftfreq

def segment_signal(x, seg_len, step):
    x = np.asarray(x)
    n = len(x)
    starts = np.arange(0, n - seg_len + 1, step)
    print(starts)
    segs = np.stack([x[s:s+seg_len] for s in starts], axis=0)
    return segs

def multitaper_spectra(x_segments, y_segments, fs, NW=3.5, K=None, detrend=True):
    assert x_segments.shape == y_segments.shape
    nseg, seg_len = x_segments.shape
    if K is None:
        K = int(np.floor(2*NW - 1))
        K = max(K, 1)
    tapers, eigs = dpss(seg_len, NW, K, return_ratios=True,sym=False)
    f = rfftfreq(seg_len, d=1/fs)
    Sxx = np.zeros_like(f, dtype=float)
    Syy = np.zeros_like(f, dtype=float)
    Sxy = np.zeros_like(f, dtype=complex)
    if detrend:
        x_segments = x_segments - x_segments.mean(axis=1, keepdims=True)
        y_segments = y_segments - y_segments.mean(axis=1, keepdims=True)
    for i in range(nseg):
        xseg = x_segments[i]
        yseg = y_segments[i]
        for k in range(K):
            tx = xseg * tapers[k]
            ty = yseg * tapers[k]
            X = rfft(tx)
            Y = rfft(ty)
            Sxx += (np.abs(X)**2)
            Syy += (np.abs(Y)**2)
            Sxy += (X * np.conj(Y))
    N_avg = nseg * K
    Sxx /= N_avg
    Syy /= N_avg
    Sxy /= N_avg
    C = (np.abs(Sxy)**2) / (Sxx * Syy + 1e-20)
    C = np.clip(C, 0.0, 1.0)
    return f, Sxx, Syy, Sxy, C, N_avg, K

def theoretical_coherence_threshold(alpha, N_avg):
    if N_avg <= 1:
        return 1.0
    return 1.0 - alpha**(1.0/(N_avg - 1.0))


def fisher_ci_msc(C, N_avg, alpha=0.05):
    from scipy.stats import norm
    C = np.clip(C, 1e-15, 1-1e-15)
    q = np.sqrt(C)
    z = np.arctanh(q)
    if N_avg <= 1:
        se = np.inf
    else:
        se = 1.0/np.sqrt(2.0*(N_avg - 1.0))
    zcrit = norm.ppf(1 - alpha/2.0)
    z_lo = z - zcrit*se
    z_hi = z + zcrit*se
    q_lo = np.tanh(z_lo)
    q_hi = np.tanh(z_hi)
    C_lo = np.clip(q_lo**2, 0.0, 1.0)
    C_hi = np.clip(q_hi**2, 0.0, 1.0)
    return C_lo, C_hi

def phase_randomize_segments(segments, rng):
    nseg, seg_len = segments.shape
    sur = np.zeros_like(segments)
    for i in range(nseg):
        x = segments[i]
        X = rfft(x)
        mag = np.abs(X)
        phases = rng.uniform(0, 2*np.pi, size=mag.shape)
        phases[0] = 0.0
        if seg_len % 2 == 0:
            phases[-1] = 0.0
        Xr = mag * np.exp(1j*phases)
        sur[i] = irfft(Xr, n=seg_len)
    return sur

def phase_randomize_per_segment(y_segments, rng):
    #! Deprecated - identical to above
    nseg, seg_len = y_segments.shape
    y_surr = np.zeros_like(y_segments)
    for i in range(nseg):
        Y = rfft(y_segments[i])
        mag = np.abs(Y)
        phases = rng.uniform(0, 2*np.pi, size=mag.shape)
        phases[0] = 0.0
        if seg_len % 2 == 0:
            phases[-1] = 0.0
        Yr = mag * np.exp(1j*phases)
        y_surr[i] = irfft(Yr, n=seg_len)
    return y_surr


def surrogate_coherence_distribution(x_segments, y_segments, fs, NW, K, n_iter=200, detrend=True, seed=0):
    rng = np.random.default_rng(seed)
    coh_list = []
    for b in range(n_iter):
        y_surr = phase_randomize_segments(y_segments, rng)
        f, Sxx, Syy, Sxy, C, N_avg, K_used = multitaper_spectra(x_segments, y_surr, fs, NW=NW, K=K, detrend=detrend)
        coh_list.append(C)
    return f, np.stack(coh_list, axis=0), N_avg, K_used


def theoretical_fw_threshold(alpha, N_avg, m):
    # alpha: desired familywise error (e.g. 0.05)
    # N_avg: effective avg count used in coherence (your N_avg)
    # m: number of independent frequency bins (assumed)
    alpha_prime = 1 - (1 - alpha)**(1.0/m) # Sidak per-bin alpha
    c_star = 1.0 - alpha_prime**(1.0/(N_avg - 1)) # per-bin coherence threshold
    return c_star


def get_fwer_global_threshold_surrogates(C_surr,alpha):
    max_cohs = C_surr.max(axis=1)        # max over frequencies, per surrogate
    global_thr = np.percentile(max_cohs, 100*(1-alpha))
    return global_thr

def multitaper_coherence_grandavg(x_segments, y_segments, fs, NW=3.5, K=None, detrend=True):
    #! Deprecated - identical to above multitaper coherence estimate.
    assert x_segments.shape == y_segments.shape
    nseg, seg_len = x_segments.shape
    if K is None:
        K = int(np.floor(2*NW - 1))
        K = max(K, 1)
    tapers, eigs = dpss(seg_len, NW, K, return_ratios=True)
    f = rfftfreq(seg_len, d=1/fs)
    nfreq = len(f)
    if detrend:
        x_segments = x_segments - x_segments.mean(axis=1, keepdims=True)
        y_segments = y_segments - y_segments.mean(axis=1, keepdims=True)

    Sxx = np.zeros(nfreq, dtype=float)
    Syy = np.zeros(nfreq, dtype=float)
    Sxy = np.zeros(nfreq, dtype=complex)

    for i in range(nseg):
        for k in range(K):
            X = rfft(x_segments[i] * tapers[k])
            Y = rfft(y_segments[i] * tapers[k])
            Sxx += np.abs(X)**2
            Syy += np.abs(Y)**2
            Sxy += X * np.conj(Y)

    N_avg = nseg * K
    Sxx /= N_avg; Syy /= N_avg; Sxy /= N_avg
    C = (np.abs(Sxy)**2) / (Sxx * Syy + 1e-20)
    C = np.clip(C, 0.0, 1.0)
    return f, Sxx, Syy, Sxy, C, N_avg, K


def surrogate_null_grandavg(x_segments, y_segments, fs, NW=3.5, K=None, detrend=True, n_iter=200, seed=0):
    #! Deprecated - identical to above
    rng = np.random.default_rng(seed)
    Cs = []
    for _ in range(n_iter):
        y_surr = phase_randomize_per_segment(y_segments, rng)
        f, Sxx, Syy, Sxy,C, N_avg, K_used = multitaper_coherence_grandavg(x_segments, y_surr, fs, NW=NW, K=K, detrend=detrend)
        Cs.append(C)
    return f, np.stack(Cs, axis=0), N_avg, K_used
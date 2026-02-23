from elephant.spike_train_surrogates import shuffle_isis
from elephant.conversion import BinnedSpikeTrain
from elephant.spectral import multitaper_coherence
from rvm_analysis.utils import z_score
from quantities import s

def generate_surrogates_and_coherence(spiketrain, spiketrain1, N_surrogates,bin_size,coherence_func):
    """
    
    Uses a multitaper estimate to compute the coherence between a pair for spiketrains.
    """

    spiketrain_surrogates =shuffle_isis(spiketrain,N_surrogates,decimals=None)
    spiketrain1_surrogates =shuffle_isis(spiketrain1,N_surrogates,decimals=None)

    binned_spike_surrogates = BinnedSpikeTrain(
        spiketrain_surrogates,
        bin_size=bin_size,
        )#.to_array()
    binned_spikes1_surrogates = BinnedSpikeTrain(
        spiketrain1_surrogates,
        bin_size=bin_size,
        )#.to_array()

    # binned_spiketrain = binned_spike_surrogates.time_slice(10*s,stop_time-20*s)
    # binned_heartrate = binned_spikes1_surrogates.time_slice(10*s,stop_time-20*s)

    bsa = binned_spike_surrogates.to_array()
    bsa1 = binned_spikes1_surrogates.to_array()

    print(bsa1.shape)
    print(bsa.shape)

    freqs, cohs = [],[]
    for i in range(bsa.shape[0]):
        freq, coh, _ = coherence_func(bsa[i], bsa1[i])
        freqs.append(freq)
        cohs.append(coh)

    return spiketrain_surrogates,spiketrain1_surrogates,bsa,bsa1,freqs,cohs
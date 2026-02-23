"""
Contains the Spike2DataManager class, which has a lot of functions for the
bulk processing of spiketrains. Also has a lot of single functions,
and most of statistics plots used.
! This does need clearing up still.
### Relax - it's not that complicated
All this module does is provide some loose functions for working with
spiketrains, and a single class `Spike2DataManager` which you use like
`dm = Spike2DataManager(blocks)`. Then you can run `dm.{do_smth}` and it does
it on whatever data you just imported.

This provides an easy interface for
working with the spiketrains. Alternatively, to get all the spiketrains
individually, you can run `dm.spiketrain_iterator(cell_type='All')`
which will return a *generator* over all spiketrains.
"""
from spike2neo.CEDio_updated import CedIO

from copy import deepcopy
import os
import numpy as np
from typing import Callable, Optional

import neo
from neo.core import SpikeTrain, Event, Block, Segment

import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

from scipy.signal import correlate
from sklearn.decomposition import FastICA
from cycler import cycler
from pathlib import Path
import pandas as pd

import quantities as pq
from quantities import s, ms, Hz
from elephant.kernels import GaussianKernel
from elephant.statistics import mean_firing_rate, cv, isi, time_histogram, instantaneous_rate
from elephant.conversion import BinnedSpikeTrain
from elephant.spectral import welch_psd
from viziphant.statistics import plot_time_histogram

from rvm_analysis.save_tools import save_in_folder
from rvm_analysis.plotting_tools import create_spiketrain_label, hist_grid, add_grad_legends
from rvm_analysis.colours import on_off_colour_generator, get_cell_colour


def get_previous_event(i, combined_events,ignore_heat=True):
    last_event=  combined_events[i-1] if i> 0 else (0*s,"start")
    if ignore_heat:
        return last_event if last_event[1] != "heat" else get_previous_event(
            i-1,combined_events,ignore_heat=ignore_heat
            )
    else:
        return last_event
            
def get_next_event(i, combined_events,t_stop, ignore_flick=True):
    next_event=  combined_events[i+1] if i< len(combined_events) - 1 else (t_stop,"end")
    if ignore_flick:
        return next_event if next_event[1] != "flick" else get_next_event(
            i+1,combined_events,t_stop,ignore_flick=ignore_flick
            )
    else:
        return next_event

def get_file_name(object):
    """ Gets the file name of a neo object via the file origin property."""
    try:
        print(object.file_origin)
        file_name = Path(object.file_origin).stem
        return file_name
    except AttributeError as e:
        print(e, "Object passed did not have attribute `file_origin`.")
    except Exception as e:
        return f"Unknown {e}"
    
def get_aligned_time_series(times, ts1, ts2, plot=False):

    if plot:
        plt.plot(times, ts1)
        plt.plot(times, ts2)
        plt.show()

    # Calculate cross-correlation
    cross_corr = correlate(ts1 - np.mean(ts1), ts2 - np.mean(ts2), mode="full")

    # Calculate lags
    lags = np.arange(-len(ts1) + 1, len(ts1))

    # Find the lag with the maximum cross-correlation
    optimal_lag = lags[np.argmax(cross_corr)]

    # Align the time series by shifting ts2
    if optimal_lag > 0:
        ts1_aligned = ts1[optimal_lag:]
        ts2_aligned = ts2[: len(ts1) - optimal_lag]
    else:
        ts1_aligned = ts1[: len(ts2) + optimal_lag]
        ts2_aligned = ts2[-optimal_lag:]

    # Plot the aligned time series
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(ts1_aligned, label="Time Series 1 (Aligned)")
        plt.plot(ts2_aligned, label="Time Series 2 (Aligned)")
        plt.legend()
        plt.title("Aligned Time Series")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()
    return ts1_aligned, ts2_aligned, optimal_lag


def split_neutral_cells(dm, groupings_path: str) -> dict[list[neo.SpikeTrain]]:
    """
    Returns the spiketrains for each group, as a list of lists of spiketrains.
    `groupings_path` is a path to a json file with the groups as a dict.
    """
    with open(groupings_path) as f:
        groupings = json.load(f)

    grouped_spiketrains = {}
    spiketrains = list(dm.spiketrain_iterator(cell_type="All"))
    for key, spiketrain_indices in groupings.items():
        single_group = []
        for index in spiketrain_indices:
            single_group.append(spiketrains[index])
        grouped_spiketrains[key] = single_group
    return grouped_spiketrains

    
def get_all_pair_indices(N, duplicates=True):
    """
    Returns all pair indices up to N.
    With replacement if duplicates = True
    """
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))
    return pairs

def perform_ICA(B, components=3):

    B_normed = (B - np.mean(B, axis=0)) / np.std(B, axis=0)
    ica = FastICA(n_components=components)
    S_ = ica.fit_transform(B)  # Reconstruct signals

    # Plot results
    plt.figure(figsize=(25, 10), dpi=200)

    models = [
        B_normed,
        S_,
    ]
    names = ["Mixed signals", "ICA Recovered Signals"]

    for i, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(2, 1, i)
        plt.title(name)
        plt.plot(model, linewidth=0.5)

    plt.tight_layout()
    plt.show()


def plot_isi_spike_time_plot(
    spiketrain, use_index=True, include_rate=False, ax_to_use=None,s=0.5,fontsize=15
):
    isi_s = isi(spiketrain)
    N_subplots = 2 if include_rate else 1
    if not ax_to_use:

        _, ax = plt.subplots(N_subplots, 1, figsize=(25, 5), sharex=True)
        ax = ax if isinstance(ax, np.ndarray) else [ax]
    else:
        ax = [ax_to_use]
    N = len(isi_s)
    X = np.arange(0, N) if use_index else spiketrain.magnitude[1:]
    print(ax)
    ax[0].set_ylabel("ISI length (s)",fontsize=fontsize)
    ax[0].scatter(X, isi_s, s=s, marker="x", color="red")
    if use_index:
        ax[0].set_xlabel("ISI index",fontsize=fontsize)
    if not ax_to_use:
        if N_subplots == 2:
            ax[1].set_xlabel("Time (s)")
            rate = instantaneous_rate(spiketrain, 0.1 * s, kernel=GaussianKernel(1 * s))
            ax[1].plot(rate.times, rate)
    plt.show()


def plot_scatter_for_descriptors_with_groups(
    grouped_spiketrains,
    filename: str,
    desc_1: Callable,
    desc_2: Callable,
    xlabel: str,
    ylabel: str,
    title="",
    log_x=False,
    log_y=False,
    base_path="../Analysis_plots",
    dpi=300,
    figsize=(5, 5),
    markersize=6,
    svg=False,
    save=False, extra_colours=None, markers=None
):
    legend_tags = {}
    _ = plt.figure(figsize=figsize, dpi=dpi)
    for i, (key, spiketrains) in enumerate(grouped_spiketrains.items()):
        if key not in ["neutral","ON","OFF"]:
            colors = lambda spiketrain: extra_colours[key]
        else:
            colors = lambda spiketrain: get_cell_colour(spiketrains=spiketrain)
        descriptors = [
            (
                desc_1(spiketrain),
                desc_2(spiketrain),
                colors(spiketrain),
                spiketrain.name,
            )
            for spiketrain in spiketrains
        ]
        means, cvs, colors, names = zip(*descriptors)
        plt.scatter(means, cvs, color=colors[0], s=markersize,marker=markers[key] if extra_colours else None)
        legend_tags[key] = colors[0]
    if title is not None:
        plt.title(title, fontsize=20, fontweight="bold", y=1.1)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 24
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=name,
            markerfacecolor=colour,
        )
        for (name, colour) in legend_tags.items()
    ]

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    plt.legend(handles=legend_elements, fontsize=10,loc='lower right')
    save_in_folder(filename, basePath=base_path, svg=svg, save=save)


def calculate_serial_correlation_coefficient(spiketrain: neo.SpikeTrain, lag=1):
    """Calculates the autocorrelation function of spiketrain isis with themselves,
    also known as the serial correlation coefficients.
    """
    isi_s = isi(spiketrain)

    mu_squared = np.mean(isi_s) ** 2
    var = np.mean(isi_s**2)
    cross_average = np.mean(isi_s[lag:] * isi_s[0 : len(isi_s) - lag])

    return (cross_average - mu_squared) / (var - mu_squared)


def autocorr(sig, both_sides=False, biased=False, return_coeffs=False, remove_0=True):
    """
    Parameters
    ----------
    Takes a signal, either a list or a numpy array shape (N_times, ).
    Returns the autocorrelation, or autocovariance if you subtract the mean first.
    If return coeffs is `True` then divide by the max value
    first to get correlation coefficients.
    ### This Function matches with the elephant implementation.

    Returns
    -------
    corr,lags
    """
    sig = np.atleast_1d(sig)
    N = sig.shape[0]
    if return_coeffs:
        sig = sig - np.mean(sig)  #! Just in case the mean was not already removed
    corr = correlate(sig, sig, mode="full")
    lags = np.arange(-N + 1, N)

    if biased:
        corr = corr / N
    else:
        corr = corr / (N - np.abs(lags))

    if return_coeffs:
        corr = corr / corr.max()

    if not both_sides:
        if remove_0:
            corr = corr[lags > 0]
            lags = lags[lags > 0]
        else:
            corr = corr[lags >= 0]
            lags = lags[lags >= 0]

    return corr, lags


def get_welch_spectrum(spiketrain, n_segments, bin_size, window="boxcar"):
    """
    This is useful if you just want the welch spectrum from a spiketrain -
    it is better if you want both spectrum and correlation to
    do the binning only once `get_corr_and_spectrum
    """
    binned_spikes = BinnedSpikeTrain(spiketrain, bin_size=bin_size)
    binned_spikes = binned_spikes.to_array().flatten()
    freq, psd2 = welch_psd(
        binned_spikes,
        n_segments,
        fs=1 / bin_size.rescale(s),
        scaling="density",
        window="boxcar",
        return_onesided=False,
    )
    freq = np.fft.fftshift(freq)
    psd = np.fft.fftshift(psd2)
    psd = psd * (1 / bin_size.rescale(s)) ** 2
    return freq, psd


#! Relies on the old version, the fft implementation of the spectrum.
def get_spectrum(spiketrain, bin_size):

    binned_spikes = BinnedSpikeTrain(spiketrain, bin_size=bin_size)
    binned_spikes = binned_spikes.to_array().flatten()
    # Calculate spectral density
    freq, psd = spectral_density(binned_spikes, bin_size)
    return freq, psd


def plot_corr_and_spectrum(
    corr,
    lags,
    freq,
    psd,
    log_corr=True,
    log_freq=False,
    title="Spike Train Autocorrelation and Spectrum",
    figsize=(25, 3),
    highdef=True,
):
    """Quickly plots both autocorrelation function and power spectrum."""
    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=300 if highdef else 100)
    plt.suptitle(title)
    ax[0].plot(lags, corr)
    ax[0].set_xlabel("lag (5ms bins)")
    if log_corr:
        ax[0].set_xscale("log")
    ax[1].plot(freq, psd)
    ax[1].set_xlabel("frequency (Hz)")
    if log_freq:
        ax[1].set_xscale("log")
    plt.show()

from scipy.signal import detrend

def get_corr_and_spectrum(
    spiketrain,
    n_welch_segs=8,
    welch_window="boxcar",
    bin_size=5 * ms,
    biased_autocov=False,
    remove_binned_mean=False,
    remove_0=True,
    both_sides=True,
    return_coeffs=False,
    fft_length=None,linear_detrend=False
):
    """
    Computes the autocovariance (or autocorrelation function if zeroing),
    and the power spectrum of a spiketrain.
    If `use_elephant_histogram` is true then there
    might be a performance increase,
    but no difference in result, as the two are equivalent.

    ## Use this one to get both spectrum and correlation at the same time,
    # efficiently.

    Returns
    -------
    corr, lags, freq, psd

    """
    binned_spikes = BinnedSpikeTrain(spiketrain, bin_size).to_array().flatten()

    if remove_binned_mean or return_coeffs:
        binned_spikes = binned_spikes - np.mean(binned_spikes)

    if linear_detrend:
        binned_spikes = detrend(binned_spikes,type='linear')

    corr, lags = autocorr(
        binned_spikes,
        both_sides=both_sides,
        biased=biased_autocov,
        remove_0=remove_0,
        return_coeffs=return_coeffs,
    )

    # Compute the welch spectrum
    freq, psd = welch_psd(
        binned_spikes,
        n_welch_segs,
        fs=1.0 / bin_size.rescale(s),
        scaling="density",
        window=welch_window,
        return_onesided=False,
        nfft=fft_length,
    )  #! Welch automatically demeans the signal anyway.
    freq = (np.fft.fftshift(freq) * freq.units).rescale(Hz)
    psd = np.fft.fftshift(psd)
    psd = (
        psd * (1.0 / bin_size.rescale(s)) ** 2
    )  # * To correct for the discrete time bin size

    return corr, lags, freq, psd


#! Old version, but equivalent to welch, it does work!!
def spectral_density(binned_spikes, bin_size):
    """
    This does not use welch, but uses the direct fft of the binned spikes.
    Useful as a test comparison to welch, with 1 segment and a boxcar window.
    """
    N = len(binned_spikes)

    binned_spikes = binned_spikes - np.mean(binned_spikes)

    freq = np.fft.fftshift(np.fft.fftfreq(N, d=bin_size))
    spikes_ft = np.fft.fftshift(np.fft.fft(binned_spikes, n=N))
    psd = np.abs(spikes_ft) ** 2 / (
        N * bin_size
    )  # * To correct for the discrete time bin size
    return freq, psd


def get_n_largest_freqs(freqs, psd, n):
    # Combine freqs and psd into a single 2D array
    combined_data = np.column_stack((freqs, psd))

    # Sort the combined data array based on the PSD values (in descending order)
    sorted_data = combined_data[combined_data[:, 1].argsort()[::-1]]

    # Extract the n largest frequencies
    largest_freqs = sorted_data[:n, 0]

    return largest_freqs


def get_auto_correlation_freqs(
    spiketrains: list[neo.SpikeTrain],
    plot=False,
    bin_size=1 * ms,
    n_largest=10,
):
    """
    Returns the spectra, frequencies, and frequency resolutions of each of the
    spiketrains, and plots them.
    """
    spectra = []
    resolutions = []
    freqs = []
    for train in spiketrains:

        desc = str(train.description)
        corr, lags, freq, psd = get_corr_and_spectrum(
            train,
            n_welch_segs=1,
            welch_window="boxcar",
            bin_size=bin_size,
            biased_autocov=False,
            remove_binned_mean=False,
            remove_0=True,
            both_sides=False,
            return_coeffs=True,
        )

        psd = psd.T

        psd = psd.T[freq >= 0]
        freq = freq[freq >= 0]
        res = np.abs(freq[0] - freq[1])
        resolutions.append(res)

        max_freqs = get_n_largest_freqs(freq, psd.T, n_largest)
        freqs.extend(max_freqs)
        spectra.append((freq, psd))

        if plot:
            color = get_cell_colour(train)
            fig, ax = plt.subplots(1, 3, width_ratios=[0.1, 0.1, 0.8], figsize=(25, 2))
            ax[0].plot(freq, psd.T, color=color)
            ax[0].set_title("Power Spectrum")
            ax[0].set_xlabel("Frequency (Hz)")
            ax[0].set_xscale("log")

            ax[1].plot((lags * bin_size).rescale(s), corr, color=color)
            ax[1].set_xlabel("Lag (s)")
            ax[1].set_title("Autocorrelation as a function of Lag")
            ax[1].set_xscale("log")

            ax[2].eventplot(
                [sp.magnitude for sp in train], linewidths=0.05, color=color
            )
            ax[2].set_title(f"spiketrain {desc}, {train.name} ({get_file_name(train)})")
            ax[2].set_xlabel("Time (s)")
            plt.show()
    return spectra, freqs, resolutions


def get_auto_correlation_freqs_with_rate(
    spiketrains: list[neo.SpikeTrain],
    plot=False,
    bin_size=1 * ms,
    n_largest=10,
    dpi=150,
    n_welch_segs=1,
    figsize=(25, 5),
    window="boxcar",
    fft_length=None, linear_detrend=False
):
    """
    Returns the spectra, maximum frequencies, and frequency resolutions of each of the
    spiketrains, and plots them. Spectra contains both the frequencies and the psds
    """
    spectra = []
    resolutions = []
    freqs = []
    for train in spiketrains:

        desc = str(train.description)
        corr, lags, freq, psd = get_corr_and_spectrum(
            train,
            n_welch_segs=n_welch_segs,
            welch_window=window,
            bin_size=bin_size,
            biased_autocov=False,
            remove_binned_mean=False,
            remove_0=True,
            both_sides=False,
            return_coeffs=True,
            fft_length=fft_length,linear_detrend=linear_detrend
        )

        psd = psd.T

        psd = psd.T[freq >= 0]
        freq = freq[freq >= 0]
        res = np.abs(freq[0] - freq[1])
        resolutions.append(res)

        max_freqs = get_n_largest_freqs(freq, psd.T, n_largest)
        freqs.extend(max_freqs)
        spectra.append((freq, psd))

        # print(np.trapz(psd.T,freq),"area under psd")
        psd /= np.trapz(psd.T, freq)
        if plot:
            figure = plt.figure(dpi=dpi, figsize=figsize)
            outer: plt.GridSpec = gridspec.GridSpec(
                2, 3, figure=figure, width_ratios=[0.1, 0.1, 0.8]
            )

            # make nested gridspecs
            # gss1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = outer[0])
            # gss2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = outer[1])
            gss2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2])

            color = get_cell_colour(train)
            ax0 = figure.add_subplot(outer[0])
            ax0.plot(freq, psd.T, color=color)
            ax0.set_title("Power Spectrum")
            ax0.set_xlabel("Frequency (Hz)")
            ax0.set_xscale("log")

            ax1 = figure.add_subplot(outer[1])
            ax1.plot((lags * bin_size).rescale(s), corr, color=color)
            ax1.set_xlabel("Lag (s)")
            ax1.set_title("Autocorrelation as a function of Lag")
            ax1.set_xscale("log")
            ax2 = gss2.subplots(sharex=True)
            ax2[0].eventplot(
                [sp.magnitude for sp in train], linewidths=0.05, color=color
            )
            ax2[0].set_title(
                f"spiketrain {desc}, {train.name} ({get_file_name(train)})"
            )
            rate = instantaneous_rate(train, sampling_period=0.1 * s)
            ax2[1].plot(rate.times, rate)
            ax2[1].set_xlabel("Time (s)")

            plt.show()
    return spectra, freqs, resolutions


def colors_to_simple_names(colors, labels):
    on, off, neutral = 0, 0, 0
    names = []
    for i, c in enumerate(colors):
        if c == "r":
            names.append(f"OFF {off}")
            off += 1
        elif c == "g":
            names.append(f"ON {on}")
            on += 1
        elif c == "b":
            names.append("NEUTRAL")
            neutral += 1
        else:
            names.append(labels[i])
    return names


class Spike2DataManager:
    """
    Properly imports the spike2 data into a neo list of blocks.
    By storing the blocks in a class instance,
    methods can be defined on them and the excel spreadsheet.
    This makes iterating over the trial structure and neo datastructure easier.

    ## Example:

    ```py
    dm = Spike2DataManager('./data','excel.xlsx')
    print(dm.blocks)
    print(dm.experiment_notes)
    dm.plot_blocks(rate_sampling_period=0.1* s)
    ```
    """

    def __init__(self, data_directory: Path, excel_notes_path: Path = None):
        self.data_directory = data_directory
        self.excel_notes_path = excel_notes_path
        # self.blocks: list[neo.Block] = self._read_all_files(data_directory)
        if excel_notes_path is not None:
            self.experiment_notes: pd.DataFrame = pd.read_excel(self.excel_notes_path)
        self._blocks = None
        self._cell_attribute_database: pd.DataFrame = None
        self.heart_rate_channel = ["8", "EKG"]

    @property
    def blocks(self) -> list[neo.Block]:
        if self._blocks is None:
            raise AttributeError("You must first call `read_blocks_into_memory`")
        return self._blocks

    @blocks.setter
    def blocks(self, value):
        self._blocks = value

    def read_blocks_into_memory(self):
        self.blocks = self._read_all_files(self.data_directory)
        for i, spiketrain in enumerate(self.spiketrain_iterator(cell_type="All")):
            spiketrain.description = i

    def update_spike_names(self, column_name,cell_type_column_name, row_cell_labelling=False):
        self._update_spike_names(column_name, row_cell_labelling=row_cell_labelling,cell_type_column_name=cell_type_column_name)

    def update_names_from_list(self, names_list):
        """Updates the names of the spikes using the information from a spreadsheet."""
        for i, block in enumerate(self.blocks):
            segment = block.segments[0]
            labels = names_list[i].split("/")
            for j, spiketrain in enumerate(segment.spiketrains):
                spiketrain.name = labels[j]

    def _read_all_files(self, dirname: str) -> list:
        """
        Reads all of the spike2 (.smrx) files present in the directory
        into a list of neo blocks.
        Uses the api from the neo library and the sonpy library to convert
        the spike2 files to a cedrawio class, then
        then uses the CEDio class from neo to finish the conversion to blocks and
        segments.

        The classes from neo contained several bugs which have been fixed in
        copies of the files `cedrawio_updated` and `CEDio_updated`.
        """
        blocks = []
        for file in os.listdir(dirname):
            #! Need to limit the search to spike2 (.smrx) files
            reader = CedIO(os.path.join(dirname, file))
            blocks.append(reader.read(lazy=False)[0])
        return blocks

    def standardize_event_names(self):
        for block in self.blocks:
            segment = block.segments[0]
            for j, event in enumerate(segment.events):
                if event.name in self.heart_rate_channel:
                    event.name = "EKG"
                elif event.name in ["HeatSt", "heat","Heat"]:
                    event.name = "heat"
                elif event.name in ["flick", "Flick","EMG"]:
                    event.name = "flick"
                elif event.name in ["Keyboard", "keyboard"]:
                    event.name = "keyboard"
                elif event.name in ["Pinch_On"]:
                    event.name = "pinch"
                else:
                    raise ValueError(f"Event name {event.name} is not standardizable")
        print("Standardization Complete")

    def _update_spike_names(self, column_name="Cell type", row_cell_labelling=False,cell_type_column_name="Cell Class"):
        """Updates the names of the spikes using the information from a spreadsheet."""
        unique_id = 0
        id_name_tuples = []
        if not row_cell_labelling:
            for i, block in enumerate(self.blocks):
                self.plot_blocks(slice(i,i+1))
                segment = block.segments[0]
                self.experiment_notes = self.experiment_notes.dropna(axis=1)
                print(self.experiment_notes)
                labels = self.experiment_notes[column_name][i].split("/")
                print(labels)
                print(len(segment.spiketrains))
                for j, spiketrain in enumerate(segment.spiketrains):
                    spiketrain.name = labels[j]
                    spiketrain.description = unique_id
                    id_name_tuples.append((unique_id, spiketrain.name))
                    unique_id += 1
        else:
            experiment_and_cell_type = self.experiment_notes[
                [cell_type_column_name, "Experiment"]
            ].dropna()
            for block in self.blocks:
                segment = block.segments[0]
                experiment_name = Path(segment.file_origin).stem

                print("Experiment:", experiment_name)
                cell_types = experiment_and_cell_type.loc[
                    experiment_and_cell_type["Experiment"] == experiment_name, cell_type_column_name
                ]
                print("Corresponding cell types:", cell_types)

                cell_types = cell_types.iloc[0].split("/")
                for i, spiketrain in enumerate(segment.spiketrains):
                    try:
                        spiketrain.name = cell_types[i].upper()
                    except Exception as e:
                        print(spiketrain.file_origin, experiment_name, cell_types, e)
                    spiketrain.description = unique_id
                    id_name_tuples.append((unique_id, spiketrain.name))
                    unique_id += 1

        self._cell_attribute_database = pd.DataFrame(
            id_name_tuples, columns=["ID", "Cell Type"]
        )

    @property
    def cell_attribute_database(self):
        return self._cell_attribute_database

    @cell_attribute_database.setter
    def cell_attribute_database(self, *args, **kwargs):
        raise AttributeError("Cannot modify directly")

    def count_cells(self):
        """Counts the number of cells of each type."""
        cell_types = []
        N_cells_by_block = []

        for block in self.blocks:
            cell_types_in_block = [spiketrain.name for spiketrain in block.filter(objects=neo.SpikeTrain)]
            cell_types.extend(cell_types_in_block)
            N_cells_by_block.append(len(cell_types_in_block))
        unique_cell_names, counts =  np.unique(cell_types,return_counts=True)
        print(f"{len(self.blocks)} files")
        print(f"{len(cell_types)} cells in total")
        print(f"On average, {np.mean(N_cells_by_block):.1f} cells per block")
        return {t: count for t, count in zip(unique_cell_names, counts)}

        return self._cell_attribute_database["Cell Type"].value_counts().to_dict()

    def set_cell_names(self, name: str = None):
        """Sets all spiketrains to the same name."""
        for cell in self.spiketrain_iterator(cell_type="All"):
            cell.name = name

    def set_file_origin_to_name(self):
        for block in self.blocks:
            block.file_origin = Path(block.name)

    def find_full_spiketrain_from_id(self, description: int) -> neo.SpikeTrain:
        """Takes an index and returns that spiketrain."""
        return list(self.spiketrain_iterator(cell_type="All"))[description]

    def plot_all_isis(
        self,
        log=True,
        cell_type="All",
        mean_centre=False,
        bounds: tuple = None,
        figsize=(25, 10),
    ):
        all_isi_arrays = []
        colors = []
        for i, spiketrain in enumerate(self.spiketrain_iterator(cell_type=cell_type)):
            colors.append(get_cell_colour(spiketrain))
            ISIs = isi(spiketrain)
            if log:
                all_isi_arrays.append(
                    (np.log(ISIs.magnitude), spiketrain.name, spiketrain.description)
                )
            else:
                all_isi_arrays.append(
                    (ISIs.magnitude, spiketrain.name, spiketrain.description)
                )

        mean_centered_ISI_arrays = [
            ((ISI[0] - np.mean(ISI[0])) if mean_centre else ISI[0], ISI[1], ISI[2])
            for ISI in all_isi_arrays
        ]


        hist_grid(
            mean_centered_ISI_arrays,
            len(mean_centered_ISI_arrays),
            "Mean Centered log(ISI) histograms",
            fit_dist=False,
            xlim=bounds if bounds is not None else None,
            figsize=figsize, colors = colors
        )

    def print_blocks(self, blockIndices: slice = None):
        """Prints the details of neo blocks."""
        blockIndices = blockIndices or slice(0, len(self.blocks))

        for block in self.blocks[blockIndices]:
            print(f"{block.file_origin}, Segments: {len(block.segments)}", end=", ")
            for segment in block.segments:
                print(
                    f"event channels: {[event.name for event in segment.events]}",
                    end=", ",
                )
                print(f"spike trains: {len(segment.spiketrains)}")

    def get_block_eventtrain(self, block, rate_sampling_period,count_cells=False,border_correction=False):
        eventtrain_list = []
        linewidths = []
        labels = []
        colors = []
        legend_colors = set()
        segment = block.segments[0]
        rates = []
        heart_rate: neo.Event = None
        for spiketrain in segment.spiketrains:
            # print(spiketrain.duration)
            #! This is t_end - t_start by definition, and t_end is the max event of any
            # in the file, not just the spiketrain, so this needs updating.
            # print(spiketrain[-1])
            # print(spiketrain.sampling_rate) #! This currently returns None
            rates.append(
                instantaneous_rate(spiketrain, sampling_period=rate_sampling_period,border_correction=border_correction)
            )
            eventtrain_list.append(spiketrain.magnitude)
            linewidths.append(0.1)
            color = get_cell_colour(spiketrain)
            colors.append(color)
            legend_colors.add((color, spiketrain.name))
            labels.append(
                create_spiketrain_label(spiketrain, info="name")
            )
        for j, event in enumerate(reversed(segment.events)):
            if event.name in self.heart_rate_channel:
                heart_rate = event
            else:
                eventtrain_list.append(event)
                colors.append("black")
                linewidths.append(1)
                labels.append(event.name)
        if count_cells:
            labels = colors_to_simple_names(colors, labels)
        return (
            eventtrain_list,
            colors,
            linewidths,
            labels,
            legend_colors,
            rates,
            heart_rate,
        )

    def plot_blocks(
        self,
        blockIndices: slice = None,
        rate_sampling_period=0.1 * s,
        show_rates=True,
        show_change_points=True,
        savePath=None,
        save=False,
        show_heart_rate=False,
        show_title=True,
        heart_rate_kernel_width=5 * s, figsize=(25,5), border_correction=False, plot_activity_sum=False,
    ):
        """
        Creates an event plot of blocks, showing the spike trains and stimuli on
        separate levels.
        `show rates`: Bool, whether to add an instantaneous rate plot underneath.
        `show_change_points: Bool, whether to a
        `heart_rate_channels`: list, the names of any heart rate channels.
        """
        blockIndices = blockIndices or slice(0, len(self.blocks))
        for block in self.blocks[blockIndices]:
            (
                eventtrain_list,
                colors,
                linewidths,
                labels,
                legend_colors,
                rates,
                heart_rate,
            ) = self.get_block_eventtrain(
                block,
                rate_sampling_period=rate_sampling_period, border_correction=border_correction
            )

            num_plots = np.sum([plot_activity_sum,show_heart_rate, show_rates or show_change_points, 1])
            fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
            plt.rc("xtick", labelsize=10)  # fontsize of the tick labels
            axes = axes if isinstance(axes, np.ndarray) else [axes]
            axes_gen = (a for a in axes)

            ax = next(axes_gen)
            ax.eventplot(eventtrain_list, linewidths=linewidths, colors=colors)
            ax.set_yticks(np.arange(len(eventtrain_list)), labels, fontsize=15)
            legend_elements = [
                Line2D([0], [0], color=color[0], lw=4, label=color[1])
                for color in legend_colors
            ]
            ax.legend(handles=legend_elements, fontsize=12)
            file = get_file_name(block)

            if show_rates or show_change_points:
                ax = next(axes_gen)
                if show_rates:
                    ax.set_title("Instantaneous Rates", fontsize=15)
                    ax.set_xlabel("Seconds", fontsize=12)
                    for i, rate in enumerate(rates):
                        ax.plot(rate.times, rate, color=colors[i])
                    ax.set_ylabel("Rate (Hz)", fontsize=12)
                if show_change_points:
                    crossings = self.find_all_gradient_crossings(
                        block, sampling_period=rate_sampling_period
                    )
                    ax.eventplot(
                        np.array(crossings) * rate_sampling_period,
                        linelengths=10,
                        color="blue",
                    )
                    legend_element = [
                        Line2D([0], [0], color="blue", lw=2, label="change points")
                    ]
                    ax.legend(handles=legend_element)

            if show_heart_rate and (heart_rate is not None):
                ax = next(axes_gen)
                kernel = GaussianKernel(sigma=heart_rate_kernel_width)
                rate = instantaneous_rate(
                    neo.SpikeTrain(heart_rate, heart_rate[-1], s),
                    0.1 * s,
                    kernel=kernel,
                    border_correction=True,
                )
                ax.plot(rate.times, rate * 60.0)
                fig2, ax2 = plt.subplots(figsize=(25,3))
                ax2.eventplot(heart_rate.magnitude,linewidth=0.1)
            if show_title:
                fig.suptitle(
                    f"Spikes and Events Raster Plot: {file if show_title else ''}",
                    fontsize=20,
                )
            if plot_activity_sum:
                ax = next(axes_gen)
                new_train:neo.SpikeTrain = deepcopy(block.segments[0].spiketrains[0])
                merge_train = new_train.merge(*block.segments[0].spiketrains[1:])
                binned_spikes = BinnedSpikeTrain(merge_train,bin_size=1*s)
                ax.plot(binned_spikes.bin_centers,binned_spikes.to_array().T)
            plt.xlabel("Time (s)", fontsize=15)
            plt.tight_layout()
            if save:
                save_in_folder(f"Raster_Plot_All_Cells_{file}", savePath, save=save)
            else:
                plt.show()
            

    def plot_2_blocks_fancy_figure(
        self,
        block_indicies: list[int],
        rate_sampling_period=0.1 * s,
        show_rates=True,
        show_change_points=True,
        savePath=None,
        save=None,
        heart_rate_channels=None,
        show_heart_rate=False,
        show_title=True,
        start=0 * s,
        end=None,
        linewidths_list=None,
        figsize=(25, 5),
        dpi=300,
        legend=True,
        ticks=None,
        show_keyboard=None,
        show_events=False,
        show_rate_title=False,
        title="",
        plot_type="Trial",
    ):
        """
        Creates an event plot of blocks, showing the spike trains and stimuli on
        separate levels.
        `show rates`: Bool, whether to add an instantaneous rate plot underneath.
        `show_change_points: Bool, whether to a
        `heart_rate_channels`: list, the names of any heart rate channels.
        """

        blocks = [self.blocks[blockIndex] for blockIndex in block_indicies]

        figure = plt.figure(dpi=dpi, figsize=figsize)
        outer: plt.GridSpec = gridspec.GridSpec(
            len(blocks), 1, height_ratios=[1] * len(blocks), figure=figure, hspace=0.4
        )

        # make nested gridspecs
        gss = [
            gridspec.GridSpecFromSubplotSpec(
                2,
                1,
                subplot_spec=outer[i],
                hspace=0,
                wspace=0,
                height_ratios=(0.6, 0.4),
            )
            for i in range(len(blocks))
        ]

        for trial, (gs, block) in enumerate(zip(gss, blocks)):
            (
                eventtrain_list,
                colors,
                linewidths,
                labels,
                legend_colors,
                rates,
                heart_rate_index,
            ) = self.get_block_eventtrain(
                block,
                rate_sampling_period=rate_sampling_period,
            )
            if show_keyboard is not None:
                if not show_keyboard:
                    key_idx = labels.index("keyboard")
                    eventtrain_list.pop(key_idx)
                    colors.pop(key_idx)
                    linewidths.pop(key_idx)
                    labels.pop(key_idx)
            if not show_events:
                for event in ["heat", "flick"]:
                    event_idx = labels.index(event)
                    eventtrain_list.pop(event_idx)
                    colors.pop(event_idx)
                    linewidths.pop(event_idx)
                    labels.pop(event_idx)
            else:
                labels[labels.index("flick")] = "behavior"

            plt.rcParams["xtick.labelsize"] = 24
            plt.rcParams["ytick.labelsize"] = 24

            def plot_eventplot_block(cell, fig, eventtrain_list, diff_colors):
                ax = plt.subplot(cell)

                ax.eventplot(
                    eventtrain_list,
                    linewidths=linewidths if not linewidths_list else linewidths_list,
                    colors=diff_colors,
                )
                ax.tick_params("x", which="both", direction="in")
                ax.set_xticks([])
                # Create Simple Labels for the Figure
                ax.set_yticks(
                    np.arange(len(eventtrain_list)),
                    labels=colors_to_simple_names(colors, labels),
                    fontsize=24,
                )
                # legend_elements = [
                #     Line2D([0], [0], color=color[0], lw=4, label=color[1])
                #     for color in legend_colors
                # ]
                if legend:
                    add_grad_legends(legend_colors, ax, fig, fontsize=20)
                    # ax.legend(handles=legend_elements,fontsize=12)s
                if ticks:
                    ax.xaxis.set_major_locator(
                        MaxNLocator(nbins=ticks, min_n_ticks=ticks)
                    )
                ax.set_xlim(start, end)

            def plot_rates_block(cell, fig, rates, trial):
                ax = plt.subplot(cell)
                ax.tick_params(top=True, which="both", direction="in")

                for i, rate in enumerate(rates):
                    ax.plot(rate.times, rate, color=differentiated_colors[i])
                # ax.set_ylabel("Rate\n(Hz)", fontsize=24, rotation=0)
                ax.yaxis.set_label_coords(x=-0.06, y=0)
                ax.set_xlim(start, end)
                # ax.set_xlabel("Time (s)", fontsize=24)

            cg = on_off_colour_generator()
            differentiated_colors = cg.gen_colors_from_red_green_black(colors)
            plot_eventplot_block(gs[0], figure, eventtrain_list, differentiated_colors)
            plot_rates_block(gs[1], figure, rates, trial=trial)
        if title is not None:
            plt.suptitle(title, fontsize=20, fontweight="bold")
        plt.tight_layout()
        if save:
            save_in_folder(
                f"Spike_rate_plot_{plot_type}", savePath, svg=True, save=save
            )
        else:
            plt.show()

    def plot_blocks_fancy_figure(
        self,
        blockIndices: slice = None,
        rate_sampling_period=0.1 * s,
        show_rates=True,
        show_change_points=True,
        savePath=None,
        save=None,
        heart_rate_channels=None,
        show_heart_rate=False,
        show_title=True,
        start=0 * s,
        end=None,
        linewidths_list=None,
        figsize=(25, 5),
        dpi=300,
        legend=True,
        title=None,
        ticks=None,
        show_keyboard=None,
        show_events=False,
        show_rate_title=False,
        saveTitle="Raster_Plot_All_Cells",
    ):
        """
        Creates an event plot of blocks, showing the spike trains and stimuli on
        separate levels.
        `show rates`: Bool, whether to add an instantaneous rate plot underneath.
        `show_change_points: Bool, whether to a
        `heart_rate_channels`: list, the names of any heart rate channels.
        """

        blockIndices = blockIndices or slice(0, len(self.blocks))
        for block in self.blocks[blockIndices]:
            (
                eventtrain_list,
                colors,
                linewidths,
                labels,
                legend_colors,
                rates,
                heart_rate_index,
            ) = self.get_block_eventtrain(
                block,
                rate_sampling_period=rate_sampling_period,
            )

            if show_keyboard is not None:
                if not show_keyboard:
                    try:
                        key_idx = labels.index("keyboard")
                        eventtrain_list.pop(key_idx)
                        colors.pop(key_idx)
                        linewidths.pop(key_idx)
                        labels.pop(key_idx)
                    except:
                        key_idx = labels.index("Keyboard")
                        eventtrain_list.pop(key_idx)
                        colors.pop(key_idx)
                        linewidths.pop(key_idx)
                        labels.pop(key_idx)
            if not show_events:
                for event in ["heat", "flick"]:
                    event_idx = labels.index(event)
                    eventtrain_list.pop(event_idx)
                    colors.pop(event_idx)
                    linewidths.pop(event_idx)
                    labels.pop(event_idx)
            else:
                labels[labels.index("flick")] = "Behavior"
                labels[labels.index("heat")] = "Heat"

            num_plots = np.sum([show_heart_rate, show_rates or show_change_points, 1])
            plt.rcParams["xtick.labelsize"] = 24
            plt.rcParams["ytick.labelsize"] = 24
            fig, axes = plt.subplots(
                num_plots, 1, figsize=figsize, sharex=True, dpi=dpi
            )
            axes = axes if isinstance(axes, np.ndarray) else [axes]
            axes_gen = (a for a in axes)

            ax: Axes = next(axes_gen)
            cg = on_off_colour_generator()
            differentiated_colors = cg.gen_colors_from_red_green_black(colors)
            spacing = 1
            ax.eventplot(
                eventtrain_list,
                linewidths=linewidths if not linewidths_list else linewidths_list,
                colors=differentiated_colors,
                linelengths=spacing,
                lineoffsets=spacing,
            )
            ax.set_xlim(start, end)

            ax.tick_params("x", which="both", direction="in")

            # Create Simple Labels for the Figure
            ax.set_yticks(
                np.arange(len(eventtrain_list)) * spacing,
                labels=colors_to_simple_names(colors, labels),
                fontsize=24,
            )
            if legend:
                add_grad_legends(legend_colors, ax, fig, fontsize=15)
            file = get_file_name(block)
            if ticks:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=ticks, min_n_ticks=ticks))
            if show_rates or show_change_points:
                ax = next(axes_gen)
                ax.tick_params(top=True, which="both", direction="in")

                if show_rates:
                    if show_rate_title:
                        ax.set_title("Instantaneous Rates", fontsize=15)

                    ax.set_xlabel("Time (s)", fontsize=24)

                    for i, rate in enumerate(rates):
                        ax.plot(rate.times, rate, color=differentiated_colors[i])
                    ax.set_ylabel("Rate (Hz)", fontsize=24)
                    ax.set_xlim(start, end)

                if show_change_points:
                    crossings = self.find_all_gradient_crossings(
                        block, sampling_period=rate_sampling_period
                    )
                    ax.eventplot(
                        np.array(crossings) * rate_sampling_period,
                        linelengths=15,
                        color="blue",
                    )
                    legend_element = [
                        Line2D([0], [0], color="blue", lw=2, label="change points")
                    ]
                    ax.legend(handles=legend_element, fontsize=15)

            print("heart_rate_index:", heart_rate_index)
            if show_heart_rate and (heart_rate_index is not None):
                ax = next(axes_gen)
                heartrate = block.segments[0].events[heart_rate_index]
                kernel = GaussianKernel(sigma=5 * pq.s)
                rate = instantaneous_rate(
                    neo.SpikeTrain(heartrate, heartrate[-1], s),
                    0.1 * s,
                    kernel=kernel,
                    border_correction=True,
                )
                ax.plot(rate.times, rate)
            if show_title:
                if title is None:
                    plt.suptitle(
                        f"Spikes and Events Raster Plot: {file if show_title else ''}",
                        fontsize=20,
                    )
                else:
                    plt.suptitle(title, fontsize=20)
            plt.xlabel("Time (s)", fontsize=15)
            if end is not None:
                plt.xlim(start, end)
            else:
                plt.xlim(-5, plt.xlim()[1] + 5)
            plt.tight_layout()
            if not show_rate_title:
                plt.subplots_adjust(hspace=0)
            if save:
                save_in_folder(f"{saveTitle}_{file}", savePath, svg=True, save=save)
            else:
                plt.show()

    def plot_scatter_for_descriptors(
        self,
        filename: str,
        desc_1: Callable,
        desc_2: Callable,
        xlabel: str,
        ylabel: str,
        title="",
        log_x=False,
        log_y=False,
        base_path="../Analysis_plots",
        dpi=300,
        figsize=(5, 5),
        cell_type="All",
        markersize=6,
        svg=True,
        save=False,
    ):

        descriptors = [
            (
                desc_1(spiketrain),
                desc_2(spiketrain),
                get_cell_colour(spiketrains=spiketrain),
                spiketrain.name,
            )
            for spiketrain in self.spiketrain_iterator(cell_type=cell_type)
        ]
        desc_1, desc_2, colors, names = zip(*descriptors)

        _ = plt.figure(figsize=figsize, dpi=dpi)
        if title is not None:
            plt.title(title, fontsize=20, fontweight="bold", y=1.1)
        plt.scatter(desc_1, desc_2, c=colors, s=markersize)
        plt.xlabel(xlabel, fontsize=15, fontweight="normal")
        plt.ylabel(ylabel, fontsize=15, fontweight="normal")
        if log_x:
            plt.xscale("log")
        if log_y:
            plt.yscale("log")

        unique_names = set(names)
        print(unique_names)
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=name,
                markerfacecolor=get_cell_colour(names=name),
            )
            for name in unique_names
        ]
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.legend(handles=legend_elements)
        save_in_folder(filename, basePath=base_path, svg=svg, save=save)

    def plot_descriptor_scatter_3d(
        self,
        filename: str,
        desc_1: Callable,
        desc_2: Callable,
        desc_3: Callable,
        xlabel: str,
        ylabel: str,
        zlabel: str,
        title="",
        log_x=False,
        log_y=False,
        base_path="../Analysis_plots",
    ):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection="3d")
        plt.title(title, fontsize=15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        descriptors = []
        descriptors.extend(
            (
                desc_1(spiketrain),
                desc_2(spiketrain),
                desc_3(spiketrain),
                spiketrain.name,
            )
            for spiketrain in self.spiketrain_iterator(cell_type="All")
        )
        desc1, desc2, desc3, types = zip(*descriptors)
        colors = np.where(np.array(types) == "OFF", "r", "g")

        if log_x:
            import matplotlib.ticker as mticker

            ax.scatter(np.log10(desc1), desc2, desc3, c=colors)

            def log_tick_formatter(val, pos=None):
                return "{:.2f}".format(10.0**val)

            ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
            tick_positions = np.linspace(-1, 1, 3, endpoint=True)
            ax.xaxis.set_ticks(
                tick_positions, ["{:.1f}".format(x) for x in 10.0**tick_positions]
            )
        else:
            ax.scatter(desc1, desc2, desc3, c=colors)

        tick_positions_y = np.linspace(0, 20, 3, endpoint=True)
        ax.yaxis.set_ticks(
            tick_positions_y, ["{:.0f}".format(x) for x in tick_positions_y]
        )
        tick_positions_z = np.linspace(0, 300, 4, endpoint=True)
        ax.zaxis.set_ticks(
            tick_positions_z, ["{:.0f}".format(x) for x in tick_positions_z]
        )
        # legend_elements = [
        #     Line2D([0], [0], marker="o", color="w", label="ON", markerfacecolor="g"),
        #     Line2D([0], [0], marker="o", color="w", label="OFF", markerfacecolor="r"),
        # ]

        # plt.legend(handles=legend_elements)
        from matplotlib import rcParams

        rcParams["axes.labelpad"] = 8
        ax.view_init(30, 30, 0.0)
        ax.set_box_aspect(None, zoom=0.7)
        plt.tight_layout()

    def plot_spiketrains(self, spiketrains: neo.SpikeTrain):
        _ = plt.figure(figsize=(25, 5))
        if isinstance(spiketrains, list):
            if isinstance(spiketrains[0], neo.SpikeTrain):
                plt.eventplot(
                    [train.magnitude for train in spiketrains], linewidths=0.1
                )
            else:
                plt.eventplot(spiketrains, linewidths=0.1)
        elif isinstance(spiketrains, neo.SpikeTrain):
            plt.eventplot(spiketrains.magnitude, linewidths=0.1)
        else:
            plt.eventplot(spiketrains, linewidths=0.1)
        plt.show()

    def get_event_by_name(self, spiketrain, event_name) -> neo.Event:
        for event in spiketrain.segment.events:
            if event.name == event_name:
                return event
        raise ValueError("No such event name for this segment.")

    def get_files(self,cell_type='All'):
        files = []
        for spiketrain in self.spiketrain_iterator(cell_type=cell_type):
            files.append(get_file_name(spiketrain))
        return files

    def get_event_windows(
        self, cell_type="All", zero=True, size=30 * 60 * s,event_centre=True,return_blocks = False
    ) -> list[neo.SpikeTrain]:
        """
        Finds the biggest distances between flick and heat in a trial, and
        then takes size minutes prior to that heat.
        Returns the spiketrain around those times.
        Raises an error if the cut overlaps with an event.
        """
        cut_spiketrains = []
        gap_lengths = []
        block_id = -1
        file_origin_curr = None
        for spiketrain in self.spiketrain_iterator(cell_type=cell_type):
            heats = self.get_event_by_name(spiketrain, "heat").tolist()
            heats.append(spiketrain.t_stop)

            times_between_heats = np.diff(heats)
            index = int(np.argmax(times_between_heats))
            event_time = heats[index + 1]
            gap_lengths.append(times_between_heats[index])
            if heats[index + 1] - size < heats[index]:
                raise ValueError(
                    "This cut length is overlapping the previous heat trial."
                )
            cut_spikes = self.cut_out_spikes(
                spiketrain, event_time, -size, 0 * s, zero=zero,event_centre=event_centre
            )
            if return_blocks:
                if spiketrain.file_origin != file_origin_curr:
                    file_origin_curr = spiketrain.file_origin
                    spiketrain: neo.SpikeTrain = spiketrain
                    block_id +=1
                cut_spikes.annotate(block=block_id)
            cut_spiketrains.append(cut_spikes)
        print(f"min gap: {np.min(gap_lengths)}, spiketrain {np.argmin(gap_lengths)}")
        return cut_spiketrains

    def plot_blocks_single_fig(
        self,
        blockIndices: slice = None,
        rate_sampling_period=0.1 * s,
        show_rates=True,
        save=None,
    ):
        """
        Creates an event plot of blocks,
        showing the spike trains and stimuli on separate levels.
        `show rates`: Bool, whether to add an instantaneous rate plot underneath.
        `show_change_points: Bool, whether to a
        """
        blockIndices = blockIndices or slice(0, len(self.blocks))
        fig, ax = plt.subplots(len(self.blocks) * 2, 1, figsize=(25, 50), sharex=True)
        for i, block in enumerate(self.blocks[blockIndices]):
            eventtrain_list, colors, linewidths, labels, legend_colors, rates = (
                self.get_block_eventtrain(
                    block, rate_sampling_period=rate_sampling_period
                )
            )

            ax[2 * i].eventplot(eventtrain_list, linewidths=linewidths, colors=colors)
            ax[2 * i].set_yticks(np.arange(len(eventtrain_list)), labels, fontsize=15)
            legend_elements = [
                Line2D([0], [0], color=color[0], lw=4, label=color[1])
                for color in legend_colors
            ]
            ax[2 * i].legend(handles=legend_elements)
            file_name = Path(block.file_origin).stem
            ax[2 * i].set_title(f"{file_name}", fontsize=15)

            if show_rates:
                ax[2 * i + 1].set_xlabel("Seconds", fontsize=12)
                for j, rate in enumerate(rates):
                    ax[2 * i + 1].plot(rate.times, rate, color=colors[j])
                ax[2 * i + 1].set_ylabel("Rate (Hz)", fontsize=12)

        plt.suptitle("Spikes and Events Raster Plot for All Cells", fontsize=25)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        if save:
            save_in_folder("Raster_Plot_All_Cells_Single", save)
        else:
            plt.show()

    def spiketrain_iterator(
        self,
        cell_type="All",
        blockIndices: slice = None,
        blockList: list[neo.Block] = None,
    ):
        """Iterates over all blocks specified, returning the spiketrains.
        (We can always access the block via spiketrain.block)"""
        if blockList:
            blocks = blockList
        else:
            blocks = self.blocks[blockIndices or slice(0, len(self.blocks))]

        for block_id, block in enumerate(blocks):
            for spiketrain in block.filter(
                objects=neo.SpikeTrain
            ):  # segment.spiketrains:
                spiketrain.annotate(block_id = block_id)
                if cell_type == "All":
                    yield spiketrain
                elif isinstance(cell_type,list):
                    if spiketrain.name in cell_type:
                        yield spiketrain
                else:
                    if cell_type == spiketrain.name:
                        yield spiketrain

    def get_tail_to_heat_lengths(
        self, segment: neo.Segment
    ) -> tuple[list, list, list, list, list, list]:
        """
        Across all events in the segment, returns the delay between heat and tailflick,
        as well as the delay between tailflick and next heat. Also returns the delay
        between successive heat stimuli. Useful for checking whether a specific window
        falls within a single event or multiple. Currently assumes that no flicks occur 
        without heats.

        A better version would combine the heat and flick as a single labelled list in
        time order, then flag any deviations from "heat, flick, heat, flick etc"
        """
        # get the time between the heat and the tail flick

        # * Get heat, flick events by name, avoids recording channels issues
        heats: neo.Event = segment.filter(objects=neo.Event, name="heat")[0]
        flick = segment.filter(objects=neo.Event, name="flick")[0]

        keyboard = segment.filter(objects=neo.Event,name = "Keyboard")
        #! Deal with Keyboard list being empty
        if len(keyboard) > 0:
            keyboard = keyboard [0]
        else:
            keyboard = neo.Event(units=s,name="Keyboard")


        # # Combine the two lists with an identifier for each.
        # combined = [(time, "heat") for time in heats] + [(time, "flick") for time in flick]

        # # Sort the combined list by the first element of each tuple (the time element).
        # combined_sorted = sorted(combined, key=lambda x: x[0])
        # print(combined_sorted)

        def combine_events(heats: neo.Event,flick: neo.Event,keyboard: neo.Event):
            full_list = [(h,"heat") for h in list(heats)]
            full_list.extend([(f,"flick") for f in list(flick)])
            full_list.extend([(k,"keyboard") for k in list(keyboard)])
            full_list.sort(key=lambda x: x[0])

            return full_list
        
        combined_events = combine_events(heats,flick,keyboard)
        heat_gaps = np.diff(heats)

        flicks = []
        heat_flick_delays = []
        tail_to_heat_lengths = []
        for i, event in enumerate(combined_events):
            if event[1] == "heat":
                if (i < len(combined_events) -1) and (combined_events[i+1][1] =="flick"):
                    heat_flick_delays.append(combined_events[i+1][0] - event[0])
                    flicks.append(combined_events[i+1][0])
                else:
                    train_indices = [s.description for s in segment.spiketrains]
                    heat_flick_delays.append(None)
                    print(f"No corresponding flick, missing trains {train_indices}, event {len(heat_flick_delays)}")
                    flicks.append(None)

            if event[1] == "flick":
                if (i < len(combined_events) -1) and (combined_events[i+1][1] == "heat"):
                    tail_to_heat_lengths.append(combined_events[i+1][0] - event[0]) 








        # * Instead of assuming that each heat has a flick, we look for a flick before
        # * the next heat.
        # * If no such flick occurs, set the heat flick delay to be None
        # * This maintains the list length.
        # heat_flick_delays = []
        # flicks = []  # the filled in flicks list, with Nones
        # j = 0  #! Flick index

        # # For each heat time
        # for i, heat_time in enumerate(heats):

        #     # If at last heat index
        #     if i == len(heats) - 1:
        #         if j == len(flick) - 1:
        #             heat_flick_delays.append(flick[j] - heat_time)
        #             flicks.append(flick[j])
        #         else:
        #             heat_flick_delays.append(None)
        #             flicks.append(None)
        #     else:
        #         if heats[i + 1] > flick[j]:
        #             heat_flick_delays.append(flick[j] - heat_time)
        #             flicks.append(flick[j])
        #             j += 1
        #         else:
        #             flicks.append(None)
        #             heat_flick_delays.append(None)

        # tail_to_heat_lengths = []
        # for i, delay in enumerate(heat_flick_delays[:-1]):
        #     if delay is None:
        #         tail_to_heat_lengths.append(None)
        #     else:
        #         tail_to_heat_lengths.append(heat_gap[i] - delay)

        return heat_flick_delays, tail_to_heat_lengths, heat_gaps, flicks, heats, combined_events
    
    def get_all_firing_rate_type_TF_latency_tuples(self,t_start, cell_type, mean_relative=False,flick_relative=False):
        """
        Returns all the TF latencies, both as a combined list, and split by cell.

        ####
        Returns
        ####
        Tuples `(tuple_list, dict[cell_description, TF_tuple])`. 
        - TP_tuple contains `(cell_type, spike_trains, mean_firing_rate, TF_latency)` 
        tuples for all cells in a dataset. *Beware: Considers cells individually*
        If mean relative is true, divides firing rates by the mean of each cell (making zeros
        relative to the mean).
        """
        num_viable_trials = 0
        output_tuples = []
        tuples_by_spiketrain = {}
        for cell_index, spiketrain in enumerate(self.spiketrain_iterator(cell_type=cell_type)):
            # print(i)
            segment: neo.Segment = spiketrain.segment
            heat_flick_delays, tail_to_heat_lengths, heat_gap, flicks, heats, combined_events = self.get_tail_to_heat_lengths(segment)
            heat_flick_delays = np.array(heat_flick_delays)
            mean_flick_delay = np.mean(heat_flick_delays[~(heat_flick_delays == None)])
            for heat_index, heat in enumerate(heats):
                if heat_flick_delays[heat_index] is not None:
                    cut_spikes = self.cut_out_spikes(spiketrain,heat,t_start,0*s)
                    # spiketrain[(heat - t_start < spiketrain) & (spiketrain < heat)]

                    # print(cut_spikes.t_start,cut_spikes.t_stop)
                    # fig = plt.figure(figsize=(5,2))
                    # plt.title(f"{spiketrain.description},{spiketrain.name}")
                    # plt.eventplot(cut_spikes.magnitude,color=get_cell_colour(spiketrain))
                    # plt.show()
                    num_viable_trials +=1
                    mean_rate = mean_firing_rate(cut_spikes)

                    assert(mean_rate == len(cut_spikes)/np.abs(t_start))
                    if mean_relative:
                        mean_rate -= mean_firing_rate(spiketrain)
                    if flick_relative:
                        delay = heat_flick_delays[heat_index] - mean_flick_delay
                    else:
                        delay= heat_flick_delays[heat_index]
                    output_tuples.append([
                        cell_index, 
                        heat_index, 
                        Path(spiketrain.file_origin).stem,
                        spiketrain.description,
                        spiketrain.name,
                        spiketrain,
                        cut_spikes,
                        float(mean_rate),
                        float(delay), heat])
            # tuples_by_spiketrain[j] = spiketrain_tuples
        print(num_viable_trials,"viable trials")

        TF_latencies_df = pd.DataFrame(data=output_tuples,columns=[
            "cell_index","heat_index_in_cell","file_origin","train_index_from_dataset","cell_type","spiketrain",
            "cut_spikes_pre_trial","mean_rate_pre_trial","TF_latency","heat_time"
        ])

        return TF_latencies_df
    
    def apply_function_around_all_events(self,event_type,segment,function,t_start,t_end
    ) -> list:
        """
        Applies a function around an event, for a segment.

        Heat rules:
        - Allow between (prev flick, next heat)

        Flick rules:
        - Allow between (prev flick, next heat)
        """
        function_results: list[neo.SpikeTrain] = []

        heat_flick_delays_single_file, flicks_to_heats, heat_gaps, flicks, heats, combined_events = (
            self.get_tail_to_heat_lengths(segment)
        )

        # Run through the combined events and select the type we need
        for i, (event_time, event_name) in enumerate(combined_events):
            if event_name == event_type:
                ignore_heat = True if event_type == "flick" else False
                previous_event = get_previous_event(i, combined_events,ignore_heat=ignore_heat)
                next_event = get_next_event(i,combined_events,segment.spiketrains[0].t_stop,ignore_flick=not ignore_heat) #! Fix how this is getting t_stop!!!

                if (next_event[0] - event_time < t_end):
                    print(f"Next event [{next_event[1]},{next_event[0]}] cuts off window")
                    continue
                elif (event_time + t_start < previous_event[0]):
                    print(f"Previous event {previous_event} cuts off window")
                    continue 

                #* Apply function around the event time
                func_result = function( #!The function needs to include the event/spiketrain its going to slice
                    event_time, t_start, t_end, zero=False
                )
                if func_result is not None:
                    function_results.append(func_result)
                else:
                    print("End of spiketrain reached")        

        return function_results

    def get_all_event_spike_trains(
        self, train, segment, event_type, t_start, t_end
    ) -> list[neo.SpikeTrain]:
        """
        Gets a list of cut spiketrains around an event.
        Takes a spiketrain and a segment, because we need all the heat and flick
        times to decide whether to allow the cut out or not.

        Heat rules:
        - Allow between (prev flick, next heat)

        Flick rules:
        - Allow between (prev flick, next heat)
        """
        spikes_list: list[neo.SpikeTrain] = []

        heat_flick_delays_single_file, flicks_to_heats, heat_gaps, flicks, heats, combined_events = (
            self.get_tail_to_heat_lengths(segment)
        )


        # Run through the combined events and select the type we need
        for i, (event_time, event_name) in enumerate(combined_events):
            if event_name == event_type:
                ignore_heat = True if event_type == "flick" else False
                previous_event = get_previous_event(i, combined_events,ignore_heat=ignore_heat)
                # print("previous event",previous_event)
                next_event = get_next_event(i,combined_events,train.t_stop,ignore_flick=not ignore_heat)

                if (next_event[0] - event_time < t_end):
                    print(f"Next event [{next_event[1]},{next_event[0]}] cuts off window")
                    continue
                elif (event_time + t_start < previous_event[0]):
                    print(f"Previous event {previous_event} cuts off window")
                    continue 

                # Attempt to cut out a window
                cut_spikes = self.cut_out_spikes(
                    train, event_time, t_start, t_end, zero=False
                )
                if cut_spikes is not None:
                    spikes_list.append(cut_spikes)
                else:
                    print("End of spiketrain reached")
        return spikes_list

    def plot_isi_hist(self, blockIndices: slice, log=False, cutoff: float = None):
        for spiketrain in self.spiketrain_iterator("All", blockIndices):
            fig, axes = plt.subplots(1, figsize=(25, 5))
            if log:
                isi_array = np.log10(np.array(isi(spiketrain)))
            else:
                isi_array = np.array(isi(spiketrain))
            if cutoff:
                isi_array = isi_array[np.where(isi_array < cutoff)]
            axes.hist(isi_array, bins=200)
            axes.set_ylabel("Count")
            # axes[0].boxplot(isi_array,vert=False,flierprops=black_x) # type:axes.Axes
            # plt.delaxes(axes[0])
            plt.suptitle("Interspike Interval Histogram and Boxplot")
            plt.xlabel(f"{'Log10' if log else ''} Interspike Interval (s)")
            plt.show()

    def cut_out_spikes(
        self, train: neo.SpikeTrain, event_time, t_start, t_end, zero=False, event_centre=True
    ) -> neo.SpikeTrain:
        """
        **UNSAFE, USE WITH CARE** 
        This only returns None if the spiketrain t_stop is exceeded.
        It does not check for any other conditions.
        
        Given an event time, attempts to cut out of the train, a window
        between `event_time + t_start` and `event_time + t_end`.
        Returns None if the window exceeds `spiketrain.t_stop`.
        """
        window_start = event_time + t_start
        window_end = event_time + t_end

        if window_end > train.t_stop:
            print("Reached end of spiketrain, missing this event.")
            return None
        # cut_spikes = train[(window_start < train) & (train < window_end)]
        cut_spikes = train.time_slice(window_start, window_end)

        if event_centre:
            cut_spikes = cut_spikes - event_time
            cut_spikes.t_start = t_start
            cut_spikes.t_stop = t_end

        if zero:
            cut_spikes = cut_spikes - t_start
            cut_spikes.t_start = 0 * s
            cut_spikes.t_stop = t_end - t_start

        cut_spikes.description = train.description
        cut_spikes.segment = train.segment

        return cut_spikes
    
    def get_all_cut_heartrates(self,event_type, cell_type, t_start,t_end):
        """Goes through all the animals, and if a cell of cell_type was recorded, gets the heartrate around a trial."""
        cut_heart_beats = []
        for block in self.blocks:
            segment = block.segments[0]
            #* Extract the heart rate and Spike rate
            heart_beats = None
            for event in segment.events:
                if event.name in self.heart_rate_channel:
                    heart_beats: neo.SpikeTrain = neo.SpikeTrain(event,event[-1],s)
            
            def cut_heart_rate_func(heart_beats: neo.SpikeTrain):
                def cut_heart_rate(event_time,t_start,t_end,zero=False):
                    cut_beats:neo.SpikeTrain = heart_beats.time_slice(event_time + t_start,event_time + t_end)
                    zeroed_cut_beats = cut_beats.time_shift(-event_time)
                    return zeroed_cut_beats
                return cut_heart_rate
            
            if heart_beats is not None:
                cut_heart_rate = cut_heart_rate_func(heart_beats)

                cut_rates = self.apply_function_around_all_events(event_type,segment,cut_heart_rate,t_start,t_end
                )

                cut_heart_beats.extend(cut_rates)
        
        return cut_heart_beats
            
                    



    def get_all_cut_spikes(
        self, event_type, cell_type, t_start, t_end
    ) -> tuple[list[neo.SpikeTrain], list[list[neo.SpikeTrain]]]:
        """
        Goes through all the spiketrains in a dataset, along with an event,
        and cuts the spiketrains around the time of that event, returning the cuts
        as a list.

        Returns
        ---
        spikes_list, list_of_list_of_spike_trains_by_cell
        """
        list_of_list_of_spike_trains_by_cell: list[list[neo.SpikeTrain]] = []
        spikes_list: list[neo.SpikeTrain] = []
        max_flicks = 0
        max_heats = 0
        for block in self.blocks:
            segment = block.segments[0]
            flicks = segment.filter(objects=neo.Event, name="flick")
            if len(flicks) == 0:
                raise ValueError("No events called `flick` in this segment. Are your segment events named correctly?")
            max_flicks += len(segment.filter(objects=neo.Event, name="flick")[0]) * (len(segment.filter(objects=neo.SpikeTrain, name=cell_type)))
            max_heats += len(segment.filter(objects=neo.Event, name="heat")[0]) * (len(segment.filter(objects=neo.SpikeTrain, name=cell_type)))

            for train in segment.spiketrains:
                if cell_type == "All" or (train.name == cell_type):
                    # Cut the spike train around the event, return spike train list
                    split_spike_trains_by_event = self.get_all_event_spike_trains(
                        train,
                        segment,
                        event_type=event_type,
                        t_start=t_start,
                        t_end=t_end,
                    )
                    spikes_list.extend(split_spike_trains_by_event)
                    # Keep the spiketrains separate by cell
                    list_of_list_of_spike_trains_by_cell.append(
                        split_spike_trains_by_event
                    )
        print("max possible flicks: ",max_flicks)
        print("Max possible heats: ",max_heats)

        return spikes_list, list_of_list_of_spike_trains_by_cell

    def util_color_by_file_origin(
        self, spiketrain: neo.SpikeTrain, use_file=True
    ) -> tuple:
        """Returns a color depending on the file origin or cell type if use file is
        False of the spike train."""
        key = spiketrain.file_origin if use_file else spiketrain.description
        # Generate a spectrum of shades of blue
        num_shades = 5  # Number of shades to generate
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / num_shades) for i in range(num_shades)]

        # Assign a color based on the hash of the file origin
        hash_val = hash(key)
        index = hash_val % num_shades
        return colors[index]
    

    def plot_spikes_around_event(
        self,
        event: str,
        t_start: pq.Quantity,
        t_end: pq.Quantity,
        cell_type="All",
        sorted_by: Optional[str] = None,
        color_by_file=True,
        save=None,
        figsize=(25, 5),
        dpi=150,
        title=None,
        pre_average_cut=[10 * s, 0 * s], plot=True
    ) -> tuple[list[neo.SpikeTrain], list[list[neo.SpikeTrain]]]:
        """
        Displays Spike timings of OFF or ON cells around either the heat or
        flick events.

        ### Parameters
        `event`: 'heat' or 'flick'. Which event to lock the spike trains to.
        `t_start`: the amount of time before alignment event to use. Should be negative.
        ` t_end`: the amount of time after alignment event to end. Should be positive.
        `sorted_by`: Option to sort the resulting spikes by 'mean' or 'cv'.
        `color_by_file`: Colors the trains by file

        ### Returns
        `spikes_list_sorted: list[neo.SpikeTrain]`, a sorted list of all the spike
        trains which were plotted.
        """

        spikes_list: list[neo.SpikeTrain] = []

        cut_spikes_list, trials = self.get_all_cut_spikes(
            event, cell_type, t_start, t_end
        )
        spikes_list.extend(cut_spikes_list)
        if plot:
            fig, axs = plt.subplots(
                2, 1, figsize=figsize, sharex=True, height_ratios=[0.8, 0.2], dpi=dpi
            )
            if title is not None:
                plt.suptitle(f"{title}", fontsize=20)
            else:
                plt.suptitle(
                    f"The spike timings of {cell_type} cells around the {event} event,"
                    f"window "
                    f"({t_start.magnitude},{t_end.magnitude}) {t_end.dimensionality}",
                    fontsize=20,
                )
            if sorted_by is None:
                spikes_list_sorted = [sp for sp in spikes_list]
            elif sorted_by == "mean":
                spikes_list_sorted = sorted(
                    [sp for sp in spikes_list], key=lambda x: mean_firing_rate(x)
                )
            elif sorted_by == "cv":
                spikes_list_sorted = sorted(
                    [sp for sp in spikes_list], key=lambda x: cv(isi(x))
                )
            elif sorted_by == "pre-firing":

                def mean_pre_firing(spikes):
                    pre_spikes = spikes[
                        (spikes < 0 * s - pre_average_cut[1])
                        & (spikes > 0 * s - pre_average_cut[0])
                    ]
                    # * cut before the behaviour
                    return mean_firing_rate(pre_spikes)

                spikes_list_sorted = sorted(
                    [sp for sp in spikes_list], key=lambda x: mean_pre_firing(x)
                )

            colors = [
                self.util_color_by_file_origin(train, color_by_file)
                for train in spikes_list_sorted
            ]
            axs[0].eventplot(
                [train.magnitude for train in spikes_list_sorted],
                linewidths=0.5,
                color=colors,
            )

            axs[0].set_ylabel("Event Number")
            axs[0].vlines(0, 0, len(spikes_list), label=f"{event}", colors="red")
            if sorted_by == "pre-firing":
                axs[0].vlines(
                    -pre_average_cut[1],
                    0,
                    len(spikes_list),
                    label="cut late",
                    colors="orange",
                )
                axs[0].vlines(
                    -pre_average_cut[0],
                    0,
                    len(spikes_list),
                    label="cut early",
                    colors="green",
                )
            histogram = time_histogram(
                spikes_list, bin_size=0.1 * s, t_start=t_start, t_stop=t_end
            )

            hist:Axes = plot_time_histogram(histogram, axes=axs[1])
            for child in hist.get_children():
                try:
                    child.set_facecolor(get_cell_colour(spikes_list[0])) 
                except:
                    break
            plt.xlabel("Seconds")
            axs[0].legend()
            if save:
                save_in_folder(f"{cell_type}_{event}_raster_plot_PTSH", save)
            else:
                plt.show()
        return spikes_list, trials

    def cut_blocks_to_times(self, times: list[pq.Quantity]):
        """
        Cuts all the blocks to a time per block. Cuts all spike trains
        and events associated with that block. Modifies in place.
        """
        for i, block in enumerate(self.blocks):
            block.segments[0] = block.segments[0].time_slice(
                0 * s, times[i], reset_time=True
            )

    def plot_spikes_around_event_fancy_figure(
        self,
        event: str,
        t_start: pq.Quantity,
        t_end: pq.Quantity,
        cell_type="All",
        sorted_by: str = "mean",
        color_by_file=False,
        save=None,
        figsize=(25, 5),
        dpi=300,
        title=False,
    ) -> tuple[list[neo.SpikeTrain], list[list[neo.SpikeTrain]]]:
        """
        Displays Spike timings of OFF or ON cells around either heat or flick events.

        ### Parameters
        `event`: 'heat' or 'flick'. Which event to lock the spike trains to.
        `t_start`: the amount of time before the alignment event to use.
        Should be negative.
        ` t_end`: the amount of time after the alignment event to end.
        Should be positive.
        `sorted_by`: Option to sort the resulting spikes by 'mean' or 'cv'.
        `color_by_file`: Colors the trains by file

        ### Returns
        `spikes_list_sorted: list[neo.SpikeTrain]`, a sorted list of all the spike
        trains which were plotted.
        """

        spikes_list: list[neo.SpikeTrain] = []

        cut_spikes_list, trials = self.get_all_cut_spikes(
            event, cell_type, t_start, t_end
        )
        spikes_list.extend(cut_spikes_list)
        from cycler import cycler

        color = "r" if cell_type == "OFF" else "g"
        plt.rcParams["axes.prop_cycle"] = cycler(color=color)  # bgrcmyk
        fig, axs = plt.subplots(
            2, 1, figsize=figsize, sharex=True, height_ratios=[0.8, 0.2], dpi=dpi
        )
        if title:
            plt.suptitle(
                f"The spike timings of {cell_type} cells around the {event} event"
                ", window ({t_start.magnitude},{t_end.magnitude})"
                "{t_end.dimensionality}",
                fontsize=20,
            )
        if sorted_by is None:
            spikes_list_sorted = [sp for sp in spikes_list]
        elif sorted_by == "mean":

            spikes_list_sorted = sorted(
                [sp for sp in spikes_list],
                key=lambda x: mean_firing_rate(
                    self.find_full_spiketrain_from_id(x.description)
                ),
            )
        elif sorted_by == "cv":
            spikes_list_sorted = sorted(
                [sp for sp in spikes_list], key=lambda x: cv(isi(x))
            )
        colors = [
            self.util_color_by_file_origin(train, color_by_file)
            for train in spikes_list_sorted
        ]
        axs: list[Axes]
        axs[0].eventplot(
            [train.magnitude for train in spikes_list_sorted],
            linewidths=0.2,
            color=colors,
        )

        axs[0].set_ylabel("Cell Number")
        axs[0].vlines(
            0,
            -0.5,
            len(spikes_list) - 0.5,
            label="behavior" if event == "flick" else event,
            colors="darkblue",
            linewidth=0.8,
        )
        axs[0].set_ylim(-0.5, len(spikes_list))

        histogram = time_histogram(
            spikes_list, bin_size=0.1 * s, t_start=t_start, t_stop=t_end
        )
        hist:Axes = plot_time_histogram(histogram, axes=axs[1])
        for child in hist.get_children():
            try:
                child.set_facecolor(get_cell_colour(spikes_list[0])) 
            except:
                break
        axs[1].vlines(
            0,
            0,
            np.max(histogram) + 0.5,
            label=f"{event}",
            colors="darkblue",
            linewidth=0.8,
            clip_on=False,
        )
        plt.xlabel("Seconds")
        plt.xlim(t_start - 1 * s, t_end + 1 * s)
        axs[1].set_ylim(0, np.max(histogram) + 0.5)
        axs[0].legend()
        plt.subplots_adjust(hspace=0)
        if save:
            save_in_folder(f"{cell_type}_{event}_raster_plot_PTSH", save, svg=True)
        else:
            plt.show()
        return spikes_list_sorted, trials

    def plot_spikes_around_event_fancy_figure_grid(
        self,
        cuts_flick: tuple[pq.Quantity] = (-30 * s, 150 * s),
        cuts_behaviour: tuple[pq.Quantity] = (-30 * s, 50 * s),
        sorted_by: str = "mean",
        color_by_file=False,
        save=None,
        figsize=(25, 5),
        dpi=300,
        title=False,
    ) -> tuple[list[neo.SpikeTrain], list[list[neo.SpikeTrain]]]:

        heat_linecolor = "darkmagenta"
        flick_linecolor = "darkblue"

        figure = plt.figure(dpi=dpi, figsize=figsize)
        outer: plt.GridSpec = gridspec.GridSpec(
            2, 2, height_ratios=[1, 1], figure=figure, hspace=0.15, wspace=0.25
        )

        # make nested gridspecs
        gs1 = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[0], hspace=0, height_ratios=(0.7, 0.3)
        )
        gs2 = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[1], hspace=0, height_ratios=(0.7, 0.3)
        )
        gs3 = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[2], hspace=0, height_ratios=(0.7, 0.3)
        )
        gs4 = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[3], hspace=0, height_ratios=(0.7, 0.3)
        )

        def plot_eventplot(cell, event, cell_type, spikes_list, label):

            ax = plt.subplot(cell)

            color = "r" if cell_type == "OFF" else "g"
            color = get_cell_colour(names=[cell_type])
            plt.rcParams["axes.prop_cycle"] = cycler(color=color)  # bgrcmyk

            # sort by mean
            spikes_list_sorted = sorted(
                [sp for sp in spikes_list],
                key=lambda x: mean_firing_rate(
                    self.find_full_spiketrain_from_id(x.description)
                ),
            )

            colors = [
                self.util_color_by_file_origin(train, color_by_file)
                for train in spikes_list_sorted
            ]

            print(event, cell_type, ": ", len(spikes_list_sorted))
            ax.eventplot(
                [train.magnitude for train in spikes_list_sorted],
                linewidths=0.2,
                color=colors,
            )
            if label is not None:
                ax.text(
                    -0.025,
                    1.15,
                    label,
                    transform=ax.transAxes,
                    fontsize=16,
                    fontweight="bold",
                    va="top",
                    ha="right",
                )
            # ax.set_ylabel("Cell Number")
            ax.vlines(
                0,
                -0.5,
                len(spikes_list) - 0.5,
                label="behavior" if event == "flick" else event,
                colors=heat_linecolor if event == "heat" else flick_linecolor,
                linewidth=0.8,
            )
            ax.set_ylim(-0.5, len(spikes_list))
            if cell_type != "ON":
                ax.set_ylabel("Trial", rotation=90, fontsize=15)
            # else:
            #   ax.set_yticks([])
            ax.set_xticks([])

            #TODO This is a bug, and inherits from the function above.
            #TODO These should be defined so they inherit correctly.
            ax.set_xlim(t_start - 1 * s, t_end)
            ax.tick_params("both", direction="inout")
            ax.yaxis.set_major_locator(
                ticker.MultipleLocator(10)
            )  # Set maximum number of y-axis ticks
            ax.tick_params(axis="y", labelrotation=90)  # ax.legend()

        def plot_rateplot(cell, event, cell_type, spikes_list):

            ax = plt.subplot(cell)
            ax.tick_params("both", direction="inout")
            histogram = time_histogram(
                spikes_list, bin_size=0.1 * s, t_start=t_start, t_stop=t_end
            )
            _ = plot_time_histogram(histogram, axes=ax)
            ax.vlines(
                0,
                0,
                np.max(histogram) + 0.5,
                label=f"{event}",
                colors=heat_linecolor if event == "heat" else flick_linecolor,
                linewidth=0.8,
                clip_on=False,
            )
            if cell_type == "ON":
                ax.set_ylabel("")
            else:
                ax.set_ylabel("Count", rotation=90, fontsize=15)
            # plt.xlabel("Seconds")
            ax.set_xlim(t_start - 1 * s, t_end)
            ax.set_ylim(0, np.max(histogram) + 0.5)

            if event == "heat":
                ax.xaxis.set_major_locator(
                    ticker.MultipleLocator(20)
                )  # Set maximum number of y-axis ticks
            ax.tick_params(axis="y", labelrotation=90)  # ax.legend()

            # if cell_type == 'ON':
            #   ax.set_yticklabels([])
            # if cell_type == 'ON':
            #   ax.set_xticklabels([])
            ax.set_xlabel("")

        # plt.subplots_adjust(hspace=0.0)
        # make outer gridspec

        # Adding big row labels
        figure.text(
            0.0,
            0.66,
            "Behaviour-Aligned",
            va="center",
            ha="center",
            rotation="vertical",
            fontsize=15,
            fontweight="bold",
        )
        figure.text(
            0.0,
            0.26,
            "Stimulus-Aligned",
            va="center",
            ha="center",
            rotation="vertical",
            fontsize=15,
            fontweight="bold",
        )

        figure.text(
            0.30,
            1 - 0.08,
            "OFF-Cells",
            va="center",
            ha="center",
            rotation="horizontal",
            fontsize=15,
            fontweight="bold",
        )
        figure.text(
            0.70,
            1 - 0.08,
            "ON-Cells",
            va="center",
            ha="center",
            rotation="horizontal",
            fontsize=15,
            fontweight="bold",
        )

        gss = [gs1, gs2, gs3, gs4]
        events = ["flick", "flick", "heat", "heat"]
        cell_types = ["OFF", "ON", "OFF", "ON"]
        for index, (gs, event, cell_type) in enumerate(zip(gss, events, cell_types)):
            t_start, t_end = cuts_flick if event == "flick" else cuts_behaviour

            spikes_list: list[neo.SpikeTrain] = []

            cut_spikes_list, trials = self.get_all_cut_spikes(
                event, cell_type, t_start, t_end
            )
            spikes_list.extend(cut_spikes_list)

            plot_eventplot(
                gs[0], event, cell_type, spikes_list, label=None
            )  # chr(ord('A') + index))
            plot_rateplot(gs[1], event, cell_type, spikes_list)

        plt.tight_layout()

        if save:
            save_in_folder("Full_grid_raster_plot_PTSH", save, svg=True)
        else:
            plt.show()

    def plot_spikes_around_event_fancy_figure_grid_cell_rows(
        self,
        t_start: pq.Quantity,
        t_end: pq.Quantity,
        sorted_by: str = "mean",
        color_by_file=False,
        save=None,
        figsize=(25, 5),
        dpi=300,
        axes_label_size=15,
        title=False,
    ) -> tuple[list[neo.SpikeTrain], list[list[neo.SpikeTrain]]]:
        plt.rcParams["xtick.labelsize"] = axes_label_size
        plt.rcParams["ytick.labelsize"] = axes_label_size
        heat_linecolor = "darkmagenta"
        flick_linecolor = "darkblue"

        figure = plt.figure(dpi=dpi, figsize=figsize)
        outer: plt.GridSpec = gridspec.GridSpec(
            2, 2, height_ratios=[1, 1], figure=figure, hspace=0.1, wspace=0.25
        )

        # make nested gridspecs
        gs1 = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[0], hspace=0, height_ratios=(0.7, 0.3)
        )
        gs2 = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[1], hspace=0, height_ratios=(0.7, 0.3)
        )
        gs3 = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[2], hspace=0, height_ratios=(0.7, 0.3)
        )
        gs4 = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[3], hspace=0, height_ratios=(0.7, 0.3)
        )

        def plot_eventplot(cell, event, cell_type, spikes_list, label):

            ax = plt.subplot(cell)

            color = "r" if cell_type == "OFF" else "g"
            plt.rcParams["axes.prop_cycle"] = cycler(color=color)  # bgrcmyk

            # sort by mean
            spikes_list_sorted = sorted(
                [sp for sp in spikes_list],
                key=lambda x: mean_firing_rate(
                    self.find_full_spiketrain_from_id(x.description)
                ),
            )

            colors = [
                self.util_color_by_file_origin(train, color_by_file)
                for train in spikes_list_sorted
            ]

            print(event, cell_type, ": ", len(spikes_list_sorted))
            ax.eventplot(
                [train.magnitude for train in spikes_list_sorted],
                linewidths=0.2,
                color=colors,
            )
            if label is not None:
                ax.text(
                    -0.025,
                    1.15,
                    label,
                    transform=ax.transAxes,
                    fontsize=16,
                    fontweight="bold",
                    va="top",
                    ha="right",
                )

            ax.vlines(
                0,
                -0.5,
                len(spikes_list) - 0.5,
                label="behavior" if event == "flick" else event,
                colors=heat_linecolor if event == "heat" else flick_linecolor,
                linewidth=0.8,
            )
            ax.set_ylim(-0.5, len(spikes_list))
            if event == "heat":
                ax.set_ylabel("Trial", rotation=90, fontsize=15)
            # else:
            #   ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xlim(t_start - 1 * s, t_end)
            ax.tick_params("both", direction="inout")
            ax.yaxis.set_major_locator(
                ticker.MultipleLocator(10)
            )  # Set maximum number of y-axis ticks
            ax.tick_params(axis="y", labelrotation=90)  # ax.legend()

        def plot_rateplot(cell, event, cell_type, spikes_list):

            ax = plt.subplot(cell)
            ax.tick_params("both", direction="inout")
            histogram = time_histogram(
                spikes_list, bin_size=0.1 * s, t_start=t_start, t_stop=t_end
            )
            _ = plot_time_histogram(histogram, axes=ax)
            ax.vlines(
                0,
                0,
                np.max(histogram) + 0.5,
                label=f"{event}",
                colors=heat_linecolor if event == "heat" else flick_linecolor,
                linewidth=0.8,
                clip_on=False,
            )
            if cell_type == "ON":
                ax.set_xticks([])
            if event == "heat":
                ax.set_ylabel("Count", rotation=90, fontsize=15)
            else:
                ax.set_ylabel("")
            # plt.xlabel("Seconds")
            ax.set_xlim(t_start - 1 * s, t_end)
            ax.set_ylim(0, np.max(histogram) + 0.5)

            if event == "heat" and cell_type != "ON":
                ax.xaxis.set_major_locator(
                    ticker.MultipleLocator(20)
                )  # Set maximum number of y-axis ticks
            ax.tick_params(axis="y", labelrotation=90)  # ax.legend()

            if cell_type == "OFF":
                ax.set_xlabel("Time since event (s)", fontsize=15)
                # ax.yaxis.set_major_locator(
                #     ticker.MultipleLocator(25)
                # )  # Set maximum number of y-axis ticks
            else:
                ax.set_xlabel("")

        # Adding big row labels
        locations = [(0.03, 0.66), (0.03, 0.26), (0.3, 1 - 0.08), (0.7, 1 - 0.08)]
        names = ["ON-Cells", "OFF-Cells", "Stimulus-Aligned", "Behaviour-Aligned"]
        rotations = ["vertical", "vertical", "horizontal", "horizontal"]

        for loc, name, rot in zip(locations, names, rotations):
            print(loc)
            figure.text(
                loc[0],
                loc[1],
                name,
                va="center",
                ha="center",
                rotation=rot,
                fontsize=15,
                fontweight="bold",
            )

        gss = [gs1, gs2, gs3, gs4]
        events = ["heat", "flick", "heat", "flick"]
        cell_types = ["ON", "ON", "OFF", "OFF"]
        for index, (gs, event, cell_type) in enumerate(zip(gss, events, cell_types)):
            if event == "flick":
                t_end = 150 * s  # 250 *s
                t_start = -30 * s
            else:
                t_end = 50 * s  # 250 *s
                t_start = -30 * s
            spikes_list: list[neo.SpikeTrain] = []

            cut_spikes_list, trials = self.get_all_cut_spikes(
                event, cell_type, t_start, t_end
            )
            spikes_list.extend(cut_spikes_list)

            plot_eventplot(
                gs[0], event, cell_type, spikes_list, label=None
            )  # chr(ord('A') + index))
            plot_rateplot(gs[1], event, cell_type, spikes_list)

        plt.tight_layout()

        if save:
            save_in_folder("Full_grid_raster_plot_PTSH", save, svg=True)
        else:
            plt.show()

    def get_all_cell_pairs_across_types(
        self, duplicates=True, sampling_period=0.1 * s
    ) -> list:
        """Returns four tuples, [(type1, type2), (train1, train2), (rate1, rate2), (index1, index2)],
        for all possible pairs of cells recorded simultaneously.
        Draws tuples without replacement if duplicates = False.
        """
        blocks =self.blocks
        paired_rates = []
        for block in blocks:
            segment = block.segments[0]
            spiketrains = segment.spiketrains
            N_spiketrains = len(segment.spiketrains)
            if N_spiketrains < 2:
                continue 
            else:
                if duplicates:
                    pair_indices = get_all_pair_indices(N_spiketrains)
                    print(len(pair_indices), " pairs")
                    for indices in pair_indices:
                        train_1 = spiketrains[indices[0]]
                        train_2 = spiketrains[indices[1]]
                        types = [train.name for train in [train_1,train_2]]

                        rate1 = instantaneous_rate(
                                train_1, sampling_period=sampling_period
                            )
                        
                        rate2 = instantaneous_rate(
                                train_2, sampling_period=sampling_period
                            )
                        
                        paired_rates.append([types,(train_1,train_2),(rate1,rate2),(indices[0],indices[1])])
        return paired_rates


    def get_all_cell_pairs(
        self, blockList: list[neo.Block] = None, sampling_period=0.1 * s
    ) -> list:
        """Returns a tuple (on_rate, off_rate),
        for all possible pairs of ON and OFF cells combined in the list blockList.
        """

        blocks = blockList if blockList else self.blocks
        paired_rates = []
        for block in blocks:
            segment = block.segments[0]
            types = [train.name for train in segment.spiketrains]
            on_count = types.count("ON")
            off_count = types.count("OFF")
            if (on_count > 0) and (off_count > 0):
                for i, train in enumerate(segment.spiketrains):
                    for j, train_2 in enumerate(segment.spiketrains):
                        if j > i:
                            if train_2.name != train.name:
                                if train.name == "OFF":
                                    off_train, on_train = train, train_2
                                else:
                                    on_train, off_train = train, train_2
                                off_rate = np.array(
                                    instantaneous_rate(
                                        off_train, sampling_period=sampling_period
                                    )
                                )
                                on_rate = np.array(
                                    instantaneous_rate(
                                        on_train, sampling_period=sampling_period
                                    )
                                )
                                single_file_pair_points = np.array(
                                    list(zip(on_rate, off_rate))
                                )
                                paired_rates.append(single_file_pair_points)
        return paired_rates

    def find_all_gradient_crossings(
        self, block: neo.Block = None, sampling_period=0.1 * s
    ) -> list[int]:
        """
        Returns all the times where the instantaneous rates of an ON and an OFF-cell
        in the same block
        a) cross over
        b) have opposite gradient signs (going in opposite directions).
        This is a simple method of detecting change points.
        """
        block = block if block else self.blocks

        crossings = []
        for pair_points in self.get_all_cell_pairs(
            [block], sampling_period=sampling_period
        ):

            on_rate = pair_points[:, 0].flatten()
            off_rate = pair_points[:, 1].flatten()

            for i in range(len(on_rate) - 1):
                # if (np.sign(grad_on[i]) != np.sign(grad_off[i])):
                if (on_rate[i] < off_rate[i]) and (on_rate[i + 1] > off_rate[i + 1]):
                    crossings.append(i)
                elif on_rate[i] > off_rate[i] and on_rate[i + 1] < off_rate[i + 1]:
                    crossings.append(i)

        return crossings

    def plot_all_cell_pairs(self, path, save=False) -> np.ndarray:
        """
        Iterates through every possible ON/OFF pair of cells, and plots the
        instantaneous rate of the pair in a grid.
        Also plots one grid with all of the ON/OFF rates in the same chart.
        """
        pair_points = []
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        ax = ax.flatten()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        count = 1
        for block in self.blocks:
            segment = block.segments[0]
            types = [train.name for train in segment.spiketrains]
            on_count = types.count("ON")
            off_count = types.count("OFF")
            if (on_count > 0) and (off_count > 0):
                for i, train in enumerate(segment.spiketrains):
                    for j, train_2 in enumerate(segment.spiketrains):
                        if j > i:
                            if train_2.name != train.name:
                                print(train_2.name, train.name)
                                if train.name == "OFF":
                                    off_train, on_train = train, train_2
                                else:
                                    on_train, off_train = train, train_2
                                off_rate = np.array(
                                    instantaneous_rate(
                                        off_train, sampling_period=0.1 * s
                                    )
                                )
                                on_rate = np.array(
                                    instantaneous_rate(
                                        on_train, sampling_period=0.1 * s
                                    )
                                )
                                single_file_pair_points = np.array(
                                    list(zip(on_rate, off_rate))
                                )
                                X_single = single_file_pair_points[:, 0]
                                Y_single = single_file_pair_points[:, 1]
                                ax[count].scatter(
                                    X_single,
                                    Y_single,
                                    marker="x",
                                    s=0.01,
                                    c=colors[count - 1],
                                )
                                ax[0].scatter(X_single, Y_single, marker="x", s=0.01)
                                ax[count].set_xlabel("On Rate")
                                ax[count].set_ylabel("Off Rate")
                                ax[count].set_xlim(0, 30)
                                ax[count].set_ylim(0, 30)
                                file_name = Path(segment.file_origin).stem
                                ax[count].set_title(f"IFR of ON and OFF cell {file_name}")
                                pair_points.extend(list(zip(on_rate, off_rate)))
                                count += 1
        plt.suptitle(
            "All possible pairs of Instantaneous Firing Rates of OFF and ON cells"
        )
        plt.tight_layout()
        save_in_folder("all_cell_pairs", path, save=save)
        return np.array(pair_points)

    def plot_all_cell_pairs_fancy(
        self, path, normalize=False, plots=[2, 3], figsize=(20, 10), save=False
    ) -> np.ndarray:
        """
        Iterates through every possible ON/OFF pair of cells, and plots the
        instantaneous rate of the pair in a grid.
        Also plots one grid with all of the ON/OFF rates in the same chart.
        """
        pair_points = []
        fig, ax = plt.subplots(plots[0], plots[1], figsize=figsize)
        ax = ax.flatten()
        plt.rcParams["xtick.labelsize"] = 15
        plt.rcParams["ytick.labelsize"] = 15
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        count = 0
        for block in self.blocks:
            segment = block.segments[0]
            types = [train.name for train in segment.spiketrains]
            on_count = types.count("ON")
            off_count = types.count("OFF")
            if (on_count > 0) and (off_count > 0):
                for i, train in enumerate(segment.spiketrains):
                    for j, train_2 in enumerate(segment.spiketrains):
                        if j > i:
                            if train_2.name != train.name:
                                print(train_2.name, train.name)
                                if train.name == "OFF":
                                    off_train, on_train = train, train_2
                                else:
                                    on_train, off_train = train, train_2
                                off_rate = np.array(
                                    instantaneous_rate(
                                        off_train, sampling_period=0.1 * s
                                    )
                                )
                                on_rate = np.array(
                                    instantaneous_rate(
                                        on_train, sampling_period=0.1 * s
                                    )
                                )
                                if normalize:
                                    off_rate = (off_rate - np.mean(off_rate)) / np.std(
                                        off_rate
                                    )
                                    on_rate = (on_rate - np.mean(on_rate)) / np.std(
                                        on_rate
                                    )

                                single_file_pair_points = np.array(
                                    list(zip(on_rate, off_rate))
                                )
                                X_single = single_file_pair_points[:, 0]
                                Y_single = single_file_pair_points[:, 1]
                                ax[count].scatter(
                                    X_single,
                                    Y_single,
                                    marker="x",
                                    s=0.01,
                                    c=colors[count - 1],
                                )
                                # ax[0].scatter(X_single, Y_single, marker="x", s=0.01)
                                # ax[count].set_xlabel("On Rate", fontsize=15)
                                # ax[count].set_ylabel("Off Rate", fontsize=15)
                                if not normalize:
                                    ax[count].set_xlim(0, np.max(on_rate))
                                    ax[count].set_ylim(0, np.max(off_rate))

                                # ax[count].set_title(f"IFR of ON and OFF cell {file}")
                                pair_points.extend(list(zip(on_rate, off_rate)))
                                count += 1
        # plt.suptitle(
        #     "All possible pairs of Instantaneous Firing Rates of OFF and ON cells"
        # )
        fig.text(-0.02, 0.4, "On Rates (Hz)", rotation=90, fontsize=20)
        fig.text(0.4, -0.05, "Off Rates (Hz)", fontsize=20)
        plt.tight_layout()
        save_in_folder("all_cell_pairs", path, save=save)
        return np.array(pair_points)

    def get_dominant_signal_frequencies(
        self,
        blockList: list[neo.Block] = None,
        n_freqs=2,
        do_plot=True,
        cell_type="All",
        show_freq_subplot=False,
    ) -> list:
        blocks = blockList if blockList else self.blocks

        freqs_list = []
        for block in blocks:
            if cell_type != "All":
                if cell_type not in [
                    spiketrain.name for spiketrain in block.segments[0].spiketrains
                ]:
                    continue
            plt.figure(figsize=(10, 6))

            for spiketrain in block.segments[0].spiketrains:
                if cell_type == "All":
                    pass
                elif spiketrain.name != cell_type:
                    continue

                samp_period = 0.1 * s
                instant_rate = np.array(
                    instantaneous_rate(spiketrain, sampling_period=samp_period)
                ).flatten()

                times = np.arange(0, len(instant_rate)) * samp_period
                # Perform Fourier Transform
                fft_result = np.fft.fft(instant_rate - np.mean(instant_rate)) / len(
                    instant_rate
                )
                freqs = np.fft.fftfreq(len(instant_rate), times[1] - times[0])
                freqs_pos = freqs[: len(freqs) // 2]
                fft_result_pos = fft_result[: len(fft_result) // 2]

                # Find dominant frequencies and phases
                dominant_freq_indices = np.argsort(np.abs(fft_result_pos))[::-1][
                    :n_freqs
                ]  # Top 5 frequencies
                dominant_freqs = freqs_pos[dominant_freq_indices]
                freqs_list.extend(dominant_freqs)
                dominant_phases = np.angle(fft_result_pos[dominant_freq_indices])
                dominant_magnitudes = fft_result[dominant_freq_indices]

                num_subplots = 3 if show_freq_subplot else 2

                # Plot signal
                ax1 = plt.subplot(num_subplots, 1, 1)
                plt.plot(
                    times,
                    instant_rate,
                    label=spiketrain.name,
                    color=get_cell_colour(spiketrain),
                )
                plt.eventplot(spiketrain.segment.events[1], linewidths=2)
                plt.title("Signal")
                plt.xlabel("Time")
                plt.ylabel("Amplitude")

                # Plot frequency decomposition.
                plt.subplot(num_subplots, 1, 2)
                plt.plot(
                    freqs_pos, np.abs(fft_result_pos), color=get_cell_colour(spiketrain)
                )
                plt.xlim(0, 0.1)
                plt.title("Frequency Decomposition")
                plt.xlabel("Frequency")
                plt.ylabel("Magnitude")

                # Overlay dominant frequencies
                for freq, phase in zip(dominant_freqs, dominant_phases):
                    plt.axvline(x=freq, color="black", linestyle="--")

                if show_freq_subplot:
                    # Plot sinusoids lined up with signal
                    plt.subplot(num_subplots, 1, 3, sharex=ax1)

                summed_signal = np.zeros_like(times).magnitude
                summed_signal = summed_signal.astype(complex)

                for freq, phase, value in list(
                    zip(dominant_freqs, dominant_phases, dominant_magnitudes)
                )[0:n_freqs]:
                    sinusoid = 2 * value * np.exp(2j * np.pi * freq * times)
                    summed_signal += sinusoid
                    if show_freq_subplot:
                        plt.plot(times, sinusoid.real, linestyle="--", linewidth=1)

                plt.subplot(num_subplots, 1, 1)
                plt.plot(times, summed_signal.real + np.mean(instant_rate))
            plt.legend()
            plt.tight_layout()
            if do_plot:
                plt.show()
            else:
                _ = plt.close()
                _ = plt.clf()
        return freqs_list

    def plot_serial_correlation_coeffs_grid(self, cell_type="All"):
        """PLots the serial correlation Coefficients of all the cells"""
        plotting_tuples = []  # (lags, coeffs, spiketrain.name,spiketrain.description)
        from rvm_analysis.plotting_tools import plot_grid_from_list

        for spiketrain in self.spiketrain_iterator(cell_type=cell_type):
            isi_s = isi(spiketrain)
            mean_isi = np.mean(isi_s)

            corr, lags = autocorr(
                isi_s - mean_isi, both_sides=False, biased=True, return_coeffs=True
            )

            plotting_tuples.append(
                (lags, corr, spiketrain.name, spiketrain.description)
            )
        _ = plot_grid_from_list(
            plotting_tuples,
            len(plotting_tuples),
            "The Serial Correlation Coefficients of All Spike Trains, at a given lag",
        )
        return plotting_tuples
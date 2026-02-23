"""
This module contains all of the small snippets which get re-used but
are not full function. It also contains all the import codes, which use specific paths.
```data_path``` can be changed to import data from a different location.
"""
import os
import neo
import mat73
import numpy as np
import pandas as pd
from pathlib import Path
from neo.io import NWBIO
from copy import deepcopy
from neo import AnalogSignal
from scipy.stats import chi2
from collections import Counter
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt
from quantities import Quantity, s, Hz, ms

from elephant.statistics import instantaneous_rate
from elephant.kernels import GaussianKernel
from elephant.conversion import BinnedSpikeTrain

from rvm_analysis.Spike2Tools import Spike2DataManager
from rvm_analysis.colours import get_cell_colour

data_path = Path("../../data")


def import_NWB_data(path: Path)-> Spike2DataManager:
    """
    Imports a set of NWB files as a set of neo blocks. 
    Uses neo's default converter (which cannot handle keywords in the NWB file.)
    In our case, each neo file is a single block, but this does not have to be the case.
    Returns a Spike2DataManager object with a property dm.blocks containining a list of blocks,
    and functions for plotting these blocks (e.g.  dm.plot_blocks()).
    """
    all_blocks = []
    for filepath in path.rglob("*.nwb"):
        reader =  NWBIO(filepath,'r')
        blocks = reader.read_all_blocks()
        all_blocks.extend(blocks)

    dm = Spike2DataManager(path)
    dm.blocks = all_blocks
    return dm

def import_only_evoked_cells_nwb():
    """ Imports only the evoked nwb data."""
    dm = import_NWB_data(data_path / "saved_nwb_data/evoked_data")
    return dm

def import_only_evoked_and_evoked_ongoing_cells_nwb():
    """ Imports the evoked and evoked_ongoing nwb data."""
    dm = import_NWB_data(data_path / "saved_nwb_data/ongoing_data")
    return dm

def import_ongoing_and_evoked_ongoing_cells_nwb():
    """ Imports the ongoing and evoked ongoing nwb data."""
    dm = import_NWB_data(data_path / "saved_nwb_data/ongoing_data")
    return dm

def import_ongoing_cells_nwb():
    """ Imports only the ongoing nwb data."""
    dm = import_NWB_data(data_path / "saved_nwb_data/ongoing_data")
    return dm


def import_extra_cells():
    dm = Spike2DataManager(
        data_path / "Further_ON_OFF_Pairs_Mary/spike_files",
        excel_notes_path=data_path / "Further_ON_OFF_Pairs_Mary/Summary of Paired Recordings.xlsx",
    )
    dm.read_blocks_into_memory()

    def combine_cells(row):
        cells = []
        for entry in list(row):
            entry = str(entry).strip()
            if entry != "nan":
                cells.append(entry)
            else:
                return "/".join(cells)
        return "/".join(cells)

    excel_notes = pd.read_excel(
        data_path / "Further_ON_OFF_Pairs_Mary/Summary of Paired Recordings.xlsx",
        sheet_name=None,
    )

    cell_data = {}
    for index, row in excel_notes["LHB Muscimol"].iterrows():
        cell_data[row[0]] = [combine_cells(row[2:6]), row[6]]
    for index, row in excel_notes["LHB Bicuculline"].iterrows():
        cell_data[row[0]] = [combine_cells(row[2:6]), row[6]]

    for block in dm.blocks:
        file_name = Path(block.file_origin).stem
        labels = cell_data[file_name][0].split("/")
        event_labels = cell_data[file_name][1].split(",")
        for j, spiketrain in enumerate(block.segments[0].spiketrains):
            spiketrain.name = labels[j]
    
        segment = block.segments[0]
        for event in segment.events:
            print(event.name)
            if event.name in ["Keyboard","keyboard"]:
                print(event.labels)
                if len(event) != 0:
                    event.labels = event_labels
                print(event.labels)

        for i, events in enumerate(segment.events):
            if events.name.lower() == "keyboard":

                labels_str = events.labels.astype(str)

                # masks for each infusion type
                bic_mask = np.char.find(labels_str, "bicuculline_infusion") != -1
                mus_mask = np.char.find(labels_str, "muscimol_infusion") != -1

                # --- create new events ---
                if np.any(bic_mask):
                    bic_events = neo.Event(
                        times=events.times[bic_mask],
                        labels=events.labels[bic_mask],
                        name="bicuculline_infusion"
                    )
                    segment.events.append(bic_events)

                if np.any(mus_mask):
                    mus_events = neo.Event(
                        times=events.times[mus_mask],
                        labels=events.labels[mus_mask],
                        name="muscimol_infusion"
                    )
                    segment.events.append(mus_events)

                # --- remove moved events from keyboard ---
                keep_mask = ~(bic_mask | mus_mask)

                remaining_events = neo.Event(
                    times=events.times[keep_mask],
                    labels=events.labels[keep_mask],
                    name=events.name
                )

                # replace the old Keyboard event
                segment.events[i] = remaining_events
                print("split_out_drug")
    return dm


def import_neutral_cells_extra():
    dm = Spike2DataManager(data_path / "Zhigang_Neutral_Extra/cells")
    dm.read_blocks_into_memory()
    dm.set_cell_names("Neutral")
    dm.standardize_event_names()

    cell_pinch_labels = pd.read_excel(
        data_path / "Zhigang_Neutral_Extra/cell_pinch_labels.xlsx",
        sheet_name=None,
    )
    
    for block in dm.blocks:
        segment = block.segments[0]

        # loop over all sheets
        for df in cell_pinch_labels.values():
            print(Path(block.file_origin).name)
            match = df[df["Experiment"] ==Path(block.file_origin).name]
            if not match.empty:
                labels = match["pinch_labels"].iloc[0].split(",")

                for event in segment.events:
                    if event.name == "keyboard":
                        event: neo.Event = event
                        print(event.labels)
                        event.labels = labels
                        print(event.labels)
                print("found_match,renamed_labels")
                break
    return dm


def import_on_off_trials():
    dm = Spike2DataManager(
        data_path / "Tail_Flick_Recordings_De_Preter/Spike_Files",
        data_path / "Tail_Flick_Recordings_De_Preter/experimentnotes.xlsx",
    )  # "data manager"
    dm.read_blocks_into_memory()
    dm.update_spike_names("Cell type",row_cell_labelling=True,cell_type_column_name="Cell type")
    return dm


def import_neutral_cells():
    dm = Spike2DataManager(data_path / "Neutral_Cell_Recordings_Melissa/")
    dm.read_blocks_into_memory()
    dm.set_cell_names("Neutral")
    return dm


def import_spontaneous_cells():
    dm = Spike2DataManager(
        data_path / "Spontaneous_Recordings_Melissa_Zhigang/"
        "Pared_Spike_Files",
        data_path / "Spontaneous_Recordings_Melissa_Zhigang/"
        "CA_Ongoing activity expts. (05-09-24).xlsx",
    )
    dm.read_blocks_into_memory()
    # dm.update_spike_names("Cell Class", row_cell_labelling=True,cell_type_column_name="Cell Class")
    unique_id = 0
    id_name_tuples = []
    experiment_and_cell_type = dm.experiment_notes[
        ["Cell Class", "Experiment","pinch_sequence"]
    ]#.dropna()
    print(experiment_and_cell_type)
    for block in dm.blocks:
        segment = block.segments[0]
        head, tail = os.path.split(segment.file_origin)
        experiment = tail.split("_cut_pared.smrx")[0] #! This is not standard for pathlib
        
        print("Experiment:", experiment)
        cell_types = experiment_and_cell_type.loc[
            experiment_and_cell_type["Experiment"] == experiment, "Cell Class"
        ]
        print("Corresponding cell types:", cell_types)

        cell_types = cell_types.iloc[0].split("/")
        for i, spiketrain in enumerate(segment.spiketrains):
            try:
                spiketrain.name = cell_types[i].upper()
            except Exception as e:
                print(spiketrain.file_origin, experiment, cell_types, e)
            spiketrain.description = unique_id
            id_name_tuples.append((unique_id, spiketrain.name))
            unique_id += 1
    dm.standardize_event_names()

    # Now go back throuhgh and re-label all the events by their correct pinch sequences
    for block in dm.blocks:
        head, tail = os.path.split(block.segments[0].file_origin)
        experiment = tail.split("_cut_pared.smrx")[0] #! This is not standard for pathlib
        pinch_sequences = experiment_and_cell_type.loc[
            experiment_and_cell_type["Experiment"] == experiment, "pinch_sequence"
        ]
        print("Corresponding pinch_sequences:", pinch_sequences)

        for event in block.segments[0].events:
            if event.name == "keyboard":
                if len(event) !=0:
                    pinch_sequences = pinch_sequences.iloc[0].split(",")
                    event.labels = pinch_sequences

    return dm


def get_test_times(cut_spikes, kernel_width, cut_time, remove=0, sampling_period=None):
    """
    Takes a list of cut spikes, and a cut time and sampling period.

    Calculates a rate function for each spike train.

    Splits the rate function into a train and test rate function.

    train_times and test_times calculated once, so cut_spikes
    must all have the same t_start and t_stop.

    Remove removes the start and end of the signal, because the rate function
    border estimation can be faulty sometimes.

    ## Returns
    `ON_train, ON_test, train_times, test_times`

    """
    rates = instantaneous_rate(
        cut_spikes,
        sampling_period=kernel_width if sampling_period is None else sampling_period,
        kernel=GaussianKernel(kernel_width),
        # border_correction=True,
    )
    if cut_time is not None:
        train_times = rates.times[(rates.times < cut_time)][remove:]
        test_times = rates.times[rates.times >= cut_time]
        test_times = test_times[: len(test_times) - remove]
        ON_train, ON_test = test_train_split(rates, cut_time, remove)
    else:
        train_times = rates.times
        train_times = train_times[remove : train_times.shape[0] - remove]
        test_times = [] * Hz
        ON_train, ON_test = test_train_split(rates, None, remove)
    return (
        ON_train.magnitude.T,
        ON_test.magnitude.T,
        train_times.magnitude.T,
        test_times.magnitude.T,
    )


def test_train_split(
    cut_rate: AnalogSignal, train_seconds: Quantity, remove
) -> tuple[list, list]:
    """
    Splits the cut spikes into a training and a test set.
    Returns: (train_data, test_data)
    """
    if train_seconds is not None:
        N = cut_rate.shape[1]
        time_slices = np.tile(cut_rate.times < train_seconds, (N, 1)).T
        train_data = cut_rate[time_slices][remove:, :]
        test_data = cut_rate[np.invert(time_slices)]
        test_data = test_data[: test_data.shape[0] - remove, :]
    else:
        train_data = cut_rate
        train_data = train_data[remove : train_data.shape[0] - remove, :]
        test_data = [] * Hz

    # times = cut_rate.times
    # N = cut_rate.shape[1]
    # print(cut_rate.shape)
    # print(times.shape)
    # for j in range(N):
    #     rate = cut_rate[:, j].flatten()
    #     print(rate.shape)
    #     train_data.append(rate[times < train_seconds])
    #     test_data.append(rate[times >= train_seconds])
    return (train_data, test_data)


def poisson_confidence_interval(Y_mean, confidence=0.05):
    """
    Calculates the exact confidence interval for a Poisson-distributed count `x`
    using the chi-squared distribution relationship.

    Parameters:
    - x: observed count (integer).
    - alpha: significance level (float), default is 0.05 for a 95% confidence interval.

    Returns:
    - A tuple (lambda_lower, lambda_upper) representing the confidence interval for λ.
    """
    if Y_mean == 0:
        # Special case: If x = 0, the lower bound should be 0 and upper bound
        # is derived.
        lambda_lower = 0
        lambda_upper = chi2.ppf(1 - confidence / 2, 2 * (Y_mean + 1)) / 2
    else:
        # General case for x > 0
        lambda_lower = chi2.ppf(confidence / 2, 2 * Y_mean) / 2
        lambda_upper = chi2.ppf(1 - confidence / 2, 2 * (Y_mean + 1)) / 2

    return (lambda_lower, lambda_upper)


def get_empirical_nlpds(cell_data, discrete=True, plot=False):
    """
    Gets the NLPD of the test data, estimated from the empirical
    distribution of the training data. Works for discrete or
    continuous data.
    """
    train_ys, test_ys, train_x, text_x = cell_data
    nlpds = []
    for train_y, test_y in zip(train_ys, test_ys):
        if plot:
            _ = plt.figure(figsize=(25, 3))
            plt.plot(train_x, train_y)
            plt.show()
        if discrete:
            counts = Counter(train_y)
            total_count = len(train_y)

            # Calculate probability for each value in Y_test
            probs = [counts[y] / total_count for y in test_y]
            if plot:
                plt.bar(probs)
                plt.show()
        else:
            kde = gaussian_kde(train_y)  # Fit KDE to the data
            probs = kde.evaluate(test_y)  # Estimate density at each point in Y_test
            if plot:
                if isinstance(train_y, Quantity):
                    values = np.linspace(0, np.max(train_y.magnitude), 100)
                else:
                    values = np.linspace(0, np.max(train_y), 100)
                dist = kde.evaluate(values)  # Estimate density at each point in Y_test
                plt.plot(values, dist)
                plt.scatter(test_y, np.zeros_like(test_y))
        nlpd = -np.mean(np.log(probs))
        if plot:
            plt.title(nlpd)
            plt.show()
        nlpds.append(nlpd)
    return nlpds


def bin_spikes_and_split(cut_spikes, bin_width, cut_time,end_time=None):
    """
    Takes a list of cut spikes and generates a test and training dataset of binned
    spike trains, suitable for modelling using a poisson likelihood. 

    ---
    Returns:
    - Binned training spike counts
    - Binned testing spike counts
    - Training Bin centres
    - Test Bin centres
    """
    t_stop = cut_spikes[0].t_stop if end_time is None else end_time
    binned_spikes = BinnedSpikeTrain(
        cut_spikes, bin_size=bin_width, t_start=0 * s, t_stop=t_stop
    )

    train_spikes: BinnedSpikeTrain = binned_spikes.time_slice(
        0 * s, cut_time, copy=True
    )
    test_spikes = binned_spikes.time_slice(cut_time, t_stop, copy=True)

    return (
        train_spikes.to_array(),
        test_spikes.to_array(),
        np.array(train_spikes.bin_centers),
        np.array(test_spikes.bin_centers),
    )


def fit_exponential(x, y, initial=None, ax=None, plot_function=True, color="red"):
    """
    Fits an exponential function to the data, and plots the result along with the data
    on axes.
    """
    # Fit the function a * np.exp(b * t) + c to x and y
    popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, x, y, initial)

    a = popt[0]
    b = popt[1]
    c = popt[2]

    # Create the fitted curve
    x_fitted = np.linspace(np.min(x), np.max(x), 100)
    y_fitted = a * np.exp(b * x_fitted) + c

    # Plot
    if ax is None:
        fig, ax = plt.subplots()
    if plot_function:
        ax.plot(x, y, label="Raw data")
    ax.plot(x_fitted, y_fitted, "k", label="Fitted curve", color=color)
    return a, b, c


def fit_sigmoid(x, y, initial=None, ax=None, plot_function=True, color="red"):
    """
    Fits a sigmoid function to the data, and plots the result along with the data
    on axes. y = a (1/(1+exp(kx)) - 0.5) + c
    """
    def sigmoid(t,a,c,d, k):
        return a* (1 / (1 + np.exp(k * (t-d))) - 0.5) + c
    popt, pcov = curve_fit(sigmoid, x, y, initial)

    a = popt[0]
    c = popt[1]
    d = popt[2]
    k = popt[3]

    # Create the fitted curve
    x_fitted = np.linspace(np.min(x), np.max(x), 100)
    y_fitted = sigmoid(x_fitted,a,c,d,k)

    # Plot
    if ax is None:
        fig, ax = plt.subplots()
    if plot_function:
        ax.plot(x, y, label="Raw data")
    ax.plot(x_fitted, y_fitted, "k", label="Fitted curve", color=color)
    return a, k, c


def fit_exponential_2_param(
    x, y, initial=None, ax=None, plot_function=True, color="red"
):
    """
    Fits an exponential function to the data, and plots the result along with the data
    on axes.
    """
    # Fit the function a * np.exp(b * t) + c to x and y
    popt, pcov = curve_fit(lambda t, a, b: a * np.exp(b * t), x, y, initial)

    a = popt[0]
    b = popt[1]

    # Create the fitted curve
    x_fitted = np.linspace(np.min(x), np.max(x), 100)
    y_fitted = a * np.exp(b * x_fitted)

    # Plot
    if ax is None:
        fig, ax = plt.subplots()
    if plot_function:
        ax.plot(x, y, label="Raw data")
    ax.plot(x_fitted, y_fitted, "k", label="Fitted curve", color=color)
    return a, b


def fit_linear(x, y, initial=None, ax=None, plot_function=False, color="purple"):
    """
    Fits an exponential function to the data, and plots the result along with the data
    on axes.
    """
    # Fit the function a * np.exp(b * t) + c to x and y
    popt, pcov = curve_fit(lambda t, a, b: a * t + b, x, y, initial)

    a = popt[0]
    b = popt[1]

    # Create the fitted curve
    x_fitted = np.linspace(np.min(x), np.max(x), 100)
    y_fitted = a * x_fitted + b

    # Plot
    if ax is None:
        fig, ax = plt.subplots()
    if plot_function:
        ax.plot(x, y, label="Raw data")
    ax.plot(x_fitted, y_fitted, "k", label="Fitted curve", color=color)

    return a, b


def get_spike_count_array(
    spikes_list: list[neo.SpikeTrain], bin_size=50 * ms, plot=True, mean=True, log=False
):
    """
    Takes a list of spikes and gets the sum of the counts in each bin. Uses bin edges. 
    Uses elephant's binned spiketrain structure under the hood.

    Returns
    -------
    times, counts, variances, binned_spikes (elephant object)
    """
    binned_spikes: BinnedSpikeTrain = BinnedSpikeTrain(spikes_list, bin_size)
    times = binned_spikes.bin_edges[0:-1]
    binned_spikes_array = binned_spikes.to_array()
    if mean:
        counts = np.mean(binned_spikes_array, axis=0)
    else:
        counts = np.sum(binned_spikes_array, axis=0)
    variances = np.var(binned_spikes_array, axis=0)
    if plot:
        plt.plot(times, counts)
        if log:
            plt.yscale("log")
        plt.show()
    return times.magnitude, counts, variances, binned_spikes



def color_from_class(classes):
    colours = get_cell_colour(names=["NEUTRAL", "ON", "OFF", "Unknown"])
    return list(map(lambda x: colours[int(x)], classes))


def high_pass_filter_signal(data,cutoff,fs,order):
    """ High pass filters a signal. """
    sos = butter(order, Wn=cutoff,fs=fs, btype='highpass', output='sos')
    return sosfiltfilt(sos, data)


def combine_datasets_into_data_manager(*dms: Spike2DataManager):
    """
    Combines the block lists of Spike2DataManagers to make analysis easier.
    The resulting datamanager has access to the same functions, but now runs them
    across all blocks.
    Obviously parameters like data directory and excel path are now meaningless, but
    this is a useful conviencience.
    """
    dm_combined = Spike2DataManager(None,None)
    dm_combined.blocks = []
    for dm in dms:
        blocks: list[neo.Block] = dm.blocks
        dm_combined.blocks.extend(deepcopy(blocks))

    return dm_combined


def z_score(signal):
    """Removes the mean of a signal, and divides by the standard deviation. """
    return (signal - np.mean(signal)) / np.std(signal)

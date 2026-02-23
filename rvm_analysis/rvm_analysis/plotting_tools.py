"""
Module which contains the functions for the more complex plots, including histogram
grids, gradient legends for plotting multiple ON- and OFF-cells in shades of green
and red, correlation matrices etc.
"""
import neo
from quantities import s
import pandas as pd
import seaborn as sns
import ptitprince as pt
from typing import Optional

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
from distfit import distfit
from scipy.stats import skew, kurtosis

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerPatch
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

from elephant.conversion import BinnedSpikeTrain
from elephant.statistics import mean_firing_rate
from elephant.spike_train_correlation import correlation_coefficient
from elephant.neo_tools import get_all_spiketrains
from elephant.spike_train_correlation import correlation_coefficient
from elephant.statistics import isi

from rvm_analysis.save_tools import save_in_folder
from rvm_analysis.colours import get_cell_colour


def hist_grid(
    iterator,
    length,
    title,
    colors,
    fit_dist=True,
    folder_path=None,
    file_name=None,
    xlim: Optional[tuple] = None,
    figsize=(25, 15),
    save=False,
):
    """Takes an iterator of tuples, (values, cell type, description),
    and plots the their histograms, with kurtosis and skewness.
    """
    fig, axes = plt.subplots(int(np.ceil(length / 5)), 5, figsize=figsize)
    axs: list[Axes] = axes.flatten()
    for i, values in enumerate(iterator):
        plot_values = np.array(values[0])
        if xlim is not None:
            plot_values = plot_values[
                (plot_values >= xlim[0]) & (plot_values <= xlim[1])
            ]
        else:
            plot_values = values[0]
        axs[i].hist(
            plot_values,
            color=colors[i],
            bins=100,
            density=True,
        )
        axs[i].set_title(
            f"Spiketrain {values[2]}, skew: {round(skew(plot_values),2)},"
            f"kurtosis: {round(kurtosis(plot_values,fisher=True),2)}",
            fontsize=15,
        )
        if xlim is not None:
            axs[i].set_xlim(xlim[0], xlim[1])
        # axs[i].set_xticks([],[])

        # # Plot normal distribution curve
        # axs[i].plot(x, p, 'k', linewidth=2)
        # mu, std = norm.fit(values[0])
        # xmin, xmax = axs[i].get_xlim()
        # x = np.linspace(xmin, xmax, 100)
        # p = norm.pdf(x, mu, std)

        if fit_dist:
            dist = distfit(
                distr=["johnsonsu"], verbose="silent"
            )  #! Not considering genextreme here as the
            #! long neuron spikes are more related to the trail structure?
            dist.fit_transform(values[0])  # Fit distributions on empirical data X
            dist.predict(
                values[0], verbose="silent"
            )  # Predict the probability of the response variables
            dist.plot(ax=axs[i], n_top=2, cii_properties=None, title="")

    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    if folder_path:
        save_in_folder(
            file_name if file_name is not None else title, folder_path, save=save
        )
    else:
        plt.show()


def plot_grid_from_list(
    iterator, length, title, folder_path=None, file_name=None, save=False
):
    """Takes an iterator of tuples, (locations, values, cell type, description),
    and plots the locations against values.
    """
    fig, axs = plt.subplots(int(np.ceil(length / 5)), 5, figsize=(25, 15))
    axs: list[Axes] = axs.flatten()
    for i, values in enumerate(iterator):

        axs[i].plot(values[0], values[1], color="r" if values[2] == "OFF" else "g")
        axs[i].set_title(f"Spiketrain {values[2]}", fontsize=15)

    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    if folder_path:
        save_in_folder(
            file_name if file_name is not None else title, folder_path, save=save
        )
    else:
        plt.show()


def plot_events_and_signals(
    event_lists: list[list] = None,
    continuous_funcs: list[tuple[callable, np.ndarray]] = None,
    continuous_signals: list[neo.AnalogSignal] = None,
    labels: list = None,
    colors=None,
    title: str = "",
):
    """
    Takes a list of tuples containing (functions,times), and a list of events,
    and plots then on an event plot.
    Labels should be events first, then continuous functions.
    """
    fig, axes = plt.subplots(figsize=(25, 6))
    ax: Axes = axes
    y_idx = 0
    if event_lists is not None:
        N_events = len(event_lists)
        ax.eventplot(
            event_lists,
            lineoffsets=np.arange(N_events),
            colors=colors[:N_events],
        )
        y_idx += N_events

    # Plot continuous functions
    if continuous_funcs is not None:
        for idx, (func, times) in enumerate(continuous_funcs):
            y = func(times)
            y = y / (2 * np.max(np.abs(y)))
            ax.plot(times, y + y_idx, color=colors[y_idx])
            y_idx += 1

    if continuous_signals is not None:
        for idx, (times, y) in enumerate(continuous_signals):
            y = y / (2 * np.max(np.abs(y)))
            ax.plot(times, y + y_idx, color=colors[y_idx])
            y_idx += 1

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signals and Events")
    plt.title(title)
    # ax.grid(True)
    plt.show()


def create_spiketrain_label(spiketrain, info="name"):
    """Creates a string bale for a spiketrain. info is one of `detailed`, `name`."""
    if info == "detailed":
        label = f"{spiketrain.description}:"
        f"$\\mu$: {mean_firing_rate(spiketrain).rescale('Hz'): .2f}"
        f", cv: {cv(isi(spiketrain)): .2f}"
        return label
    else:
        return spiketrain.name.upper()


def add_grad_legends(legend_colors, ax, fig, fontsize=12):

    # Custom colormaps for red and green gradients
    green_cmap = mcolors.LinearSegmentedColormap.from_list(
        "green_gradient", ["#00ff00", "#004d00"]
    )
    red_cmap = mcolors.LinearSegmentedColormap.from_list(
        "red_gradient", ["#ff0000", "#800000"]
    )

    blue_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_blue", ["#ffffff", "#0000ff"]
    )

    # Custom legend handler for a gradient patch
    class HandlerGradientPatch(HandlerPatch):
        def __init__(self, cmap, **kw):
            self.cmap = cmap
            super().__init__(**kw)

        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))

            x = xdescent
            y = ydescent
            width = width
            height = height

            rect = Rectangle((x, y), width, height, transform=trans, lw=0)
            rect.set_clip_on(False)

            ax = fig.add_axes([0, 0, 1, 1], frameon=False)
            ax.imshow(
                gradient,
                extent=[x, x + width, y, y + height],
                aspect="auto",
                cmap=self.cmap,
                transform=trans,
            )
            ax.axis("off")

            return [rect]

    # Create dummy rectangles to use as handles in the legend
    dummy_rects = [Rectangle((0, 0), 1, 1), Rectangle((0, 0), 1, 1)]

    cmaps = {"OFF": red_cmap, "ON": green_cmap, "Neutral": blue_cmap}

    # Add the custom legend with two gradient patches
    ax.legend(
        [dummy_rects[i] for i in range(len(legend_colors))],
        [color[1] for color in legend_colors],
        handler_map={
            dummy_rects[i]: HandlerGradientPatch(cmaps[color[1]])
            for i, color in enumerate(list(legend_colors))
        },
        loc="upper right",
        fontsize=fontsize,
    )


def plot_scatter_for_descriptors_fancy_figure(
    spiketrains,
    filename: str,
    desc_1: callable,
    desc_2: callable,
    xlabel: str,
    ylabel: str,
    title="",
    log_x=False,
    log_y=False,
    base_path="../Analysis_plots",
    dpi=300,
    figsize=(5, 5),
    save=False,
):
    descriptors = []

    descriptors.extend(
        (desc_1(spiketrain), desc_2(spiketrain), spiketrain.name)
        for spiketrain in spiketrains
    )

    _ = plt.figure(figsize=figsize, dpi=dpi)
    if title is not None:
        plt.title(title, fontsize=20, fontweight="bold", y=1.1)
    means, cvs, types = zip(*descriptors)
    colors = np.where(np.array(types) == "OFF", "r", "g")
    plt.scatter(means, cvs, c=colors)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="ON", markerfacecolor="g"),
        Line2D([0], [0], marker="o", color="w", label="OFF", markerfacecolor="r"),
    ]
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    plt.legend(handles=legend_elements)
    save_in_folder(filename, basePath=base_path, svg=True, save=save)


def plot_combined_spectra(
    spectra,
    filename: str,
    base_path="../Analysis_plots",
    plot_title="The Concurrent Power Spectra of Cells",
    normalize=False,
    figsize=(5, 5),
    dpi=300,
    ylog=False,
    xlog=False,
    lw=0.5,
    xlim=(0, 1),
    ylim=None,
    xlabel=None,
    save=False,
):
    from cycler import cycler

    _ = plt.figure(figsize=figsize, dpi=dpi)
    cycle = cycler(c=plt.get_cmap("tab10").colors)
    plt.gca().set_prop_cycle(cycle)
    for freq, psd in spectra:
        psd = psd.flatten()
        if np.max(psd) > 140:
            pass  # print(freq[np.argmax(psd)])
        if normalize:
            psd /= np.trapz(psd.T, freq)
        plt.plot(freq, psd, linewidth=lw)
    if ylog:
        plt.yscale("log")
    if xlog:
        plt.xscale("log")
    else:
        plt.xscale("linear")
    plt.title(plot_title, fontsize=20)
    if xlabel is None:
        plt.xlabel("Frequency (Hz)", fontsize=24)
    else:
        plt.xlabel(xlabel, fontsize=24)
    plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if normalize:
        plt.ylabel("Normalized\nPower (1/Hz)", fontsize=24)
    else:
        plt.ylabel("Power")
    plt.tick_params(labelsize=24)
    save_in_folder(filename, basePath=base_path, svg=True, save=save)


def plot_2_combined_spectra(
    spectra1,
    spectra2,
    filename: str,
    base_path="../Analysis_plots",
    plot_title="The Concurrent Power Spectra of Cells",
    normalize=False,
    figsize=(5, 5),
    dpi=300,
    ylog=False,
    xlog=False,
    lw=0.5,
    xlim=(0, 1),
    ylim=None,
    save=False,
):
    fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)
    for i, spectra in enumerate([spectra1, spectra2]):
        for freq, psd in spectra:
            psd = psd.flatten()
            if np.max(psd) > 140:
                print(freq[np.argmax(psd)])
            if normalize:
                psd /= np.trapz(psd.T, freq)
                print(np.trapz(psd.T, freq), "new cumulative power")
            ax[i].plot(freq, psd, linewidth=lw)
        ax[i].set_ylabel("Normalized\nPower (1/Hz)")
        if ylog:
            plt.yscale("log")
        if xlog:
            plt.xscale("log")
        plt.suptitle(plot_title)
        plt.xlabel("Frequency (Hz)")
        plt.xlim(xlim[0], xlim[1])
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
    save_in_folder(filename, basePath=base_path, svg=True, save=save)


def sort_corr_matrix(correlation_matrix):
    # Step 1: Extract the first row (or column) for correlations with first spike train
    first_row = correlation_matrix[0, :]

    # Step 2: Get the sorted indices based on the first row
    sorted_indices = np.argsort(-first_row)  # Use negative sign for descending order

    # Step 3: Reorder the rows and columns of the correlation matrix
    sorted_correlation_matrix = correlation_matrix[sorted_indices, :][:, sorted_indices]
    return sorted_correlation_matrix, sorted_indices


def compute_z_scored_similarity_matrix(binned_spike_array):

    # Z-score normalization
    mean = np.mean(binned_spike_array, axis=1, keepdims=True)
    std = np.std(binned_spike_array, axis=1, keepdims=True)
    A_normed = (binned_spike_array - mean) / std

    similarity_matrix = A_normed @ A_normed.T

    plt.imshow(sort_corr_matrix(similarity_matrix)[0])
    plt.colorbar()
    plt.show()


def plot_corr_matrix(
    spiketrains,
    bin_size=0.05 * s,
    sorted=True,
    plot_smooth_rates=False,
    magnitude=False,
    human_classes=None,
):
    binned_spikes = BinnedSpikeTrain(spiketrains, bin_size)
    binned_spikes_wide = BinnedSpikeTrain(spiketrains, 5 * s)
    B = binned_spikes_wide.to_array()

    correlation_matrix = correlation_coefficient(binned_spikes)
    if magnitude:
        correlation_matrix = np.abs(correlation_matrix)
    A = binned_spikes.to_array()
    times = np.linspace(0 * s, A.shape[1] * 5 * s, A.shape[1])

    sorted_corr, indices = sort_corr_matrix(correlation_matrix)

    def corr_vals(i, j):
        return np.round(
            correlation_matrix[list(indices)[i] if sorted else i][
                list(indices)[j] if sorted else j
            ],
            2,
        )

    heatmap = go.Heatmap(
        z=sorted_corr if sorted else correlation_matrix,
        x=[f"Train {i}" for i in range(correlation_matrix.shape[0])],
        y=[f"Train {i}" for i in range(correlation_matrix.shape[1])],
        colorscale="Viridis",
        showscale=True,
        hoverongaps=False,
        hoverinfo="text",
        text=[
            [
                f"Train {list(indices)[i] if sorted else i} vs"
                f"Train {list(indices)[j] if sorted else j}, "
                f"corr={corr_vals(i,j)}"
                for j in range(correlation_matrix.shape[1])
            ]
            for i in range(correlation_matrix.shape[0])
        ],
    )

    subplots = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        specs=[[{"type": "heatmap"}, {"type": "scatter"}]],
        subplot_titles=("Correlation Matrix", "Spike Trains"),
    )

    subplots.add_trace(heatmap, row=1, col=1)

    # Placeholder for spike train plot (to be updated on hover)
    spike_train_plot = go.Scatter(x=[], y=[], mode="lines", name="Spike Train 1")
    subplots.add_trace(spike_train_plot, row=1, col=2)

    # Add an empty second scatter trace for the second spike train
    spike_train_plot_2 = go.Scatter(x=[], y=[], mode="lines", name="Spike Train 2")
    subplots.add_trace(spike_train_plot_2, row=1, col=2)
    subplots.update_layout(legend=dict(x=1.1, y=0.5, traceorder="normal"))

    f = go.FigureWidget(subplots)

    corr = f.data[0]
    scatter2 = f.data[1]
    scatter = f.data[2]

    def update_point(trace, points, selector):
        try:
            k, m = points.point_inds[0]
            with f.batch_update():
                i, j = list(indices)[k], list(indices)[m]
                scatter.x = times
                scatter.name = ["No", "ON", "OFF"][
                    int(human_classes[i if sorted else k])
                ]
                if plot_smooth_rates:
                    scatter.y = B[i if sorted else k]
                    scatter2.y = B[j if sorted else m]
                else:
                    scatter.y = A[i if sorted else k]
                    scatter2.y = A[j if sorted else m]
                scatter2.x = times
                scatter2.name = ["No", "ON", "OFF"][
                    int(human_classes[j if sorted else m])
                ]

        except Exception as e:
            e
            #! This is because sometimes the cursor goes out of bounds
            pass

    corr.on_hover(update_point)
    return f, correlation_matrix


def plot_corr_matrix_fancy_fig(
    spiketrains,
    bin_size=0.05 * s,
    magnitude=False,
    human_classes=None,
    figsize=(15, 7),
    ratio=[0.5, 0.5],
    basepath="../Grant_Images",
    save=False,
):
    binned_spikes = BinnedSpikeTrain(spiketrains, bin_size)
    binned_spikes_wide = BinnedSpikeTrain(spiketrains, 5 * s)
    B = binned_spikes_wide.to_array()
    times_B = np.linspace(0 * s, B.shape[1] * 5 * s, B.shape[1])

    correlation_matrix = correlation_coefficient(binned_spikes)
    if magnitude:
        correlation_matrix = np.abs(correlation_matrix)

    sorted_corr, indices = sort_corr_matrix(correlation_matrix)

    fig = plt.figure(figsize=figsize)

    outer: plt.GridSpec = gridspec.GridSpec(1, 2, figure=fig, width_ratios=ratio)

    gss = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.1)

    # Plotting the correlation matrix
    ax_heatmap = fig.add_subplot(outer[0])
    sns.heatmap(
        sorted_corr,
        ax=ax_heatmap,
        cmap="viridis",
        cbar=True,
        square=True,
        cbar_kws={"location": "left"},
    )
    ax_heatmap.set_title("Correlation Matrix")
    ax_heatmap.set_xticks(np.arange(correlation_matrix.shape[0]) + 0.5)
    ax_heatmap.set_yticks(np.arange(correlation_matrix.shape[1]) + 0.5)

    # Fixed spike train plot for Train 4 and Train 13
    i, j = 29, 1  # Indices for Train 4 and Train 13
    k, m = 29, 26  # Indices for Train 4 and Train 13

    i, j, k, m = [list(indices)[point] for point in [i, j, k, m]]

    ax_spike_trains = gss.subplots(sharex=True)

    def color_from_name(name):
        return ["b", "g", "r"][["No", "ON", "OFF"].index(name)]

    def name_from_index(index):
        return ["No", "ON", "OFF"][int(human_classes[index])]

    ax_spike_trains[0].set_title("Spike Trains", loc="left")
    ax_spike_trains[0].plot(
        times_B,
        B[i],
        label=f"{name_from_index(i)} cell",
        color=color_from_name(name_from_index(i)),
        linewidth=0.9,
    )
    ax_spike_trains[0].plot(
        times_B,
        B[j],
        label=f"{name_from_index(j)} cell",
        color=color_from_name(name_from_index(j)),
        linewidth=0.9,
    )
    ax_spike_trains[0].legend(bbox_to_anchor=(1.0, 1.45))

    ax_spike_trains[0].set_ylabel("Spikes per bin")

    ax_spike_trains[1].plot(
        times_B,
        B[k],
        label=f"{name_from_index(k)} cell",
        color=color_from_name(name_from_index(k)),
        linewidth=0.9,
    )
    ax_spike_trains[1].plot(
        times_B,
        B[m],
        label=f"{name_from_index(m)} cell",
        color=color_from_name(name_from_index(m)),
        linewidth=0.9,
    )
    # ax_spike_trains[1].legend(loc='upper right')
    ax_spike_trains[1].set_ylabel("Spikes per bin")
    ax_spike_trains[1].set_xlabel("Bin time (s)")

    # fig.suptitle("Sorted Correlation plot for multicellular recordings", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    save_in_folder("sorted_correlation_plot", basePath=basepath, svg=True, save=save)


def plot_corr_matrix_fancy_fig_max_corr(
    spiketrains,
    bin_size=0.05 * s,
    magnitude=False,
    human_classes=None,
    figsize=(15, 7),
    basepath="../Grant_Images",
    save=False,
):
    binned_spikes = BinnedSpikeTrain(spiketrains, bin_size)
    correlation_matrix = correlation_coefficient(binned_spikes)
    if magnitude:
        correlation_matrix = np.abs(correlation_matrix)

    sorted_corr, indices = sort_corr_matrix(correlation_matrix)

    fig, ax_heatmap = plt.subplots(figsize=figsize)

    # Mask the diagonal
    np.fill_diagonal(sorted_corr, np.nan)

    # Plotting the correlation matrix with custom highlighting
    sns.heatmap(
        sorted_corr,
        ax=ax_heatmap,
        cmap="viridis",
        cbar=True,
        square=True,
        cbar_kws={"location": "left"},
    )
    ax_heatmap.set_title("Correlation Matrix")
    ax_heatmap.set_xticks(np.arange(correlation_matrix.shape[0]) + 0.5)
    ax_heatmap.set_yticks(np.arange(correlation_matrix.shape[1]) + 0.5)

    # Highlighting the highest correlation in each row and column (excluding diagonal)
    for i in range(sorted_corr.shape[0]):
        row_max = np.nanargmax(sorted_corr[i, :])
        col_max = np.nanargmax(sorted_corr[:, i])

        # Draw lower triangle (row max)
        ax_heatmap.add_patch(
            plt.Polygon(
                ((row_max, i), (row_max + 1, i), (row_max, i + 1)),
                color="red",
                alpha=0.5,
            )
        )

        # Draw upper triangle (col max)
        ax_heatmap.add_patch(
            plt.Polygon(
                ((i, col_max), (i + 1, col_max), (i + 1, col_max + 1)),
                color="blue",
                alpha=0.5,
            )
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    save_in_folder("sorted_correlation_plot", basePath=basepath, svg=True, save=save)


def plot_interactive_grid(grid_vals, spiketrains, human_classes=None):
    binned_spikes_wide = BinnedSpikeTrain(spiketrains, 5 * s)
    B = binned_spikes_wide.to_array()
    times_B = np.linspace(0 * s, B.shape[1] * 5 * s, B.shape[1])

    correlation_matrix = grid_vals

    heatmap = go.Heatmap(
        z=correlation_matrix,
        x=[f"Train {i}" for i in range(correlation_matrix.shape[0])],
        y=[f"Train {i}" for i in range(correlation_matrix.shape[1])],
        colorscale="Viridis",
        showscale=True,
        hoverongaps=False,
        hoverinfo="text",
        text=[
            [
                f"Train {i} vs Train {j}, corr={np.round(correlation_matrix[i][j],2)}"
                for j in range(correlation_matrix.shape[1])
            ]
            for i in range(correlation_matrix.shape[0])
        ],
    )

    subplots = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        specs=[[{"type": "heatmap"}, {"type": "scatter"}]],
        subplot_titles=("Correlation Matrix", "Spike Trains"),
    )

    subplots.add_trace(heatmap, row=1, col=1)

    # Placeholder for spike train plot (to be updated on hover)
    spike_train_plot = go.Scatter(x=[], y=[], mode="lines", name="Spike Train 1")
    subplots.add_trace(spike_train_plot, row=1, col=2)

    # Add an empty second scatter trace for the second spike train
    spike_train_plot_2 = go.Scatter(x=[], y=[], mode="lines", name="Spike Train 2")
    subplots.add_trace(spike_train_plot_2, row=1, col=2)
    subplots.update_layout(legend=dict(x=1.1, y=0.5, traceorder="normal"))

    f = go.FigureWidget(subplots)

    corr = f.data[0]
    scatter2 = f.data[1]
    scatter = f.data[2]

    def update_point(trace, points, selector):
        try:
            k, m = points.point_inds[0]
            with f.batch_update():
                scatter.x = times_B
                scatter.name = ["No", "ON", "OFF"][int(human_classes[k])]
                scatter.y = B[k]
                scatter2.y = B[m]

                scatter2.x = times_B
                scatter2.name = ["No", "ON", "OFF"][int(human_classes[m])]

        except Exception as e:
            e  # If the cursor goes out of bounds
            pass

    corr.on_hover(update_point)
    return f, correlation_matrix


def plot_2d_consecutive_isi_s(dm, cell_type="All"):
    """Creates a 2d histogram of consecutive ISI_s for all spiketrains
    using plotly.
    """
    # Generate random data
    for spiketrain in dm.spiketrain_iterator(cell_type=cell_type):
        isi_s = isi(spiketrain)

        # Create a DataFrame
        df = pd.DataFrame({"x": isi_s[0:-1], "y": isi_s[1:]})

        # Create a 2D histogram
        fig = px.density_heatmap(
            df,
            x="x",
            y="y",
            nbinsx=100,
            nbinsy=100,
            color_continuous_scale="Viridis",
            title=f"Consecutive ISI_s for spiketrain {spiketrain.description}",
        )
        # Show the plot
        fig.show()

def plot_metric_kovalevskaya(
    results: dict,
    metric=None,
    param=None,
    xlabel=None,
    figsize=(6, 6),
    dpi=300,
    xlim=None,
    log=False,
):
    """
    Plots custom raincloud plots (violin + strip) for a metric across ON, OFF, and NEUTRAL cells,
    without using boxplots.
    """
    if metric:
        plot_data = {
            result: values["metrics"][metric] for result, values in results.items()
        }
    elif param:
        plot_data = {
            result: [cell_params[param] for cell_params in values["param_arrays"]]
            for result, values in results.items()
        }
    else:
        raise ValueError("You must specify either 'metric' or 'param'.")

    # Flatten the data into a dataframe
    data = []
    for group, values in plot_data.items():
        data.extend([(group, v) for v in values])
    df = pd.DataFrame(data, columns=["Group", "Value"])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot half-violins using seaborn (manually oriented left for horizontal)
    sns.violinplot(
        x="Value",
        y="Group",
        data=df,
        scale="width",
        inner=None,
        linewidth=0,
        cut=0,
        bw=0.5,
        orient="h",
        palette={name: get_cell_colour(names=name) for name in plot_data.keys()},
        ax=ax,
    )

    # Overlay stripplot
    sns.stripplot(
        x="Value",
        y="Group",
        data=df,
        orient="h",
        jitter=0.15,
        size=3,
        alpha=0.5,
        color="k",
        ax=ax,
    )

    # Optional vertical line for R2 = 0
    if metric == "R2":
        min_y, max_y = ax.get_ylim()
        span = max_y - min_y
        plt.vlines(0, min_y + 0.2 * span, max_y - 0.2 * span, color="gray", linestyle="--")

    ax.set_xlabel(xlabel if xlabel is not None else metric, fontsize=14)
    ax.set_ylabel("Cell Type", fontsize=14)
    ax.set_yticklabels(plot_data.keys())

    if xlim is not None:
        ax.set_xlim(xlim)
    if log:
        ax.set_xscale("log")

    plt.tight_layout(pad=2)
    plt.show()


def plot_metric(
    results: dict,
    metric=None,
    param=None,
    xlabel=None,
    figsize=(6, 6),
    dpi=300,
    xlim=None,
    log=False,
):
    """
    Plots the rainclouds for a metric for on, off, and neutral cells.
    """
    if metric:
        plot_data = {
            result: values["metrics"][metric] for result, values in results.items()
        }
    elif param:
        plot_data = {
            result: [cell_params[param] for cell_params in values["param_arrays"]]
            for result, values in results.items()
        }

    x = list(plot_data.values())
    labels = [[key] * len(value) for key, value in plot_data.items()]
    labels = sum(labels, [])
    # Combine the data and create a categorical variable
    data = np.concatenate(x)

    # Create the figure and axis
    fig, axes = plt.subplots(figsize=figsize, dpi=dpi)
    ax: Axes = axes
    # Create the raincloud plot with customized colors and no outliers in the box plot

    pt.RainCloud(
        x=labels,
        y=data,
        ax=ax,
        orient="h",
        width_viol=0.6,
        width_box=0.2,
        palette={name: get_cell_colour(names=name) for name in plot_data.keys()},
        bw=0.5,
        alpha=0.6,
        move=0.0,
        box_showfliers=False,
        # box_show=False
    )
    if metric == "R2":
        min_y, max_y = ax.get_ylim()
        span = max_y - min_y
        plt.vlines(0, min_y + 0.2 * span, max_y - 0.2 * span)

    # Set the title and labels
    # ax.set_title(metric, fontsize=16)
    ax.set_xlabel(xlabel if xlabel is not None else metric, fontsize=14)
    ax.set_ylabel("Cell Type", fontsize=14)
    plt.tight_layout(pad=9)
    # Adjust y-axis labels to "ON" and "OFF"
    _ = ax.set_yticklabels(plot_data.keys())
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if log:
        plt.xscale("log")
    plt.show()


def plot_sorted_multicellular_data(block: neo.Block, start=None, end=None,event_names=["flick","Pinch_On"],
                                   event_colours = ['y','lime','black','purple'],sort=True):

    fig = plt.figure(figsize=(15, 10), dpi=300)

    if (start is not None) and (end is not None):
        spiketrains = [
            spikes[(spikes < 2000) & (spikes > 1500)]
            for spikes in get_all_spiketrains(block)
        ]
    else:
        spiketrains = get_all_spiketrains(block)

    if sort:
        binned_spikes = BinnedSpikeTrain(spiketrains, bin_size=5 * s)
        correlations = correlation_coefficient(binned_spikes)
        _, indices = sort_corr_matrix(correlations)
    else:
        indices = np.arange(len(spiketrains))

    names = [spiketrains[i].name for i in indices]
    plt.eventplot(
        [spiketrains[i].magnitude for i in indices],
        linewidths=0.03,
        colors=get_cell_colour(names=names),
    )
    # plt.title("RVM Multicellular Activity",fontsize=40)
    ylims = plt.ylim()
    for i, event_name in enumerate(event_names):
        if block.filter(name = event_name) != []:
            try:
                plt.vlines(
                    block.filter(name=event_name),
                    -0.5,
                    len(get_all_spiketrains(block)) + 0.5,
                    color=event_colours[i],
                    label=event_name,
                )
            except Exception as e:
                print(f"{event_name} not in block, or other error")

    plt.legend(fontsize=15)
    plt.xlim(-5, 2705)
    plt.ylim(-0.5, len(get_all_spiketrains(block)) - 0.5)
    plt.xlabel("Time (s)", fontsize=20)
    plt.show()


def basic_plot(
    x,
    y,
    title,
    xlabel,
    ylabel,
    titlesize,
    fontsize,
    figsize,
    linewidth,
    linestyle,
) -> Axes:
    """Provides a really basic plot, for which all the parameters must be set. This means no lazy plotting."""
    fig, axes = plt.subplots(figsize=figsize)
    ax: Axes = axes
    if linestyle is "x":
        ax.scatter(x, y, s=linewidth)
    else:
        ax.plot(x, y, linewidth=linewidth, linestyle=linestyle)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=fontsize,
    )
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax

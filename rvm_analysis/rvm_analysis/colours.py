from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import neo
from neo.core.spiketrainlist import SpikeTrainList
import matplotlib.colors as mcolors
from typing import Optional, Union

class on_off_colour_generator:
    def __init__(self):
        self.idx_on = 0
        self.idx_off = 0
        self.idx_neutral = 0

    def gen_colors(self, event_names):
        # Generate unique shades of red for 'OFF' events
        off_colors = plt.get_cmap("Reds")(np.linspace(0.3, 1, event_names.count("OFF")))
        # Generate unique shades of green for 'ON' events
        on_colors = plt.get_cmap("Greens")(np.linspace(0.3, 1, event_names.count("ON")))
        neutral_colors = plt.get_cmap("Blues")(np.linspace(0.3, 1, event_names.count("NEUTRAL")))

        # Create a list to hold the colors
        event_colors = []

        # Assign colors based on the event type
        for event in event_names:
            if event == "OFF":
                event_colors.append(off_colors[self.idx_off])
                self.idx_off += 1
            elif event == "ON":
                event_colors.append(on_colors[self.idx_on])
                self.idx_on += 1
            elif event == "NEUTRAL":
                event_colors.append(neutral_colors[self.idx_neutral])
                self.idx_neutral += 1

        return event_colors

    def gen_colors_from_red_green_black(self, colors):
        # Generate unique shades of red for 'OFF' events
        off_colors = plt.get_cmap("Reds")(np.linspace(0.7, 1, colors.count("r")))
        # Generate unique shades of green for 'ON' events
        on_colors = plt.get_cmap("Greens")(np.linspace(0.7, 1, colors.count("g")))

        # Create a list to hold the colors
        event_colors = []

        # Assign colors based on the event type
        for color in colors:
            if color == "red" or color == "r":
                event_colors.append(off_colors[self.idx_off])
                self.idx_off += 1
            elif color == "green" or color == "g":
                event_colors.append(on_colors[self.idx_on])
                self.idx_on += 1
            else:
                event_colors.append(color)
                self.idx_on += 1

        return event_colors


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


def get_cell_colour(spiketrains: Union[list[neo.SpikeTrain], neo.SpikeTrain,None] = None, names: Union[list[str],None,str] = None):
    """Gets the colour of a spiketrain from its name, for labelling.
    'OFF' is red, 'ON' is green, otherwise blue. Also has a names input.
    """
    colours = defaultdict(
        # lambda: "gray", {"off": "r", "on": "g", "neutral": "b", "neutral_extra": "b"}
        lambda: "gray",
        {
            "off": "#ffb000",
            "on": "#dc267f",
            "neutral": "#648fff",
            "neutral_extra": "#648fff",
            "Unknown": "#a281db",
            "Ignore": "#896882",
            "unlabelled": "#d78e87"
        },
    )
    if spiketrains is not None:
        if isinstance(spiketrains, list) or isinstance(spiketrains, tuple) or isinstance(spiketrains, SpikeTrainList):
            return [colours[str(spiketrain.name).lower()] for spiketrain in spiketrains]
        else:
            return colours[str(spiketrains.name).lower()]
    elif names is not None:
        if isinstance(names, list) or isinstance(names, tuple):
            return [colours[name.lower()] for name in names]
        else:
            return colours[names.lower()]
    else:
        raise ValueError("No spiketrains or names given")


def generate_color_spectrum(base_hex, num_colors=10,var_size=0.3):
    """Takes a base colour and generates variations in lightness around it."""
    rgb = mcolors.hex2color(base_hex)
    h, l, s = mcolors.rgb_to_hsv(rgb)
    lightness_variations = np.linspace(max(0, l - var_size), min(1, l + var_size), num_colors)
    spectrum = [mcolors.to_hex(mcolors.hsv_to_rgb((h, s, lv))) for lv in lightness_variations] #type: ignore

    return spectrum
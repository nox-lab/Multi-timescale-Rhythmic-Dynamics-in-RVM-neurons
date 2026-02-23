"""
This file contains functions to analyse the trial dynamics of ON, OFF and Neutral cells.
"""
from pathlib import Path
import elephant.conversion as conv
import pandas as pd

def create_trial_df(bin_size,spiketrain_list_by_cell):
    """
    Takes a list of spiketrains by cell and a bin size,
    and builds a long form data frame which regression can be performed on.
    """
    # Collect rows to build a long-form DataFrame
    rows = []

    for cell_idx, trials in enumerate(spiketrain_list_by_cell):
        for trial_idx, spiketrain in enumerate(trials):
            # Get animal ID from file_origin
            animal_id = Path(spiketrain.file_origin).name

            binned = conv.BinnedSpikeTrain(spiketrain, bin_size=bin_size)
            counts = binned.to_array().flatten()
            
            times = binned.bin_edges[0:-1]

            # Add rows for each bin
            for i, (t, c) in enumerate(zip(times, counts)):
                rows.append({
                    'index': len(rows),
                    'animal': animal_id,
                    'cell': cell_idx,
                    'trial': trial_idx,
                    'time': float(t.magnitude),
                    'count': c
                })

    df = pd.DataFrame(rows)
    return df
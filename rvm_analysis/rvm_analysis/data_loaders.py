import os
import requests
from tqdm import tqdm
from neo.io import NWBIO
from pathlib import Path
from pynwb import NWBHDF5IO

from rvm_analysis.Spike2Tools import Spike2DataManager

# Filter files by protocol
def has_protocol(filepath, protocols):
    """Check if an NWB file has a given protocol in its metadata."""
    with NWBHDF5IO(path=filepath, mode='r') as io:
        nwbfile = io.read()  # nwb is now a pynwb.NWBFile object

        # Get the protocol of the NWB file
        file_protocol = nwbfile.protocol.lower()
        print(file_protocol, filepath)
        x = any(p.lower().strip() == file_protocol for p in protocols)
        print(x)
        return x


def read_dataset_by_protocol(protocols: list = ["evoked"],base_data_path: Path = Path("../../data/dandiset"), convert_blocks_to_old_labels=True):
    """
    Reads an NWB dataset (usually at ../../data/dandiset) and converts to a list of neo blocks.
    Dataset protocols should be list containing combinations of "evoked", "ongoing" or "evoked/ongoing".
    """
    # Recursive search for NWB files
    all_nwb_files = list(base_data_path.rglob("*.nwb"))

    filtered_files = [f for f in all_nwb_files if has_protocol(f, protocols)]

    # Read blocks from filtered NWB files
    all_blocks = []
    for filepath in filtered_files:
        reader =  NWBIO(filepath,'r')
        blocks = reader.read_all_blocks()
        all_blocks.extend(blocks)
        reader.close()

    print(f"Found {len(all_blocks)} blocks from {len(filtered_files)} NWB files.")
    dm = Spike2DataManager(base_data_path)
    dm.blocks = all_blocks

    if convert_blocks_to_old_labels:
        dm = rename_blocks_to_old_labels(dm)

    return dm




def download_dandiset(dandiset_id, output_dir):
    """
    Download all NWB files and dandiset.yaml from a DANDI dandiset,
    directly from the API, avoiding the CLI which breaks for python 3.9.2.

    Parameters:
        dandiset_id (str): The ID of the dandiset (e.g., '001708')
        output_dir (str): Local folder to save files into
    """
    os.makedirs(output_dir, exist_ok=True)

    # Download the dandiset.yaml first
    yaml_url = f"https://api.dandiarchive.org/api/dandisets/{dandiset_id}/versions/draft/"
    r_yaml = requests.get(yaml_url)
    r_yaml.raise_for_status()
    yaml_data = r_yaml.json()

    yaml_path = os.path.join(output_dir, "dandiset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        import json
        f.write(json.dumps(yaml_data, indent=2))
    print(f"Saved dandiset.yaml to {yaml_path}")

    # Iterate through assets pages
    api_url = f"https://api.dandiarchive.org/api/dandisets/{dandiset_id}/versions/draft/assets/"

    while api_url:
        resp = requests.get(api_url)
        resp.raise_for_status()
        data = resp.json()

        for asset in data.get("results", []):
            # Only download NWB files or YAML
            if asset["path"].endswith(".nwb") or asset["path"].endswith(".yaml"):
                file_path = os.path.join(output_dir, asset["path"])
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Construct the correct download endpoint from asset_id
                asset_id = asset["asset_id"]
                download_url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"

                print(f"Downloading {asset['path']}...")

                with requests.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get("Content-Length", 0))
                    with open(file_path, "wb") as f, tqdm(
                        desc=asset["path"],
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            bar.update(len(chunk))

        # Next page if paginated
        api_url = data.get("next")

    print("All NWB files and YAML downloaded successfully!")


def rename_blocks_to_old_labels(dm):
    """
    For compatibility with all the functions which originally used the spike2 labelling of "flick" and "heat", and had "block.file_origin" as the identifier.
    """
    remapping = {
        "paw_withdrawal": "flick",
        "heat": "heat",
        "pinch": "keyboard",
        "EKG": "EKG"
    }

    spiketrain_id = 0
    for block in dm.blocks:
        segment = block.segments[0]
        print([event.name for event in segment.events])
        for event in segment.events:
            event.name = remapping[event.name]
        block.file_origin = Path(block.annotations['identifier']).with_suffix(".smrx")
        print([event.name for event in segment.events])
        print(block.file_origin)
        for spiketrain in segment.spiketrains:
            spiketrain.description = spiketrain_id
            spiketrain_id +=1
            spiketrain.file_origin = block.file_origin
    return dm
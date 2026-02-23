# Multi-timescale Rhythmic Dynamics in RVM neurons

This repository contains the code for the paper Multi-timescale Rhythmic Dynamics in RVM neurons.

## Instructions

**It is important that the code be run in a python 3.9.2 environment.** This because at the time of writing SonPy, a closed source package developed by Cambridge Engineering Design, requires this version of python.

The code is contained mostly within Jupyter notebooks, each of which performs separate (or group of separate) analyses.

To run the code:
- First install the environment (we recommend using ```pyvirtualenv``` for this), by first downloading the correct version of python at the system level, and then, from any version of python where pyvirtualenv is installed, running the command:
```
python -m virtualenv env --python={path_to_python3.9.2}
```
Occasionally this must be run twice to generate the activation files for the package. If ```./env/Scripts/activate``` does not work following the command, run it again.

- Then, using the newly activated environment, install the packages listed in requirements.py, using the command:
```
pip install --no-compile -r requirements.txt
```
The no compile flag prevents an error from a legacy GPy test using this version, which still uses Python 2 code.

## Data description
- Then download the data and unpack this into the "data" directory such that the structure is:

```
data / 

    sub-rat-1
        sub-rat-1_ses-1.nwb
    sub-rat-2
        sub-rat-2_ses-1.nwb
        sub-rat-2_ses-2.nwb etc etc
```

Most subjects have only one session. A few exceptions have more than one session. In this case, each of these sessions had at a roughly 30 minute gap between recordings, and the electrode was moved to gather new cells in each case, so for the purposes of bootstrapping we accounted for cells recorded at the same time, e.g. at the file level. This could be verified using the recording start time for each of the sessions.

From there, each notebook will import the correct subset of the data, depending on the protocol. E.g. analyses using the evoked responses will import all the "evoked/ongoing" and "evoked" data, whilst those using the ongoing data will import "ongoing" and "evoked/ongoing". 

## Data download

The dataset can either be downloaded using the dandi CLI:

```
pip install dandi
dandi download dandi:001708
```

However, this command has been known to break on python 3.9.2. In that case, the function:

```python
from rvm_analysis.data_loaders import download_dandiset

download_dandiset("001708","./data")
```

will download the data into a data directory.

### Data collection and structure

Overall, the data was collected from many experimental sessions over several years. Each nwb file contains the datetime when sampling started, showing the distribution of recordings dates. The data has been grouped by experimental structure, rather than experimenter or date, for ease of analysis. Evoked data consists of ON, OFF and NEUTRAL cells where heat stimulus trials were roughly every 5 minutes. Ongoing data consists of ON, OFF, and NEUTRAL cells with trials at the start and end of the recordings, but a large window of unstimulated activity in the middle, suitable for investigating "baseline" cell activities. Evoked/Ongoing contains only NEUTRAL cells, whose firing rates were not affected by the trials, and so these were also used for the ongoing GP fits as well as trial analyses of NEUTRAL cells.

## Code structure

- Finally, the main code consists of three sections:
  -  Jupyter notebooks in the **notebooks** folder
  -  Analysis and plotting code in the **rvm_analysis** package,
  -  Code to convert from spike2 files (smrx) to neo in the **spike2neo** package.
- The **rvm_analysis** and **spike2neo** packages should be installed as editable packages by cd'ing individually to the outer rvm_analysis and spike2neo folders in the command line, and running the command below once in each directory

```
python -m pip install -e .
```


Running this for the rvm_analysis and spike2neo packages will install them in an editable form. Occasionally, if making changes to the packages, you may have to rerun this install command for the changes to be seen by the jupyter notebooks.

When trying to run files, because of the version of python, you will get the warning: "The version of Python associated with the selected kernel is no longer supported. Please consider selecting a different kernel." This cannot be helped until a newer version of the sonpy package is released.

## Notebook structure

The notebooks are split into: the Bayesian Curve Fitting models ("Bayesian Trial function fits"); the Data Conversion code which was originally used to convert the spike data to NWB format; the gpytorch models, which analyse the GP fits to the spiketrain firing rates; other image panel code, for example for Panel 1 and the heart rate analyses; and Supplementary Analyses.

**Important:** Before running the GP fitting code, the file "save cell activity.ipynb" should be run to create a faster loading copy of the data. The pickle files created are then used in the GP fits to speed up loading/reloading.
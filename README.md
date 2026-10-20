[![DOI](https://zenodo.org/badge/1252579754.svg)](https://doi.org/10.5281/zenodo.23370235)

# OPM-COREG-GUI

## What is it?

This repository provides a GUI for co-registering multiple coordinate frames used in OPM-based neuroimaging, including:

- Scalp surface extracted from structural MRI (MRI coordinate frame)
- Digitised 3D head scan (head coordinate frame)
- OPM sensors (device coordinate frame)
- Auxiliary devices (e.g., EOG electrodes)

The goal is to compute accurate **rigid transformations (6 DoF : rotation + translation)** between these coordinate systems.

The GUI is primarily designed for co-registering OPM sensors with anatomical MRI, but it also serves as a general tool for computing rigid body transformations between arbitrary 3D surfaces.

Future updates will expand support to conventional MEG, EEG, and other neuroimaging modalities.

## Features

- Interactive 3D visualisation (VTK-based)
- Pairwise landmark selection
- ICP-based refinement
- Multi-coordinate frame transformation pipeline
- Support for auxiliary devices

## What do you need?

- **3D master scan** (using structured light scanner) of the participant head with all the attachments (sensor helmet and auxiliary devices)
- **3D head scan** (using structured light scanner) of the participant head without any attachments
- **Scalp/skin surface** generated from T1 scan (e.g. from freesurfer's recon-all)
- **Fiducial coordinates** (LPA, RPA and nasion) estimated from the structural MRI
- **Measurement raw file** (.fif) containing sensor locations (or alternatively a CSV containing sensor coordinates and unit vectors)
- **Sensor helmet surface mesh** (.ply or .stl) or **sensor helmet landmark locations** (.csv)
- **Auxiliary device surface mesh** (.ply or .stl) or **auxiliary device landmark locations** (.csv)

## What do you get?

The GUI outputs following rigid body transforms (arrows in the figure below) between different coordinate frames (nodes in the figure below) :

- Head to MRI (in mm) ※
- Head to MRI (in m) (equivalent to -trans.fif in MNE-Python) ※
- Device (sensor helmet) to head (in mm) ※
- Device (sensor helmet) to head (in m) (equivalent to dev_head_t in MNE-Python) ※
- Auxiliary to device (sensor helmet) transform
- Auxiliary to master scan (in mm)
- Head to master scan (in mm)
- Device (sensor helmet) to master scan (in mm)

※ : Directly usable for localizing source activity

![Transforms summary](/docs/resources/transforms_flowchart.png)

## How to launch the GUI?

1. **On Windows** : Use a dedicated .exe file for Windows, which is available through GitHub releases. The .exe file is built using PyInstaller and includes all dependencies, so no additional installation is required.

1. **On MacOS/Linux** : Create a dedicated virtual environment and install the GUI as an editable package.

Install the GUI package in editable mode using pip in your desired virtual environment:

```bash
pip install -e .
```

Launch the GUI using either of the following commands:

```bash
opm-coreg
```

Or,

```bash
python -m opm_coreg
```

## How to handle different OPM helmets?

The GUI allows two ways of handling OPM helmets from different manufacturers :

1. **When helmet design is available (e.g. custom helmet design)** : Load helmet surface file (.stl or .ply) and mark same points on both source and target windows in the pairwise mode.

1. **When helmet design is not available (e.g. commercial systems)** : When helmet design is not available, the user is expected to load a CSV file containing coordinates of the pre-defined landmark points on the helmet. After loading the CSV file, continue to the pairwise mode as usual. When obtaining the rigid transform, only points marked on the source window will be used, rather than all the landmarks available in the CSV, allowing user to dynamically choose important and visible landmarks.

**Note**:

- Among commercial systems, the GUI is tested only on **Fieldline helmet** and OPM data.
- It is assumed that the helmet landmarks and sensors locations, both are in the device coordinate frame.
- The fitting will not work unless the helmet design or landmark CSV is loaded.
- When both helmet design and landmark CSV are loaded, nearest landmark coordinates in the CSV file will be utilized for generating rigid transform rather than the ones marked on the helmet.
- In either case, marking >=6 points in the pairwise mode will ensure better fit.

## Practical tips

- Set environment variable `OPM_COREG_GUI_ROOT_DIR` which serves as the GUI output directory.
- If the above environment variable is not set, `~/Documents/opm-coreg-gui` or similar will be created to write output files.
- A log file (.csv with timestamp) is created under `<GUI_output_directory>/logs` for each instance of the GUI opened.
- All messages displayed in the display window of the GUI (bottom-most scrollable window) are logged into this file.
- A sub-directory is created under `<GUI_output_directory>/results` when a 3D master scan is loaded in the GUI. The name is derived from the 3D master scan filename by omitting the file extension.
- In case where no 3D master scan is loaded, outputs are generated under `<GUI_output_directory>/results/master_scan_missing`.
- All rigid transforms (4×4 transformation matrices) are saved in .txt and .npy formats under `<GUI_output_directory>/results/<master_scan_filename>`
- Along with transforms, GUI also automatically saves marked points on different surfaces under `<GUI_output_directory>/results/<master_scan_filename>/cache`.
- Specify `<GUI_output_directory>/results/<master_scan_filename>` folder when loading saved transforms to continue the co-registration process.
- We recommend completing co-registration process (all steps in GUI) for a single master scan file first, before moving to the next master scan or participant.

## Notes

- Currently developing and testing on Python 3.11.14
- More detailed documentation will be added in future releases.

## Known issues

- None

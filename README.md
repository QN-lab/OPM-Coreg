# OPM-COREGISTRATION-GUI

## What is it?

This repository contains the code for GUI which is primarily used for co-registering scalp surface generated from structural MRI of the participant brain (in MRI co-ordinate space), digitized 3D scan of a participant head surface (in head co-ordinate space), OPM sensors (in device co-ordinate space) and any auxiliary devices such as Head Positioning Indication (HPI) coils, eye-blink channels etc.

## What do you need?

- 3D scan (using LIDAR based head scanner) of the participant head without any attachments
- Scalp surface generated using FreeSurfer
- Measurement raw file (.fif) containing sensor locations
- 3D scan (using LIDAR based head scanner) of the participant head with all the attachments (sensor helmet and auxiliary devices)
- Fiducial co-ordinate file containing LPA, RPA and nasion co-ordinates estimated from the structural MRI

## What do you get?

The GUI will output following transforms needed for source localization problem :
- Auxiliary to device (sensor helmet) transform
- Device (sensor helmet) to scan (3D scan with attachments) transform
- Head to scan (3D scan with attachments) transform
- Head to MRI transform

![Transforms summary](/docs/resources/transforms_flowchart.png)

## How to use?

The GUI is entirely developed in Python (3.11) and we suggest creating a dedicated environment for installing required packages and running the GUI. All required packages are enlisted in `requirements.txt`.

On Windows, run the `launch_gui.bat` file to create the virtual environment, install dependencies, and run the GUI.

On other systems, after manually creating and activating the environment, follow these steps :

```
cd OPM-COREGISTRATION-GUI
python -m src.main
```

The command to launch the GUI will be updated once the package name is established.

## Notes

- Currently developing and testing on Python 3.11.14
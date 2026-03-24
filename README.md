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

```bash
cd OPM-COREGISTRATION-GUI
python -m src.main
```

The command to launch the GUI will be updated once the package name is established.

## How to handle different OPM helmets?

The GUI allows two ways of handling OPM helmets from different manufacturers :

1. **When helmet design is available** : Load helmet STL file and mark same points on both helmet view and 3D LIDAR scan with attachments in the pairwise mode. This will fit the helmet design on the 3D LIDAR scan and generate `device to scan` transform.

1. **When helmet design is not available** : When helmet design is not available, the user is expected to load a csv file containing coordinates of the pre-defined landmark points on the helmet. After loading the csv file, continue to the pairwise mode as usual. When obtaining `device to scan` transform, only points marked on the landmark view will be used, rather than all the landmarks available in the csv, allowing user to dynamically choose important landmarks.

**Note**:

- The fitting will not work unless the helmet design or landmark csv is loaded.
- When both helmet design and landmark csv are loaded, landmark coordinates in the csv file will be utilized for fitting purposes rather than the ones marked on the helmet.
- In either case, marking >=6 points in the paiwise mode will ensure better fit.

## Notes

- Currently developing and testing on Python 3.11.14

# Building Windows Executables

1. Create a clean environment for packaging. This is important to avoid conflicts with other packages and ensure that the executable works correctly. All packaging so far has been done in a clean conda environment with Python 3.11. You can create a new environment with the following command:

```
conda create -n conda_opm_coreg python=3.11
```

2. Install the package. This will install the `opm_coreg` package in editable mode, which is necessary for PyInstaller to find the source code.

```
pip install -e .
```

3. Run following command to create a Windows executable. This will create a folder called `dist` in the current directory, which will contain the executable and all necessary files. The file `packaging/run_opm_coreg_gui.py` is a launcher script that imports the `main` function from the `opm_coreg.main` module and runs it. This is necessary because PyInstaller cannot run a module directly.

```
pyinstaller --onedir --windowed --collect-all vtk --collect-all pyvista --collect-all mne --collect-all lazy_loader -n opm-coreg .\packaging\run_opm_coreg_gui.py
```
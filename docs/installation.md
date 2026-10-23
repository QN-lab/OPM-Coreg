# Installation

OPM-Coreg can be launched using one of the following methods.

---

## Option 1: Windows Installer (Recommended)

Pre-built Windows executables are available from the GitHub Releases page.

1. Access the latest release from: [Github Releases](https://github.com/QN-lab/OPM-Coreg/releases)

2. Under the latest release, locate the **Assets** section.

3. Download and extract the archive named `opm-coreg-<version>-win64.zip`.

4. Run the executable file to launch the GUI.

No Python installation is required when using the packaged executable.

---

## Option 2: Install from PyPI

Install OPM-Coreg into a Python environment of your choice:

```bash
pip install opm-coreg
```

Launch the GUI:

```bash
opm-coreg
```

---

## Option 3: Run from Source

Clone the repository:

```bash
git clone https://github.com/QN-lab/OPM-Coreg.git
cd OPM-Coreg
```

Install the package in editable mode:

```bash
pip install -e .
```

Launch the GUI using either:

```bash
opm-coreg
```

or

```bash
python -m opm_coreg
```

---

## System Requirements

- Python 3.11 or newer (for PyPI and source installations)
- Windows (pre-built executables currently provided for Windows)
- Sufficient memory and graphics capability for working with 3D MRI surfaces and scan data

---

## Troubleshooting

### Command not found

If the command

```bash
opm-coreg
```

is not recognised, ensure the package was installed into the currently active Python environment.

### Installation issues

Verify that your Python version meets the minimum requirement:

```bash
python --version
```

OPM-Coreg requires Python 3.11 or later.

### Source installation issues

When installing from source, ensure all package dependencies were installed successfully:

```bash
pip install -e .
```

Consult the GitHub repository issue tracker for known issues and updates.

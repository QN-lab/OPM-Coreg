# Changelog

## [2.0.2] - 2026-10-19

### Fixed

- Fixed `wglMakeCurrent failed` error on Windows when using PyVista with VTK 10.3.0 and above.

### Added

- Python package name changed to `opm_coreg` and command name changed to `opm-coreg` for consistency with PyPI naming conventions.
- Added `packaging` folder with instructions for building Windows executables using PyInstaller.
- Added startup message to GUI terminal with current version number and citation information.

## [2.0.1] - 2026-06-30

### Fixed

- Correct handling of helmet landmark arrays (NumPy truth-value bug)
- GUI failure when using commercial helmet landmark CSVs
- Updated tests with correct helmet landmark arrays

## [2.0.0] - 2026-06-30

This is going to be the first openly available major version of the OPM-Coreg-GUI.

### Added

- Landmark-based pairwise matching of auxiliary/device surface to 3D master scan.
- Pairwise matching of 3D head scan to 3D master scan with subsequent refinement using ICP in the ROI formed by the marked points.
- Fiducial-based matching of scalp surface to 3D head scan with subsequent refinement using ICP in the ROI formed by the marked points.
- Saving of all generated transforms, logs, and cached data (marked points on surfaces).
- Loading of all saved data for continuation of coregistration.
- Surface toggling in the target window.
- GUI launch scripts for Windows and MacOS/Linux
- Python packaging info, unit tests, and changelog.

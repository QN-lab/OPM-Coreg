# Changelog

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

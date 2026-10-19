"""Define constants used across the application."""

from enum import Enum

PAIR_COLORS = [
    (1.0, 0.0, 0.0),  # Red
    (0.0, 0.0, 1.0),  # Blue
    (0.0, 1.0, 0.0),  # Green
    (1.0, 1.0, 0.0),  # Yellow
    (1.0, 0.0, 1.0),  # Magenta
    (0.0, 1.0, 1.0),  # Cyan
    (0.0, 1.0, 0.0),  # Lime
    (0.580, 0.0, 0.827),  # DarkViolet
    (1.0, 0.647, 0.0),  # Orange
    (1.0, 0.753, 0.796),  # Pink
]

FIDUCIAL_COLORS = {
    "NAS": (0.6, 0.0, 0.0),
    "LPA": (0.0, 0.0, 0.6),
    "RPA": (0.0, 0.4, 0.0),
    "NOSE": (1.0, 0.5, 0.0),  # orange
    "CHIN": (0.8, 0.8, 0.0),  # yellowish
}

DEEPPINK = (1.0, 0.08, 0.58)


class HelmetSystem(Enum):
    CUSTOM = 0
    FIELDLINE = 1
    CERCA = 2


HELMET_SYSTEM_DICT = {
    "fieldline": HelmetSystem.FIELDLINE,
    "cerca": HelmetSystem.CERCA,
}

FIELDLINE_LANDMARK_COLORS = {
    "A": (1.0, 0.753, 0.796),  # Pink
    "B": (0.0, 1.0, 1.0),  # Cyan
    "C": (0.2, 0.4, 0.9),
}

DEFAULT_COLOR = (0.2, 0.4, 0.9)

CITATION_TEXT = (
    "Phalke, K., Cook, H., Barontini, G., Quinn, A., & Kowalczyk, A. (2026). "
    "OPM-Coreg [Computer software]. https://doi.org/10.5281/zenodo.23370235"
)

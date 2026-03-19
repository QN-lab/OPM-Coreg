"""Entry point for the OPM-MEG coregistration GUI."""

import sys

from PyQt5.QtWidgets import QApplication

from src.opm_coreg_gui import OPMCoregGUI


def main():
    """Launch the coregistration GUI application."""
    app = QApplication(sys.argv)

    # Create and show the main window
    window = OPMCoregGUI()
    window.show()

    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

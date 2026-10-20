"""Entry point for the OPM-Coreg GUI application."""

import sys

from PyQt5.QtWidgets import QApplication

from .gui import OPMCoregGUI


def main():
    """Launch the OPM-Coreg GUI application."""
    app = QApplication(sys.argv)

    # Create and show the main window
    window = OPMCoregGUI()
    window.show()

    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

@echo off

REM Check for venv directory
if not exist "venv" (
    echo Virtual environment not found. Creating...
    python -m venv venv

    echo Activating virtual environment...
    call venv\Scripts\activate

    echo Installing dependencies...
    pip install --upgrade pip
    pip install -e .
) else (
    echo Virtual environment found.
    echo Activating virtual environment...
    call venv\Scripts\activate

)

REM Run the GUI
echo Launching the GUI...
opm-coreg-gui

REM Pause to keep the terminal open
pause
@echo off

REM Check for venv directory
if not exist "venv" (
    echo Virtual environment not found. Creating...
    python -m venv venv
    echo Installing dependencies...
    call venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo Virtual environment found.
)

REM Activate venv (on windows)
echo Activating virtual environment...
call venv\Scripts\activate

REM Run the GUI
echo Launching the GUI...
python -m src.main

REM Pause to keep the terminal open
pause
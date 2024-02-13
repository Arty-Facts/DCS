echo off
set clean=%1
if /I "%clean%" EQU "clean" rmdir /s venv 
echo on

@REM create venv if it dose not exists
if not exist "venv" python -m venv venv

@REM assume that windows systems has a nvidia GPU
venv\Scripts\pip install -e .

@REM start env
venv\Scripts\Activate
@echo off
echo Installing Forex Trading Analyzer...

REM Check Python version
python --version

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
if not exist "recordings" mkdir recordings
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "templates" mkdir templates

echo Installation complete!
echo To run the application:
echo streamlit run app.py --server.port 5000
pause

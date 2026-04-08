@echo off
echo ========================================
echo Golden 68 Framework - Starting App
echo ========================================
echo.

:: Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Start Streamlit app
echo Starting Golden 68 Framework...
echo.
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run app.py

pause

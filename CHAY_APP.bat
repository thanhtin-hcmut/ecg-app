@echo off
chcp 65001 >nul
cd /d "%~dp0"
title App phan tich ECG
echo ================================================
echo   APP PHAN TICH ECG - Streamlit
echo ================================================

rem --- Dung Python 3.12 (thu vien da duoc Windows tin cay) ---
set "PY312=%LOCALAPPDATA%\Python\pythoncore-3.12-64\python.exe"
if not exist "%PY312%" (
  echo [0/3] Chua co Python 3.12 - dang cai bang Python install manager...
  py install 3.12 --yes
  if not exist "%PY312%" py install 3.12
)
if not exist "%PY312%" (
  echo [LOI] Khong cai duoc Python 3.12. Tai tai https://www.python.org/downloads/release/python-3129/
  pause
  exit /b 1
)
"%PY312%" --version

if not exist venv312\Scripts\python.exe (
  echo.
  echo [1/3] Tao moi truong ao venv312...
  "%PY312%" -m venv venv312
)

echo.
echo [2/3] Cai thu vien ^(lan dau mat vai phut^)...
venv312\Scripts\python.exe -m pip install --upgrade pip -q
venv312\Scripts\python.exe -m pip install -r requirements.txt -c constraints.txt -q
if errorlevel 1 (
  echo [LOI] Cai thu vien that bai. Kiem tra mang Internet roi chay lai.
  pause
  exit /b 1
)

echo.
echo Kiem tra thu vien...
venv312\Scripts\python.exe -c "import numpy, pandas, scipy, matplotlib, neurokit2, streamlit; print('OK - pandas', pandas.__version__, '- streamlit', streamlit.__version__)"
if errorlevel 1 (
  echo.
  echo [LOI] Windows dang chan file thu vien ^(Smart App Control^).
  echo       Xem: Windows Security ^> App ^& browser control ^> Smart App Control.
  pause
  exit /b 1
)

if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  mkdir "%USERPROFILE%\.streamlit" 2>nul
  >"%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >>"%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo.
echo [3/3] Khoi dong app tai http://localhost:8601
echo       Dong cua so nay de tat app.
start "" cmd /c "timeout /t 6 >nul & start http://localhost:8601"
venv312\Scripts\python.exe -m streamlit run app.py --server.port 8601 --server.headless true --browser.gatherUsageStats false
pause

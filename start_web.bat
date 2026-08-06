@echo off
cd /d %~dp0
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 7861 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if %errorlevel%==0 (
    echo SP 工作台已在运行: http://127.0.0.1:7861/
    start "" http://127.0.0.1:7861/
    exit /b 0
)
python start_web.py
pause

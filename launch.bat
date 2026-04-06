@echo off
title ICT Analyzer
cd /d "C:\Users\yyk14\ict-analyzer"

:: 이미 실행 중인지 확인 (포트 8501)
netstat -ano | findstr ":8501 " > nul 2>&1
if %errorlevel% == 0 (
    echo [ICT Analyzer] 서버가 이미 실행 중입니다.
    start "" "http://localhost:8501"
    timeout /t 1 /nobreak > nul
    exit
)

:: Streamlit 서버 시작 (최소화 별도 창)
echo [ICT Analyzer] 서버 시작 중...
start /min "ICT Analyzer Server" python -m streamlit run app.py ^
    --server.port 8501 ^
    --server.headless true ^
    --browser.gatherUsageStats false

:: 서버 준비 대기 (최대 15초)
set /a cnt=0
:WAIT
timeout /t 1 /nobreak > nul
netstat -ano | findstr ":8501 " > nul 2>&1
if %errorlevel% == 0 goto OPEN
set /a cnt+=1
if %cnt% lss 15 goto WAIT

:OPEN
echo [ICT Analyzer] 브라우저 열기...
start "" "http://localhost:8501"
exit

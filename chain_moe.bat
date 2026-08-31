@echo off
REM ============================================================
REM  MoE training chain: wait for pretrain to finish, then run SFT.
REM  Pretrain itself is guarded by watchdog.bat; this script only
REM  polls the pretrain log for the completion marker and then
REM  launches the SFT watchdog.
REM
REM  ASCII only on purpose: cmd.exe reads .bat in the OEM codepage,
REM  and UTF-8 Chinese in comments can break variable expansion.
REM
REM  Start: powershell -c "Start-Process chain_moe.bat -WindowStyle Hidden"
REM ============================================================

REM  NOTE: pretrain_moe.log is held open by watchdog.bat's ">>" redirection,
REM  which does NOT grant write-sharing. Appending to it from here fails
REM  silently, so this script reads that log but writes to its own.
set PRELOG=C:\baidunetdiskdownload\Project\minimind\pretrain_moe.log
set CHAINLOG=C:\baidunetdiskdownload\Project\minimind\chain_moe.log
set SFTBAT=C:\baidunetdiskdownload\Project\minimind\wd_sft_moe.bat
set PREDONE=Epoch:[2/2](79390/79390)
set GIVEUP=MAXRETRY

echo [CHAIN] %DATE% %TIME% chain started, waiting for pretrain >> "%CHAINLOG%"

:waitpre
if not exist "%PRELOG%" goto sleep60
findstr /C:"%PREDONE%" "%PRELOG%" >nul && goto startsft
findstr /C:"%GIVEUP%" "%PRELOG%" >nul && goto abort
:sleep60
ping -n 61 127.0.0.1 >nul
goto waitpre

:startsft
echo [CHAIN] %DATE% %TIME% pretrain done-marker seen, waiting for process exit >> "%CHAINLOG%"

REM ---- The done marker is printed BEFORE the weights are written,
REM      so wait for python to actually exit or SFT would load a
REM      half-written pretrain_768_moe.pth ----
:waitexit
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*train_pretrain.py*' }) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    ping -n 21 127.0.0.1 >nul
    goto waitexit
)

echo [CHAIN] %DATE% %TIME% pretrain process exited, starting MoE full_sft in 30s >> "%CHAINLOG%"
ping -n 31 127.0.0.1 >nul
start "" /b "%SFTBAT%"
echo [CHAIN] %DATE% %TIME% wd_sft_moe.bat launched >> "%CHAINLOG%"
goto :eof

:abort
echo [CHAIN] %DATE% %TIME% pretrain watchdog gave up, NOT starting SFT >> "%CHAINLOG%"
goto :eof

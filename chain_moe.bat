@echo off
REM ============================================================
REM  MoE 训练链：预训练跑完后自动接上 full_sft
REM  预训练本身由 watchdog.bat 守护，本脚本只负责"接力"，
REM  轮询预训练日志里的完成标记，出现后再拉起 SFT 看门狗。
REM
REM  启动：powershell -c "Start-Process chain_moe.bat -WindowStyle Hidden"
REM ============================================================

set ROOT=C:\baidunetdiskdownload\Project\minimind
set PRELOG=%ROOT%\pretrain_moe.log
set PREDONE=Epoch:[2/2](79390/79390)
set GIVEUP=次仍失败，放弃

echo [CHAIN] %DATE% %TIME% 接力脚本启动，等待预训练完成 >> "%PRELOG%"

:waitpre
if not exist "%PRELOG%" goto sleep60
findstr /C:"%PREDONE%" "%PRELOG%" >nul && goto startsft
findstr /C:"%GIVEUP%" "%PRELOG%" >nul && goto abort
:sleep60
ping -n 61 127.0.0.1 >nul
goto waitpre

:startsft
echo [CHAIN] %DATE% %TIME% 预训练已打出完成标记，等待进程落盘退出 >> "%PRELOG%"

REM ---- 完成标记先于存权重打印，必须等 python 真正退出再接力，
REM      否则 SFT 会读到写了一半的 pretrain_768_moe.pth ----
:waitexit
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*train_pretrain.py*' }) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    ping -n 21 127.0.0.1 >nul
    goto waitexit
)

echo [CHAIN] %DATE% %TIME% 预训练进程已退出，30 秒后启动 MoE full_sft >> "%PRELOG%"
ping -n 31 127.0.0.1 >nul
start "" /b "%ROOT%\wd_sft_moe.bat"
echo [CHAIN] %DATE% %TIME% 已拉起 wd_sft_moe.bat >> "%PRELOG%"
goto :eof

:abort
echo [CHAIN] %DATE% %TIME% 预训练看门狗已放弃重试，不启动 SFT >> "%PRELOG%"
goto :eof

@echo off
REM ============================================================
REM  MiniMind 训练看门狗 —— MoE full_sft 阶段
REM  作用：训练进程意外退出时自动重启，靠 --from_resume 1 从最近的
REM        checkpoint 续上
REM
REM  安全性：启动时若发现同名训练脚本已在运行，会先等它结束再接管，
REM          不会起第二个进程抢显存 / 抢 checkpoint
REM
REM  启动：powershell -c "Start-Process wd_sft_moe.bat -WindowStyle Hidden"
REM ============================================================

REM ---------------- CONFIG ----------------
set PYEXE=C:\Users\Admin\anaconda3\envs\minimind\python.exe
set SCRIPTNAME=train_full_sft.py
set TRAIN=train_full_sft.py --use_moe 1 --from_weight pretrain --save_weight full_sft --batch_size 6 --accumulation_steps 3 --num_workers 4 --from_resume 1
set LOGFILE=C:\baidunetdiskdownload\Project\minimind\sft_moe.log
set DONEMARK=Epoch:[2/2](150953/150953)
set MAXRETRY=30
REM ----------------------------------------

cd /d C:\baidunetdiskdownload\Project\minimind\trainer
set N=0

echo [WATCHDOG] %DATE% %TIME% 看门狗启动，开始守护 >> "%LOGFILE%"

REM ---- 已有同名训练在跑？先等它结束，不打扰 ----
:waitexisting
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*%SCRIPTNAME%*' }) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    ping -n 31 127.0.0.1 >nul
    goto waitexisting
)

REM ---- 现有进程已结束。若日志已有完成标记，说明是正常跑完，直接退出 ----
if exist "%LOGFILE%" (
    findstr /C:"%DONEMARK%" "%LOGFILE%" >nul && goto already
)

:retry
set /a N+=1
if %N% GTR %MAXRETRY% goto dead
echo [WATCHDOG] %DATE% %TIME% 第 %N% 次拉起训练 >> "%LOGFILE%"

"%PYEXE%" -u %TRAIN% >> "%LOGFILE%" 2>&1

findstr /C:"%DONEMARK%" "%LOGFILE%" >nul && goto done

echo [WATCHDOG] %DATE% %TIME% 异常退出（第 %N% 次），30 秒后自动重启 >> "%LOGFILE%"
REM 用 ping 代替 timeout：隐藏窗口下 timeout 会因无控制台输入而报错
ping -n 31 127.0.0.1 >nul
goto retry

:done
echo [WATCHDOG] %DATE% %TIME% 训练正常完成，看门狗共拉起 %N% 次 >> "%LOGFILE%"
goto :eof

:already
echo [WATCHDOG] %DATE% %TIME% 训练已正常完成，看门狗无需介入 >> "%LOGFILE%"
goto :eof

:dead
echo [WATCHDOG] %DATE% %TIME% 连续重启 %MAXRETRY% 次仍失败，放弃 >> "%LOGFILE%"
goto :eof

# run.ps1
# 依次开启三个 PowerShell 窗口运行不同的 Python 脚本，每次间隔 5 秒

Start-Process powershell -ArgumentList "python server.py"
Start-Sleep -Seconds 7

Start-Process powershell -ArgumentList "python client.py --cid 1"
Start-Sleep -Seconds 3

Start-Process powershell -ArgumentList "python client.py --cid 2"

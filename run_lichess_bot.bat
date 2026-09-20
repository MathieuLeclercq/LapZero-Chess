@echo off
rem Lance le bot Lichess (lichess-bot + moteur uci.py) avec le venv du projet.
rem Les chemins relatifs de config.yml (logs, game_records) sont resolus depuis
rem python_src\lichess_bot, d'ou le changement de dossier.
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%python_src\lichess_bot"
"%ROOT%.venv\Scripts\python.exe" lichess-bot.py %*
endlocal

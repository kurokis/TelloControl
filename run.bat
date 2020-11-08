cd %~dp0
py -c "from lib.condalib import run_from_env; run_from_env([\"python control.py\"], env=\"telloenv\", close_after_process=True)"
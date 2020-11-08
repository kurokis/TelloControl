cd %~dp0
py -c "from lib.condalib import run_from_env; run_from_env([\"conda update conda\",\"conda env create -f environment.yml\"], env=\"base\", close_after_process=True)"
pause
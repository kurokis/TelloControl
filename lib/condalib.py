import os
from pathlib import Path


def find_anaconda():
    # Create a list of condidates for anaconda directory
    conda_dir_candidates = []

    # Windows default install directory
    conda_dir_candidates.append(Path.home()/"anaconda3")

    # Default directory when installed for all users
    conda_dir_candidates.append(Path("C:\\ProgramData\\Anaconda"))

    conda_dir_candidates.append(
        Path.home()/"AppData\\Local\\Continuum\\anaconda3")

    conda_dir = None
    for cand in conda_dir_candidates:
        if cand.is_dir():
            conda_dir = cand
    return conda_dir


def run_from_env(commands=None, env="base", close_after_process=True):
    # Finda anaconda directory
    conda_dir = find_anaconda()
    if conda_dir is None:
        print("Anaconda not found")
        return

    print("Anaconda found:", conda_dir)

    # List of commands
    cmds_ = []

    # Command to activate conda environment
    cmds_.append(str(conda_dir/"Scripts\\activate.bat")+" "+env)

    # User-defined commands
    if commands is not None:
        for command in commands:
            cmds_.append(command)

    # Join commands
    cmd_str = ""
    for cmd_ in cmds_[:-1]:
        cmd_str += cmd_ + " & "
    cmd_str += cmds_[-1]
    if close_after_process:
        cmd_str = "cmd /c \""+cmd_str+"\""
    else:
        cmd_str = "cmd /k \""+cmd_str+"\""

    # Run commands
    os.system(cmd_str)

from lib.condalib import find_anaconda

if __name__ == '__main__':
    conda_dir = find_anaconda()
    if conda_dir is None:
        print("Anaconda not found")
    else:
        print("Anaconda found at ", conda_dir)

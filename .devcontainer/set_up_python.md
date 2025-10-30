
# Steps:
1. Install pyenv as described [here](https://github.com/pyenv/pyenv?tab=readme-ov-file)
2. Run `pyenv install 3.11` (or whatever version of python that is compatible with this project like 3.14)
3. Run `pyenv local 3.11` to automatically use a new version of python locally

Additionally: to select a Pyenv-installed Python as the version to use, run one of the following commands:
    pyenv shell <version> -- select just for current shell session
    pyenv local <version> -- automatically select whenever you are in the current directory (or its subdirectories)
    pyenv global <version> -- select globally for your user account

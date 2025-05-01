### These commands are run after the docker image for the dev container is built ###

# Install pipx (https://pipx.pypa.io/stable/installation/)
sudo apt update -y
sudo apt install -y pipx

# Install poetry using pipx (https://python-poetry.org/docs/#installation)
pipx install poetry

# Configure poetry to create virtualenvs in project directory
poetry config virtualenvs.in-project true

# Install python dependencies
poetry install --no-interaction

# Install pre-commit hooks
poetry run pre-commit install

# Install playwright
# playwright install
# playwright install-deps

# Used for sonar extension code analysis
nvm install 18

# Activate virtual environment
source .venv/bin/activate
source ../.venv/bin/activate

# Show which Python interpreter is being used
which python

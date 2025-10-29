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

### Node/NVM Used for sonar extension code analysis and claude code
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Source nvm in the current shell session
export NVM_DIR="/usr/local/share/nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
# Below would be how to run this locally
# export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

nvm install --lts # Consider version 18 if we want a consistent version rather than the latest
nvm use --lts
npm install -g @anthropic-ai/claude-code

git config --global --add safe.directory /workspaces/auto-questions

# Activate virtual environment
source .venv/bin/activate

# Show which Python interpreter is being used
which python

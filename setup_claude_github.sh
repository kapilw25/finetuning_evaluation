#!/bin/bash
set -e

# Always work relative to HOME for installs
HOME_DIR="$HOME"
NVM_DIR="$HOME_DIR/.nvm"

echo ">>> Updating system..."
sudo apt update -y
sudo apt install -y git gh curl build-essential

echo ">>> Removing old Node.js if installed via apt..."
sudo apt remove -y nodejs npm || true

echo ">>> Installing NVM (Node Version Manager)..."
if [ ! -d "$NVM_DIR" ]; then
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
fi

# Load NVM immediately
export NVM_DIR="$NVM_DIR"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Persist NVM in bashrc (once)
if ! grep -q 'NVM_DIR' "$HOME_DIR/.bashrc"; then
  {
    echo 'export NVM_DIR="$HOME/.nvm"'
    echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"'
    echo 'export PATH="$HOME/.nvm/versions/node/$(nvm version)/bin:$PATH"'
  } >> "$HOME_DIR/.bashrc"
fi

echo ">>> Installing latest LTS Node.js..."
nvm install --lts
nvm alias default 'lts/*'
nvm use default

echo ">>> Installing Claude Code CLI..."
npm install -g @anthropic-ai/claude-code

echo ">>> Verifying installs..."
node -v
npm -v
command -v claude || echo "⚠️ Claude installed but not yet authenticated."

echo ">>> Configuring GitHub..."
git config --global user.name "Kapil Wanaskar"
git config --global user.email "kapilw25@gmail.com"

# Check GitHub auth status
if gh auth status &>/dev/null; then
  echo ">>> GitHub already authenticated ✅"

  # Configure git to use HTTPS with GitHub CLI credentials
  echo ">>> Configuring git to use HTTPS with GitHub CLI..."
  gh auth setup-git

  # Convert existing SSH remotes to HTTPS if they exist
  if git remote get-url origin 2>/dev/null | grep -q "git@github.com"; then
    echo ">>> Converting SSH remote to HTTPS..."
    SSH_URL=$(git remote get-url origin)
    HTTPS_URL=$(echo "$SSH_URL" | sed 's/git@github.com:/https:\/\/github.com\//')
    git remote set-url origin "$HTTPS_URL"
    echo ">>> Remote URL updated to: $HTTPS_URL"
  fi

else
  echo ">>> Authenticating with GitHub CLI (device flow)..."
  echo ">>> This will generate a 6-digit code for you to enter at https://github.com/login/device"
  gh auth login --web --protocol https

  # Configure git to use HTTPS with GitHub CLI credentials
  echo ">>> Configuring git to use HTTPS with GitHub CLI..."
  gh auth setup-git
fi

echo ">>> Reloading environment..."
source "$HOME_DIR/.bashrc"

# Detect if Claude is already logged in
CLAUDE_CONFIG_DIR="$HOME_DIR/.config/claude"
if [ -d "$CLAUDE_CONFIG_DIR" ]; then
  echo ">>> Claude already authenticated ✅"
  echo "👉 You can run 'claude' from any project."
else
  echo ">>> Claude not yet authenticated — starting login..."
  exec claude
fi

echo ">>> Setup complete!"

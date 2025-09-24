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

# SSH key setup (always in ~/.ssh, not project folder)
if [ ! -f "$HOME_DIR/.ssh/id_rsa.pub" ]; then
  echo ">>> Generating SSH key for GitHub..."
  ssh-keygen -t rsa -b 4096 -C "kapilw25@gmail.com" -f "$HOME_DIR/.ssh/id_rsa" -N ""
  echo ">>> SSH key generated. Add this public key to GitHub (https://github.com/settings/keys):"
  cat "$HOME_DIR/.ssh/id_rsa.pub"
else
  echo ">>> SSH key already exists:"
  cat "$HOME_DIR/.ssh/id_rsa.pub"
fi

# Check GitHub auth status
if gh auth status &>/dev/null; then
  echo ">>> GitHub already authenticated ✅"
else
  echo ">>> Authenticating with GitHub CLI (web login)..."
  echo ">>> Open in browser: https://github.com/login/device"
  gh auth login -w || {
    echo "⚠️ Browser launch failed (expected on headless server)."
    echo "👉 Please manually open https://github.com/login/device and enter the code above."
  }
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

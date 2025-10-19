#!/bin/bash
set -e  # stop on first error

# 1. Update & install curl
sudo apt update -y && \
sudo apt install -y curl && \

# 2. Install Node.js (LTS) and npm
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && \
sudo apt install -y nodejs && \

# 3. Install Claude Code globally
sudo npm install -g @anthropic-ai/claude-code && \

$echo "After CLAUDE LOGIN setup # Navigate to your project directory."
$echo "# cd /path/to/your/project"
$echo "git config user.name 'kapilw25' && git config user.email 'kapilw25@gmail.com'" 

# Launch Claude Code.
claude
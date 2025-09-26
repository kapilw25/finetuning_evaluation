# Install Node.js and npm: Claude Code requires Node.js (version 18 or greater) and npm. If you don't have them installed, you can do so using these commands:
# go to home directory 
sudo apt update
sudo apt install curl
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs

# Install Claude Code globally.
sudo npm install -g @anthropic-ai/claude-code

# Navigate to your project directory.
cd /path/to/your/project

# Launch Claude Code.
claude

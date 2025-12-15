#!/bin/bash

: << EOF
This scrpit sets up dependencies, and installs the appropriate vLLM fork as a submodule.

The vLLM fork is cloned as an editable package. 
Various changes are made to allow internal states to be extracted.
EOF

# fail upon error
set -e

# Colour Variables
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
NC="\033[0m"

show_help() {
    echo -e "$(cat << EOF
${GREEN}Setup Script for probe-runner${NC}

This script sets up dependencies and installs the vLLM fork as an editable package.

${BLUE}Usage:${NC}
    ./setup.sh [OPTIONS]

${BLUE}Options:${NC}
    --help              Show this help message
    --update-vllm       Update the vLLM fork from GitHub and reinstall
    --reinstall-vllm     Reinstall vLLM in editable mode (without updating)

EOF
)"
}

# check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}uv is not installed${NC}"
    echo -e "${YELLOW}Please install uv from https://docs.astral.sh/uv/getting-started/installation/${NC}"
    exit 1
else
    echo -e "${GREEN}uv is installed. Installing dependencies...${NC}"
    uv sync
    # Clean up any broken setuptools installation left by uv sync
    if [ -d ".venv/lib/python3.11/site-packages" ]; then
        for dist_info in .venv/lib/python3.11/site-packages/setuptools-*.dist-info; do
            if [ -d "$dist_info" ] && [ ! -f "$dist_info/METADATA" ]; then
                rm -rf "$dist_info" 2>/dev/null || true
            fi
        done
    fi
    # Ensure setuptools is properly installed
    uv pip install setuptools 2>/dev/null || true
fi

vllm_fork_url="https://github.com/lysandermawby/vllm.git"
vllm_fork_dir="vllm"

# Parse flags
UPDATE_VLLM=false
REINSTALL_VLLM=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      show_help
      exit 0
      ;;
    --update-vllm)
      UPDATE_VLLM=true
      shift
      ;;
    --reinstall-vllm)
      REINSTALL_VLLM=true
      shift
      ;;
    *)
      echo -e "${YELLOW}Unknown option: $1${NC}"
      echo -e "Use --help to see available options"
      exit 1
      ;;
  esac
done

# Handle vLLM fork
if [ -d "$vllm_fork_dir/.git" ]; then
    if [ "$REINSTALL_VLLM" = true ]; then
        echo "Reinstalling vLLM in editable mode..."
        uv pip install -e "$vllm_fork_dir"
    elif [ "$UPDATE_VLLM" = true ]; then
        echo "vLLM fork present. Updating..."
        git -C "$vllm_fork_dir" fetch origin
        git -C "$vllm_fork_dir" pull --ff-only origin main || {
            echo -e "${YELLOW}Could not fast-forward vLLM fork. Please resolve manually.${NC}"
        }
        echo "Reinstalling vLLM in editable mode after update..."
        uv pip install -e "$vllm_fork_dir"
    else
        echo "vLLM fork present. Skipping vLLM clone/update. (Use --update-vllm or --reinstall-vllm if needed.)"
    fi
else
    if [ -d "$vllm_fork_dir" ]; then
        echo -e "${YELLOW}Found existing $vllm_fork_dir without git metadata. This should be removed...${NC}"
        echo -e "If you want to remove this work which does not have relevant git information, run:"
        echo -e "  rm -rf \"$vllm_fork_dir\""
        echo -e "Skipping vLLM setup."
    else
        echo "Cloning vLLM fork..."
        git clone "$vllm_fork_url" "$vllm_fork_dir"
        echo "Installing vLLM common dependencies..."
        uv pip install -r "$vllm_fork_dir/requirements/common.txt"
        echo "Installing vLLM CPU dependencies..."
        uv pip install -r "$vllm_fork_dir/requirements/cpu.txt"
        echo "Installing vLLM in editable mode..."
        uv pip install -e "$vllm_fork_dir"
    fi
fi

# Install torch from CPU index (required for vLLM on CPU)
# This must be done separately to avoid index conflicts with other packages
echo "Installing PyTorch (CPU version)..."
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Install vLLM's common dependencies
echo "Installing vLLM common dependencies..."
uv pip install -r "$vllm_fork_dir/requirements/common.txt"

# Install vLLM's CPU-specific dependencies
echo "Installing vLLM CPU dependencies..."
uv pip install -r "$vllm_fork_dir/requirements/cpu.txt"

# Install vLLM as editable package
echo "Installing vLLM in editable mode..."
uv pip install -e "$vllm_fork_dir"

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

# Initialize and update git submodules if this is a git repository
if [ -d ".git" ]; then
    # Remove empty vllm directory if it exists (common when cloning without --recurse-submodules)
    if [ -d "$vllm_fork_dir" ] && [ ! -d "$vllm_fork_dir/.git" ]; then
        echo -e "${YELLOW}Found empty vllm submodule directory. Removing...${NC}"
        rm -rf "$vllm_fork_dir"
    fi
    
    echo -e "${GREEN}Initializing git submodules...${NC}"
    git submodule update --init --recursive || {
        echo -e "${YELLOW}Warning: Failed to initialize some submodules. Continuing anyway...${NC}"
    }
fi

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
    # vLLM submodule is properly initialized
    if [ "$REINSTALL_VLLM" = true ]; then
        echo "Reinstalling vLLM in editable mode..."
        # Set MAX_JOBS if not already set
        if [ -z "$MAX_JOBS" ]; then
            if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
                export MAX_JOBS=4
                echo -e "${YELLOW}Setting MAX_JOBS=4 for CUDA build to prevent OOM. Override with: export MAX_JOBS=N${NC}"
            else
                export MAX_JOBS=8
                echo -e "${YELLOW}Setting MAX_JOBS=8 for CPU build. Override with: export MAX_JOBS=N${NC}"
            fi
        fi
        uv pip install -e "$vllm_fork_dir"
    elif [ "$UPDATE_VLLM" = true ]; then
        echo "vLLM fork present. Updating..."
        git -C "$vllm_fork_dir" fetch origin
        git -C "$vllm_fork_dir" pull --ff-only origin main || {
            echo -e "${YELLOW}Could not fast-forward vLLM fork. Please resolve manually.${NC}"
        }
        echo "Reinstalling vLLM in editable mode after update..."
        # Set MAX_JOBS if not already set
        if [ -z "$MAX_JOBS" ]; then
            if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
                export MAX_JOBS=4
                echo -e "${YELLOW}Setting MAX_JOBS=4 for CUDA build to prevent OOM. Override with: export MAX_JOBS=N${NC}"
            else
                export MAX_JOBS=8
                echo -e "${YELLOW}Setting MAX_JOBS=8 for CPU build. Override with: export MAX_JOBS=N${NC}"
            fi
        fi
        uv pip install -e "$vllm_fork_dir"
    else
        echo "vLLM fork present. Skipping vLLM clone/update. (Use --update-vllm or --reinstall-vllm if needed.)"
    fi
elif [ -d "$vllm_fork_dir" ]; then
    # Directory exists but isn't a git repo - remove it and reinitialize
    echo -e "${YELLOW}Found $vllm_fork_dir without git metadata. Removing and reinitializing...${NC}"
    rm -rf "$vllm_fork_dir"
    if [ -d ".git" ]; then
        git submodule update --init --recursive "$vllm_fork_dir" || {
            echo -e "${RED}Failed to initialize vllm submodule.${NC}"
            exit 1
        }
    fi
fi

# Install common dependencies first
uv pip install -r "$vllm_fork_dir/requirements/common.txt"

# Detect hardware and install appropriate vLLM dependencies
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing vLLM CUDA dependencies..."
        
    # Install CUDA-specific dependencies
    uv pip install -r "$vllm_fork_dir/requirements/cuda.txt"
else
    echo "No CUDA detected. Installing vLLM CPU dependencies..."
    echo "Warning: vLLM on CPU has limited functionality and performance."
    uv pip install -r "$vllm_fork_dir/requirements/cpu.txt"
fi

# Install vLLM as editable package
echo "Installing vLLM in editable mode..."
# Set MAX_JOBS to prevent OOM during CUDA compilation
# CUDA compilation is memory-intensive, so limit parallelism
if [ -z "$MAX_JOBS" ]; then
    # Default to 4 jobs for CUDA builds to avoid OOM
    # Users can override by setting MAX_JOBS before running this script
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        export MAX_JOBS=4
        echo -e "${YELLOW}Setting MAX_JOBS=4 for CUDA build to prevent OOM. Override with: export MAX_JOBS=N${NC}"
    else
        # For CPU builds, use more parallelism
        export MAX_JOBS=8
        echo -e "${YELLOW}Setting MAX_JOBS=8 for CPU build. Override with: export MAX_JOBS=N${NC}"
    fi
else
    echo -e "${GREEN}Using MAX_JOBS=$MAX_JOBS (set by environment)${NC}"
fi
uv pip install -e "$vllm_fork_dir"

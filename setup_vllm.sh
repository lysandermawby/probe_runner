#!/bin/bash

: << EOF
This script is used to setup the vLLM fork as a submodule.

The vLLM fork is cloned as an editable package. 
Various changes are made to allow internal states to be extracted.
EOF

# fail upon error
set -e

vllm_fork_url="https://github.com/lysandermawby/vllm.git"
vllm_fork_dir="vllm"

# check if the vllm fork directory exists
if [ -d "$vllm_fork_dir" ]; then
    echo "The vllm fork directory already exists. Please remove it before running this script."
    rm -rf "$vllm_fork_dir"
fi

# clone the vllm fork
git clone "$vllm_fork_url" "$vllm_fork_dir" 

# install as editable package
uv pip install -e "$vllm_fork_dir"

#!/bin/bash

: << EOF
This script syncs the vLLM fork with the upstream vLLM repository.

This syncs the fork with upstream, merges changes with local edits, and pushes back.
EOF

# fail upon error
set -e

# Colour Variables
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
NC="\033[0m"

vllm_dir="vllm"
upstream_repo="https://github.com/vllm-project/vllm.git"
upstream_branch="main"
github_fork="lysandermawby/vllm"
github_sync_branch="main"

# Check if vllm directory exists and is a git repository
if [ ! -d "$vllm_dir" ]; then
    echo -e "${RED}vLLM directory not found${NC}"
    echo -e "${YELLOW}   Run setup.sh first to clone the vLLM fork${NC}"
    exit 1
fi

cd "$vllm_dir"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo -e "${RED}vLLM directory is not a git repository${NC}"
    echo -e "${YELLOW}   Run setup.sh first to clone the vLLM fork${NC}"
    exit 1
fi

# Check current branch
current_branch=$(git branch --show-current 2>/dev/null || echo "")
if [ -z "$current_branch" ]; then
    echo -e "${YELLOW}Warning: You're in a detached HEAD state${NC}"
    echo -e "${YELLOW}Creating and checking out 'main' branch...${NC}"
    git checkout -b main 2>/dev/null || git checkout main
    current_branch="main"
fi

echo -e "${GREEN}Current branch: $current_branch${NC}"

# Add upstream remote if it doesn't exist
if ! git remote | grep -q "^upstream$"; then
    echo -e "${GREEN}Adding upstream remote: $upstream_repo${NC}"
    git remote add upstream "$upstream_repo"
else
    echo -e "${GREEN}Upstream remote already exists${NC}"
    git remote set-url upstream "$upstream_repo"
fi

# Fetch from upstream
echo -e "${GREEN}Fetching latest from upstream...${NC}"
git fetch upstream "$upstream_branch"

# Show what files you've modified locally
echo ""
echo -e "${YELLOW}Files you've modified locally (will be preserved):${NC}"
git diff --name-only upstream/$upstream_branch...HEAD 2>/dev/null | head -20 || echo "  (none detected)"
echo ""

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo -e "${YELLOW}You have uncommitted changes. Stashing them...${NC}"
    git stash push -m "Stashed before upstream sync $(date +%Y-%m-%d)"
    stashed=true
else
    stashed=false
fi

# Sync fork on GitHub
echo -e "${GREEN}Syncing fork on GitHub with gh repo sync...${NC}"
if ! command -v gh >/dev/null 2>&1; then
    echo -e "${RED}gh CLI not found. Please install it: https://cli.github.com/${NC}"
    exit 1
fi

if ! gh repo sync "$github_fork" -b "$github_sync_branch"; then
    echo -e "${RED}gh repo sync failed${NC}"
    if [ "$stashed" = true ]; then
        git stash pop
    fi
    exit 1
fi

echo -e "${GREEN}Successfully synced fork with upstream${NC}"

# Pull synced changes from fork and merge with local edits
echo -e "${GREEN}Pulling synced changes from fork...${NC}"
git fetch origin "$github_sync_branch" >/dev/null 2>&1

echo -e "${GREEN}Merging origin/$github_sync_branch into $current_branch...${NC}"
echo -e "${YELLOW}   (This preserves your local edits while including all upstream changes)${NC}"

if ! git merge origin/"$github_sync_branch" --no-edit; then
    echo ""
    echo -e "${RED}Merge conflicts detected!${NC}"
    echo ""
    echo -e "${YELLOW}Conflicted files:${NC}"
    git diff --name-only --diff-filter=U
    if [ "$stashed" = true ]; then
        echo -e "${YELLOW}Restoring stashed changes...${NC}"
        git stash pop
    fi
    exit 1
fi

# Push merged changes back to fork
echo -e "${GREEN}Pushing merged changes to fork...${NC}"
if ! git push origin "$current_branch"; then
    echo -e "${RED}Push failed${NC}"
    if [ "$stashed" = true ]; then
        git stash pop
    fi
    exit 1
fi

echo -e "${GREEN}Successfully pushed to fork${NC}"

# Restore stashed changes if any
if [ "$stashed" = true ]; then
    echo -e "${GREEN}Restoring stashed changes...${NC}"
    git stash pop || echo "  (No conflicts with stashed changes)"
fi

echo ""
echo -e "${GREEN}Successfully synced with upstream!${NC}"
echo ""
echo -e "${YELLOW}Summary of changes (your local commits):${NC}"
git log --oneline upstream/$upstream_branch..HEAD | head -10
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Review the changes: git log upstream/$upstream_branch..HEAD"
echo "  2. Test your changes"
echo "  3. Update submodule reference in main repo:"
echo "     cd .. && git add vllm && git commit -m 'Update vLLM submodule'"

cd ..

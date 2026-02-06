#!/bin/bash
# ============================================================================
# CktGen Environment Setup Script
# 
# Usage: Source this file from any script using:
#   source "$(dirname "$0")/../../setup_env.sh"   # From scripts/test/xxx/
#   source "$(dirname "$0")/../setup_env.sh"      # From scripts/test/
#
# Features:
#   - Auto-detect project root directory
#   - Set PYTHONPATH
#   - Change to project root directory
# ============================================================================

# Get the directory where this script is located (scripts/)
SETUP_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Project root is one level up from scripts/
export PROJECT_ROOT=$(cd "$SETUP_SCRIPT_DIR/.." && pwd)

# Set PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Change to project root to ensure relative paths work correctly
cd $PROJECT_ROOT

# Print environment info
echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Working directory: $(pwd)"

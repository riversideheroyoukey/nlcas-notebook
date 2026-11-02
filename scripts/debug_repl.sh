#!/usr/bin/env bash
# Debug REPL script for Research Notebook

# Exit on error
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$DIR" )"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$PROJECT_ROOT"

python -m ui.debug_repl "$@"

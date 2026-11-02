#!/usr/bin/env bash
# Start script for Research Notebook

# Exit on error
set -e

# Get the directory of this script, and then the parent directory (project root)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$DIR" )"

cd "$PROJECT_ROOT"

echo "Starting Research Notebook..."

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: .venv not found. Running with global python."
fi

# Export PYTHONPATH so the notebook module is discoverable if needed
export PYTHONPATH="$PROJECT_ROOT"

# Start the uvicorn server
echo "Starting UI server..."
python -m uvicorn ui.app:app --host 127.0.0.1 --port 8000

#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "Building Julia Custom Sysimage for Research Notebook..."
echo "This process typically takes 5 to 10 minutes depending on your CPU."
echo "Installing PackageCompiler if not present and starting the build..."

julia scripts/build_julia_sysimage.jl

echo "Build complete! The JuliaAdapter will automatically use the new sysimage on next launch."

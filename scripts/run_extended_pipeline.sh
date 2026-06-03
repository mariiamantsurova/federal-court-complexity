#!/usr/bin/env bash
# Alias for run_pipeline.sh (referenced in README).
exec "$(dirname "${BASH_SOURCE[0]}")/run_pipeline.sh" "$@"

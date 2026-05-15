#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8002}

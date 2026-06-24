#!/usr/bin/env bash
set -euo pipefail

python -m gutbrainie.cli run-ner-transformer "$@"

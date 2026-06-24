#!/usr/bin/env bash
set -euo pipefail

python -m gutbrainie.cli predict-ner-dictionary "$@"

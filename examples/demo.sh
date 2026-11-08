#!/usr/bin/env bash
set -euo pipefail
python -m mixturelab report
python -m mixturelab controls
python -m mixturelab audit

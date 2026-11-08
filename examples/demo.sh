#!/usr/bin/env bash
set -euo pipefail
python -m mixturelab report
python -m mixturelab controls
python -m mixturelab audit

python -m mixturelab condition --input examples/conditional.json > /tmp/mixturelab-conditional.json
python -m mixturelab conditional-controls > /tmp/mixturelab-conditional-controls.json

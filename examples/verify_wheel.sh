#!/usr/bin/env bash
set -euo pipefail
installed=$1
source_root=$2
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export PYTHONPATH="$installed"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python - <<'PY'
import importlib,pkgutil,mixturelab
mods=sorted(m.name for m in pkgutil.walk_packages(mixturelab.__path__,mixturelab.__name__+'.'))
for m in mods:importlib.import_module(m)
assert len(mods)==9,mods
print('all ten package modules including root imported outside checkout')
PY
python -m mixturelab report > report-summary.json
python -m mixturelab controls > controls.json
python -m mixturelab fit --output fresh.json
python -m mixturelab audit > saved-audit.json
python -m mixturelab audit --report fresh.json > fresh-audit.json
python - <<'PY'
import json
from mixturelab.dataset import load
json.dump({'samples':load()[:2].tolist()},open('samples.json','w'))
PY
python -m mixturelab predict --input samples.json > predictions.json
python - <<'PYONE'
import json
from mixturelab.dataset import load
json.dump({'samples':[load()[0].tolist()]},open('one.json','w'))
PYONE
python -m mixturelab geometry --input one.json > geometry.json
python -m mixturelab ascend --input one.json --budget 16 > ascent.json
python - <<'PYGEOM'
import json
assert len(json.load(open('geometry.json'))['geometry']['hessian'])==13
r=json.load(open('ascent.json'))['trace'];assert r['updates']<=16
assert all(a['actual_gain']>=a['required_gain'] and a['actual_gain']>0 for a in r['history'][1:])
print('installed input geometry and sufficient-ascent witnesses pass')
PYGEOM
python - <<'PY'
import json
import numpy as np
from mixturelab.experiment import saved
from mixturelab.core import Model,expectation
from mixturelab.dataset import load
fresh=json.load(open('fresh.json'));original=saved()
assert fresh['selected_id']==original['selected_id']
for a,b in zip(fresh['candidates'],original['candidates']):
 assert a['id']==b['id'] and a['status']==b['status']
 for key in ('train_nll','validation_nll','test_nll'):
  assert abs(a[key]-b[key])<1e-6,(a['id'],key,a[key],b[key])
assert json.load(open('fresh-audit.json'))['ok']
assert json.load(open('saved-audit.json'))['candidates']==24
p=json.load(open('predictions.json'));fit=next(a for a in original['candidates'] if a['id']==original['selected_id'])
ll,r=expectation((load()[:2]-np.array(original['mean']))/np.array(original['scale']),Model.from_dict(fit['model']))
np.testing.assert_allclose(ll,p['log_density_standardized'],atol=1e-12)
np.testing.assert_allclose(r,p['responsibilities'],atol=1e-12)
print('all seven installed CLI commands, 24 fresh refits, holdout, covariance and identity audits pass')
PY
printf '%s\n' '{"samples":[[NaN]]}' > invalid.json
if python -m mixturelab predict --input invalid.json > invalid-output.json 2> invalid-error.txt; then
 echo 'invalid input unexpectedly passed';exit 1
else
 result=$?;test "$result" -eq 2;test ! -s invalid-output.json
fi

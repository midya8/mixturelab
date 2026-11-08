"""Frozen protocol, full candidates, train-only restart choice, validation choice."""
import json
from importlib.resources import files
from hashlib import sha256
from pathlib import Path
import numpy as np
from .core import Model, em, expectation, score
from .dataset import load, protocol, split, scaler, FEATURES, DATA_HASH
from .controls import controls


def run():
    p = protocol()
    parts = split()
    mean, scale = scaler(parts['train'])
    z = {name: (a-mean)/scale for name, a in parts.items()}
    candidates = []
    # No validation or test samples reach em(). Freeze all parameters first.
    for family in p['families']:
        for k in p['components']:
            for seed in p['seeds']:
                fit = em(z['train'], k, family, p['covariance_floor'], seed, p['max_iter'], p['tolerance_mean_log_likelihood'])
                fit.update(id=f'{family}-{k}-{seed}', family=family, k=k, seed=seed)
                fit['train_nll'] = score(z['train'], Model.from_dict(fit['model']))
                candidates.append(fit)
    configs = []
    for family in p['families']:
        for k in p['components']:
            group = [a for a in candidates if a['family']==family and a['k']==k]
            winner = min(group, key=lambda a: (a['train_nll'],a['seed']))
            model = Model.from_dict(winner['model'])
            configs.append(dict(id=winner['id'], family=family, k=k, seed=winner['seed'], train_nll=winner['train_nll'], validation_nll=score(z['validation'], model)))
    chosen = min(configs,key=lambda a:(a['validation_nll'],a['family'],a['k']))['id']
    # Only now open test scores, for every candidate rather than only the winner.
    for fit in candidates:
        model = Model.from_dict(fit['model'])
        fit['validation_nll'] = score(z['validation'],model)
        fit['test_nll'] = score(z['test'],model)
        fit['test_nll_original_units'] = fit['test_nll']+float(np.log(scale).sum())
    for row in configs:
        row['test_nll']=next(a['test_nll'] for a in candidates if a['id']==row['id'])
    return dict(version=1, protocol=p, protocol_sha256=sha256(files('mixturelab').joinpath('artifacts/protocol.json').read_bytes()).hexdigest(), source_sha256=DATA_HASH, sample_counts={k:len(v) for k,v in z.items()}, mean=mean.tolist(), scale=scale.tolist(), original_unit_log_jacobian=float(np.log(scale).sum()), candidates=candidates, configurations=configs, selected_id=chosen, controls=controls())


def saved():
    return json.loads(files('mixturelab').joinpath('artifacts/report.json').read_text())


def export_browser(report):
    raw=load()
    train=split(raw)['train']
    mean=np.array(report['mean']); scale=np.array(report['scale'])
    z=(raw-mean)/scale
    train_z=(train-mean)/scale
    # PCA is trained on training rows only and is for drawing, not fitting.
    _,_,vt=np.linalg.svd(train_z,full_matrices=False)
    projection=vt[:2].copy()
    for row in projection:
        if row[np.argmax(abs(row))]<0: row *= -1
    models=[]
    references=[]
    for conf in report['configurations']:
        fit=next(a for a in report['candidates'] if a['id']==conf['id'])
        m=Model.from_dict(fit['model'])
        logs,r=expectation(z,m)
        models.append(dict(**conf,model=fit['model'],history=fit['history'],status=fit['status'],iterations=fit['iterations'],projected_means=(m.means@projection.T).tolist(),projected_covariances=[(projection@cov@projection.T).tolist() for cov in m.covariances]))
        references.append(dict(id=conf['id'],log_density=logs.tolist(),responsibilities=r.tolist()))
    restarts=[{key:a[key] for key in ('id','family','k','seed','train_nll','validation_nll','test_nll','iterations','status')} for a in report['candidates']]
    return dict(version=1,restarts=restarts,source_sha256=DATA_HASH,features=FEATURES,raw=raw.tolist(),split_ids=report['protocol']['split_ids'],mean=report['mean'],scale=report['scale'],projection=projection.tolist(),projected=(z@projection.T).tolist(),models=models,selected_id=report['selected_id'],controls=report['controls'],original_unit_log_jacobian=report['original_unit_log_jacobian']), references


def build(root):
    root=Path(root)
    r=run()
    (root/'mixturelab/artifacts/report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    data,refs=export_browser(r)
    (root/'docs/data.js').write_text('globalThis.MIXTURE_DATA = '+json.dumps(data,separators=(',',':'),allow_nan=False)+';\n')
    (root/'mixturelab/artifacts/browser-reference.json').write_text(json.dumps(refs,separators=(',',':'),allow_nan=False)+'\n')
    print('selected',r['selected_id'])
    for row in r['configurations']: print(row)
    print('statuses',[(a['id'],a['status'],a['iterations']) for a in r['candidates']])
    return r

if __name__=='__main__':
    build(Path(__file__).resolve().parents[1])

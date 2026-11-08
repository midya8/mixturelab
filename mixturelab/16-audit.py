"""Independent saved-artifact and final-transition checks, not a fit identity proof."""
import numpy as np
from .core import Model, expectation, component_logs, maximization, score
from .dataset import protocol, split, scaler


def audit(report):
    p=protocol()
    import math
    from hashlib import sha256
    from importlib.resources import files
    from .dataset import DATA_HASH
    def finite(v):
        if isinstance(v,float) and not math.isfinite(v):raise ValueError('nonfinite report')
        if isinstance(v,dict):
            for item in v.values():finite(item)
        if isinstance(v,list):
            for item in v:finite(item)
    finite(report)
    if report['source_sha256'] != DATA_HASH or report['protocol_sha256'] != sha256(files('mixturelab').joinpath('artifacts/protocol.json').read_bytes()).hexdigest():
        raise ValueError('source/protocol hash differs')
    if report['protocol'] != p:
        raise ValueError('protocol differs')
    parts=split();mean,scale=scaler(parts['train'])
    if not np.allclose(report['mean'],mean,atol=1e-12,rtol=0) or not np.allclose(report['scale'],scale,atol=1e-12,rtol=0):
        raise ValueError('training transform differs')
    z={name:(a-mean)/scale for name,a in parts.items()}
    expected={(f,k,s) for f in p['families'] for k in p['components'] for s in p['seeds']}
    actual={(a['family'],a['k'],a['seed']) for a in report['candidates']}
    if actual != expected or len(report['candidates'])!=len(expected):
        raise ValueError('candidate grid incomplete')
    maximum=0.
    for fit in report['candidates']:
        m=Model.from_dict(fit['model'])
        if fit['id']!=f"{fit['family']}-{fit['k']}-{fit['seed']}" or m.family!=fit['family'] or len(m.weights)!=fit['k'] or m.floor!=p['covariance_floor']:
            raise ValueError('candidate identity differs')
        for name in z:
            if abs(score(z[name],m)-fit[name+'_nll'])>1e-9:
                raise ValueError('saved score differs')
        if abs(fit['test_nll_original_units']-fit['test_nll']-np.log(scale).sum())>1e-9:
            raise ValueError('unit Jacobian differs')
        h=fit['history']
        if fit['iterations']!=len(h)-1 or abs(h[-1]['mean_log_likelihood']+fit['train_nll'])>1e-9:
            raise ValueError('history endpoint differs')
        if abs(h[0]['mean_log_likelihood']+score(z['train'],Model.from_dict(fit['initial'])))>1e-9:
            raise ValueError('initial objective differs')
        for i,row in enumerate(h):
            if row['iteration'] != i or row['min_eigenvalue']<p['covariance_floor']-1e-9 or row['min_mass']<=0:
                raise ValueError('history constraint differs')
            if i:
                if min(row['gain'],row['q_gain'],row['posterior_kl'])< -1e-9 or abs(h[i]['mean_log_likelihood']-h[i-1]['mean_log_likelihood']-row['gain'])>1e-9 or abs(row['gain']-row['q_gain']-row['posterior_kl'])>1e-8:
                    raise ValueError('history identity differs')
        w=fit['last_transition']
        if w:
            before=Model.from_dict(w['before']);after=Model.from_dict(w['after'])
            ll,r=expectation(z['train'],before)
            rebuilt,_=maximization(z['train'],r,m.family,m.floor)
            for name in ('weights','means','covariances'):
                if not np.allclose(getattr(rebuilt,name),getattr(after,name),atol=1e-9,rtol=1e-9) or not np.allclose(getattr(after,name),getattr(m,name),atol=1e-12,rtol=0):
                    raise ValueError('final transition differs')
            newll,_=expectation(z['train'],after)
            oldlogs=component_logs(z['train'],before);newlogs=component_logs(z['train'],after)
            gain=np.mean(newll-ll);q=np.sum(r*(newlogs-oldlogs))/len(r);kl=np.sum(r*((oldlogs-ll[:,None])-(newlogs-newll[:,None])))/len(r)
            err=float(abs(gain-q-kl));maximum=max(maximum,err)
            for key,value in [('gain',gain),('q_gain',q),('posterior_kl',kl)]:
                if abs(h[-1][key]-value)>1e-9:raise ValueError('final identity differs')
        expected_status='likelihood_increment_tolerance' if len(h)>1 and h[-1]['gain']<=p['tolerance_mean_log_likelihood'] else 'budget_exhausted'
        if fit['status']!=expected_status:raise ValueError('stopping status differs')
    configs=[]
    for f in p['families']:
        for k in p['components']:
            winner=min((a for a in report['candidates'] if a['family']==f and a['k']==k),key=lambda a:(a['train_nll'],a['seed']))
            configs.append(winner)
    if [a['id'] for a in configs]!=[a['id'] for a in report['configurations']]:
        raise ValueError('restart selection differs')
    for original,row in zip(configs,report['configurations']):
        for key in ('id','family','k','seed','train_nll','validation_nll','test_nll'):
            if row[key]!=original[key]:raise ValueError('configuration row differs')
    chosen=min(configs,key=lambda a:(a['validation_nll'],a['family'],a['k']))['id']
    if chosen!=report['selected_id']:raise ValueError('validation selection differs')
    return dict(ok=True,candidates=len(expected),configurations=len(configs),selected_id=chosen,maximum_final_identity_error=maximum,scope='scores, stored history identities, final E/M transition, constraint and selection checks; not parameter provenance or global optimality')

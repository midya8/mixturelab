"""Strict JSON interface. No network or training API access."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from .core import Model, expectation
from .experiment import run, saved
from .controls import controls
from .audit import audit


def read_json(path):
    def reject(value):
        raise ValueError('JSON constants must be finite: '+value)
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key: '+key)
            result[key]=value
        return result
    return json.loads(Path(path).read_text(),parse_constant=reject,object_pairs_hook=pairs)


def main():
    parser=argparse.ArgumentParser(description='Constrained Gaussian-mixture EM, offline and explicit')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('report')
    sub.add_parser('controls')
    fit=sub.add_parser('fit');fit.add_argument('--output',required=True)
    check=sub.add_parser('audit');check.add_argument('--report')
    predict=sub.add_parser('predict');predict.add_argument('--input',required=True);predict.add_argument('--model-id');predict.add_argument('--report')
    args=parser.parse_args()
    try:
        if args.command=='report':
            r=saved();out={k:r[k] for k in ('sample_counts','configurations','selected_id','original_unit_log_jacobian')}
        elif args.command=='controls':out=controls()
        elif args.command=='fit':
            r=run();Path(args.output).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');out=dict(output=args.output,selected_id=r['selected_id'],candidates=len(r['candidates']))
        elif args.command=='audit':out=audit(read_json(args.report) if args.report else saved())
        else:
            r=read_json(args.report) if args.report else saved()
            model_id=args.model_id or r['selected_id']
            fit=next((f for f in r['candidates'] if f['id']==model_id),None)
            if fit is None:raise ValueError('unknown model id')
            raw=read_json(args.input)
            if not isinstance(raw,dict) or set(raw)!={'samples'}:raise ValueError('input must contain only samples')
            from .core import matrix
            samples=matrix(raw['samples'])
            if samples.shape[1]!=len(r['mean']):raise ValueError('expected 13 source features')
            z=(samples-np.array(r['mean']))/np.array(r['scale'])
            logs,resp=expectation(z,Model.from_dict(fit['model']))
            out=dict(model_id=model_id,log_density_standardized=logs.tolist(),log_density_original_units=(logs-r['original_unit_log_jacobian']).tolist(),responsibilities=resp.tolist(),scope='posterior component probabilities within this fitted density, not cultivar probabilities or calibrated confidence')
        print(json.dumps(out,indent=2,allow_nan=False))
    except (ValueError,KeyError,TypeError,ArithmeticError,OSError,np.linalg.LinAlgError) as exc:
        print('mixturelab: '+str(exc),file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':raise SystemExit(main())

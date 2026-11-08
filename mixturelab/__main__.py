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
from .geometry import geometry_raw, trace


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
    for command in ('geometry','ascend'):
        calc=sub.add_parser(command);calc.add_argument('--input',required=True);calc.add_argument('--model-id');calc.add_argument('--report')
        if command=='ascend':calc.add_argument('--budget',type=int,default=128)
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
            if not isinstance(raw['samples'],list) or any(not isinstance(row,list) or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in row) for row in raw['samples']):raise ValueError('samples must be arrays of JSON numbers, not booleans or strings')
            from .core import matrix
            samples=matrix(raw['samples'])
            if samples.shape[1]!=len(r['mean']):raise ValueError('expected 13 source features')
            z=(samples-np.array(r['mean']))/np.array(r['scale'])
            if args.command in ('geometry','ascend'):
                if len(samples)!=1:raise ValueError('geometry/ascend requires exactly one input row')
                m=Model.from_dict(fit['model'])
                out=dict(model_id=model_id,geometry=geometry_raw(samples[0],m,r['mean'],r['scale']))
                if args.command=='ascend':out['trace']=trace(z[0],m,args.budget)
                print(json.dumps(out,indent=2,allow_nan=False));return 0
            logs,resp=expectation(z,Model.from_dict(fit['model']))
            out=dict(model_id=model_id,log_density_standardized=logs.tolist(),log_density_original_units=(logs-r['original_unit_log_jacobian']).tolist(),responsibilities=resp.tolist(),scope='posterior component probabilities within this fitted density, not cultivar probabilities or calibrated confidence')
        print(json.dumps(out,indent=2,allow_nan=False))
    except (ValueError,KeyError,TypeError,ArithmeticError,OSError,np.linalg.LinAlgError) as exc:
        print('mixturelab: '+str(exc),file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':raise SystemExit(main())

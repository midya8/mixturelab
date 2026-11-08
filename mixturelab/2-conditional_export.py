"""Reference export only: no original artifacts, parameters or scores edited."""
import json
from pathlib import Path
import numpy as np
from .core import Model
from .experiment import saved
from .dataset import load
from .conditional import condition_raw, log_density, controls


def build(root):
    root=Path(root);report=saved();raw=load();references=[]
    for config in report['configurations']:
        m=Model.from_dict(next(c for c in report['candidates']if c['id']==config['id'])['model'])
        for row in (0,35,87,142,177):
            for obs in ([],list(range(13)),[0,2,4,6,8,10,12],[12,1,5]):
                r=condition_raw(m,obs,raw[row,obs],report['mean'],report['scale'])
                target=raw[row,r['free']].tolist()
                references.append(dict(id=config['id'],row=row,conditional=r,target=target,log_density=float(log_density(r,[target])[0])))
    result=dict(version=1,references=references,controls=controls())
    (root/'mixturelab/artifacts/conditional-reference.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
    (root/'docs/conditional-data.js').write_text('globalThis.CONDITIONAL_DATA = '+json.dumps(dict(controls=result['controls']),separators=(',',':'),allow_nan=False)+';\n')
    return result

if __name__=='__main__':build(Path(__file__).resolve().parents[1])

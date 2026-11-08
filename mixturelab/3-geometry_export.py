"""Reference export only. Never refits or edits v0.1 artifacts."""
import json
from pathlib import Path
import numpy as np
from .experiment import saved
from .dataset import load
from .core import Model
from .geometry import geometry_raw,trace,controls


def build(root):
    root=Path(root);report=saved();raw=load();mean=np.array(report['mean']);scale=np.array(report['scale']);refs=[]
    for config in report['configurations']:
        model=Model.from_dict(next(a for a in report['candidates'] if a['id']==config['id'])['model'])
        for row in (0,35,87,142,177):
            refs.append(dict(id=config['id'],row=row,geometry=geometry_raw(raw[row],model,mean,scale)))
    traces=[]
    for config in report['configurations']:
        model=Model.from_dict(next(a for a in report['candidates'] if a['id']==config['id'])['model']);row=report['protocol']['split_ids']['test'][0]
        traces.append(dict(id=config['id'],row=row,trace=trace((raw[row]-mean)/scale,model,128)))
    result=dict(version=1,references=refs,traces=traces,controls=controls())
    (root/'mixturelab/artifacts/geometry-reference.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
    (root/'docs/geometry-data.js').write_text('globalThis.GEOMETRY_DATA = '+json.dumps(dict(controls=result['controls']),separators=(',',':'),allow_nan=False)+';\n')
    print([(a['id'],a['trace']['status'],a['trace']['updates']) for a in traces])
    return result

if __name__=='__main__':build(Path(__file__).resolve().parents[1])

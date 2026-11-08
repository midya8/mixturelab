import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from mixturelab.dataset import load,split,protocol,scaler,FEATURES
from mixturelab.core import Model,expectation
from mixturelab.experiment import run,saved,export_browser
from mixturelab.audit import audit


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.report=saved()

    def test_source_and_label_free_fixed_split(self):
        x=load();p=protocol();self.assertEqual(x.shape,(178,13));self.assertEqual(len(FEATURES),13)
        ids=[i for v in p['split_ids'].values() for i in v]
        self.assertEqual(sorted(ids),list(range(178)));self.assertEqual(len(np.unique(x,axis=0)),178)
        self.assertEqual([len(v) for v in split().values()],[106,36,36])

    def test_scaler_uses_training_rows_only(self):
        parts=split();mean,scale=scaler(parts['train'])
        np.testing.assert_allclose(mean,self.report['mean'],atol=1e-14)
        np.testing.assert_allclose(scale,self.report['scale'],atol=1e-14)
        self.assertGreater(np.max(abs(load().mean(axis=0)-mean)),1)
        z=(parts['train']-mean)/scale
        np.testing.assert_allclose(z.mean(axis=0),0,atol=1e-14);np.testing.assert_allclose(z.std(axis=0),1,atol=1e-14)

    def test_fresh_all_24_refits_numerical_reproduction(self):
        fresh=run();self.assertEqual(fresh['selected_id'],self.report['selected_id'])
        self.assertTrue(audit(fresh)['ok'])
        for old,new in zip(self.report['candidates'],fresh['candidates']):
            self.assertEqual(old['id'],new['id']);self.assertEqual(old['status'],new['status'])
            for metric in ('train_nll','validation_nll','test_nll'):self.assertAlmostEqual(old[metric],new[metric],places=6)
            # Floating eigensolvers are not promised to reproduce exact bits.
            np.testing.assert_allclose(old['model']['means'],new['model']['means'],atol=2e-5,rtol=2e-5)

    def test_changing_holdout_cannot_change_fits_or_selection(self):
        # Poison test features while keeping the training and validation identical.
        parts=split();parts['test']=parts['test']+500
        with patch('mixturelab.experiment.split',return_value=parts):other=run()
        self.assertEqual(other['selected_id'],self.report['selected_id'])
        for a,b in zip(other['candidates'],self.report['candidates']):
            np.testing.assert_allclose(a['model']['means'],b['model']['means'],atol=1e-10)
            self.assertAlmostEqual(a['validation_nll'],b['validation_nll'],places=10)
            self.assertGreater(a['test_nll'],b['test_nll']+100)

    def test_audit_and_rejection_controls(self):
        self.assertEqual(audit(self.report)['candidates'],24)
        changes=[lambda r:r.update(selected_id='full-5-23'),lambda r:r['candidates'][0].update(test_nll=999),lambda r:r['candidates'][0]['history'][1].update(gain=-1),lambda r:r['configurations'][0].update(test_nll=999),lambda r:r['candidates'].pop(),lambda r:r['mean'].__setitem__(0,0),lambda r:r.update(source_sha256='bad'),lambda r:r['candidates'][0].update(train_nll=float('nan'))]
        for change in changes:
            r=copy.deepcopy(self.report);change(r)
            with self.assertRaises(ValueError):audit(r)

    def test_browser_export_projection_and_all_rows(self):
        d,refs=export_browser(self.report)
        self.assertEqual(len(d['models']),8);self.assertEqual(len(d['restarts']),24)
        p=np.array(d['projection']);np.testing.assert_allclose(p@p.T,np.eye(2),atol=1e-14)
        x=(load()-np.array(d['mean']))/np.array(d['scale'])
        np.testing.assert_allclose(x@p.T,d['projected'],atol=1e-14)
        for conf,ref in zip(d['models'],refs):
            m=Model.from_dict(conf['model']);ll,r=expectation(x,m)
            np.testing.assert_array_equal(ll,ref['log_density']);np.testing.assert_array_equal(r,ref['responsibilities'])
            for cov,projected in zip(m.covariances,conf['projected_covariances']):np.testing.assert_allclose(p@cov@p.T,projected,atol=1e-14)
            self.assertEqual(np.array(ref['responsibilities']).shape,(178,conf['k']))

    def test_original_units_jacobian(self):
        r=self.report;parts=split();model=Model.from_dict(r['candidates'][0]['model']);scale=np.array(r['scale']);mean=np.array(r['mean'])
        # Transform each Gaussian back to source units and evaluate directly.
        rawmodel=Model(model.weights,model.means*scale+mean,model.covariances*scale[:,None]*scale[None,:],model.family,1e-10)
        a,_=expectation((parts['test']-mean)/scale,model);b,_=expectation(parts['test'],rawmodel)
        np.testing.assert_allclose(b,a-r['original_unit_log_jacobian'],atol=1e-11)

    def test_installed_json_cli_and_strict_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'input.json';p.write_text(json.dumps({'samples':[load()[0].tolist()]}))
            a=subprocess.run([sys.executable,'-m','mixturelab','predict','--input',str(p)],capture_output=True,text=True)
            self.assertEqual(a.returncode,0,a.stderr);obj=json.loads(a.stdout);self.assertEqual(obj['model_id'],self.report['selected_id']);self.assertEqual(len(obj['responsibilities'][0]),5)
            for bad in ('{"samples":[[NaN]]}','{"samples":[],"samples":[]}','{"samples":[[1,2]]}','{"samples":[[1]],"unrelated":true}'):
                p.write_text(bad);a=subprocess.run([sys.executable,'-m','mixturelab','predict','--input',str(p)],capture_output=True,text=True)
                self.assertEqual(a.returncode,2);self.assertEqual(a.stdout,'');self.assertIn('mixturelab:',a.stderr)

if __name__=='__main__':unittest.main()

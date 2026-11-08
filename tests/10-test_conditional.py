import copy
import math
import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from mixturelab.core import Model, expectation
from mixturelab.conditional import condition, condition_raw, log_density, controls
from mixturelab.experiment import saved
from mixturelab.dataset import load


class ConditionalTests(unittest.TestCase):
    def test_independent_scalar_two_dimensional_gaussian(self):
        m=Model(np.array([1.]),np.array([[2.,-1.]]),np.array([[[4.,1.5],[1.5,3.]]]),'full',.01)
        for x in (-5.,0.,2.,9.):
            r=condition(m,[0],[x]);mu=-1+1.5/4*(x-2);variance=3-1.5**2/4
            self.assertAlmostEqual(r['mean'][0],mu,places=14)
            self.assertAlmostEqual(r['covariance'][0][0],variance,places=14)
            for y in (-7.,0.,mu,3.):
                direct=-.5*(math.log(2*math.pi*variance)+(y-mu)**2/variance)
                self.assertAlmostEqual(float(log_density(r,[[y]])[0]),direct,places=13)

    def test_independent_precision_block_oracle(self):
        rng=np.random.default_rng(901)
        for _ in range(100):
            a=rng.normal(size=(6,6));cov=a@a.T+np.eye(6)*.2;mu=rng.normal(size=6)
            m=Model(np.array([1.]),mu[None],cov[None],'full',.01)
            obs=rng.choice(6,size=3,replace=False).tolist();free=[i for i in range(6)if i not in obs];values=rng.normal(size=3)
            precision=np.linalg.inv(cov);pmm=precision[np.ix_(free,free)];pmo=precision[np.ix_(free,obs)]
            expected_cov=np.linalg.inv(pmm);expected_mean=mu[free]-np.linalg.solve(pmm,pmo@(values-mu[obs]))
            r=condition(m,obs,values)
            np.testing.assert_allclose(r['mean'],expected_mean,atol=2e-13,rtol=2e-12)
            np.testing.assert_allclose(r['covariance'],expected_cov,atol=2e-13,rtol=2e-12)

    def test_full_original_models_rows_masks_and_factorization(self):
        report=saved();z=(load()-report['mean'])/report['scale'];checked=0
        masks=[[],list(range(13)),[0,2,4,6,8,10,12],[12,1,5]]
        for config in report['configurations']:
            m=Model.from_dict(next(c for c in report['candidates']if c['id']==config['id'])['model'])
            joint=expectation(z,m)[0]
            for row,x in enumerate(z):
                for obs in masks:
                    r=condition(m,obs,x[obs]);free=r['free'];ll=float(log_density(r,[x[free]])[0])
                    self.assertAlmostEqual(joint[row],r['log_evidence']+ll,delta=2e-10)
                    self.assertAlmostEqual(sum(r['weights']),1.,delta=2e-13)
                    if free:
                        within=np.array(r['within_covariance']);between=np.array(r['between_covariance']);total=np.array(r['covariance'])
                        np.testing.assert_allclose(total,within+between,atol=1e-13)
                        # Independent raw second-moment summation, not the centered implementation.
                        mu=np.array(r['mean']);cm=np.array(r['means']);cc=np.array(r['covariances']);w=r['weights']
                        second=np.array([[sum(w[j]*(cc[j,a,b]+cm[j,a]*cm[j,b])for j in range(len(w)))-mu[a]*mu[b]for b in range(len(free))]for a in range(len(free))])
                        np.testing.assert_allclose(total,second,atol=8e-13,rtol=2e-11)
                        self.assertGreater(np.linalg.eigvalsh(total).min(),0)
                        self.assertGreaterEqual(np.linalg.eigvalsh(between).min(),-1e-12)
                    checked+=1
        self.assertEqual(checked,5696)

    def test_empty_limits_and_observed_order(self):
        m=Model(np.array([.3,.7]),np.array([[0.,1.,2.],[3.,4.,5.]]),np.array([np.eye(3),np.eye(3)*2]),'full',.01)
        r=condition(m,[],[]);self.assertEqual(r['log_evidence'],0.)
        np.testing.assert_allclose(r['weights'],m.weights,atol=1e-15)
        np.testing.assert_allclose(log_density(r,[[0.,2.,3.]]),expectation([[0.,2.,3.]],m)[0],atol=1e-14)
        full=condition(m,[2,0,1],[5.,3.,4.]);self.assertEqual(full['free'],[]);self.assertEqual(full['mean'],[])
        np.testing.assert_array_equal(log_density(full,[[],[]]),[0.,0.])
        a=condition(m,[2,0],[6.,2.]);b=condition(m,[0,2],[2.,6.])
        for key in ['weights','mean','covariance','within_covariance','between_covariance']:
            np.testing.assert_allclose(a[key],b[key],atol=1e-14)

    def test_original_units_jacobians_and_moments(self):
        m=Model(np.array([.4,.6]),np.array([[0.,1.,2.],[3.,4.,5.]]),np.array([[[2.,.3,.2],[.3,1.,.1],[.2,.1,3.]],np.eye(3)*2]),'full',.01)
        center=np.array([5.,-2.,10.]);scale=np.array([.1,3.,7.]);obs=[2,0];x=np.array([2.,4.]);raw=x*scale[obs]+center[obs]
        a=condition(m,obs,x);b=condition_raw(m,obs,raw,center,scale)
        np.testing.assert_allclose(b['mean'],np.array(a['mean'])*scale[1]+center[1],atol=1e-13)
        np.testing.assert_allclose(b['covariance'],np.array(a['covariance'])*scale[1]**2,atol=1e-12)
        self.assertAlmostEqual(a['log_evidence']-sum(np.log(scale[obs])),b['log_evidence'],places=13)
        np.testing.assert_allclose(log_density(b,[[9.]]),log_density(a,[[(9-center[1])/scale[1]]])-math.log(scale[1]),atol=1e-12)
        self.assertEqual(condition_raw(m,[0,1,2],center,center,scale)['free'],[])

    def test_diagonal_components_can_have_dependent_mixture_coordinates(self):
        c=controls();a=c['increasing_variance'];self.assertAlmostEqual(a['unconditional_target_variance'],1.09,places=13)
        self.assertAlmostEqual(a['conditional_target_variance'],25.1,places=12)
        self.assertGreater(a['conditional_target_variance'],a['unconditional_target_variance'])
        g=c['mean_in_gap'];self.assertEqual(g['conditional']['mean'],[0.])
        self.assertGreater(g['log_density_at_component_mean']-g['log_density_at_mean'],30.)

    def test_label_permutation_duplicate_split_and_no_mutation(self):
        m=Model(np.array([.3,.7]),np.array([[0.,1.],[3.,4.]]),np.array([[[2.,.3],[.3,1.]],np.eye(2)*2]),'full',.01)
        before=copy.deepcopy(m);a=condition(m,[0],[1.2]);p=Model(m.weights[::-1],m.means[::-1],m.covariances[::-1],'full',.01)
        b=condition(p,[0],[1.2]);split=Model(np.array([.1,.2,.7]),m.means[[0,0,1]],m.covariances[[0,0,1]],'full',.01);c=condition(split,[0],[1.2])
        for key in ['mean','covariance','within_covariance','between_covariance']:
            np.testing.assert_allclose(a[key],b[key],atol=1e-14);np.testing.assert_allclose(a[key],c[key],atol=1e-14)
        for key in ['weights','means','covariances']:np.testing.assert_array_equal(getattr(m,key),getattr(before,key))

    def test_numeric_scope_and_underflow_logweights(self):
        m=Model(np.array([.5,.5]),np.array([[0.,0.],[100.,100.]]),np.array([np.eye(2),np.eye(2)]),'full',.01)
        for obs,val in [([True],[0]),([0,0],[0,0]),([-1],[0]),([2],[0]),([0],[float('nan')]),([0],[]),([0],[1e308])]:
            with self.assertRaises((ValueError,np.linalg.LinAlgError)):condition(m,obs,val)
        r=condition(m,[0],[0.]);self.assertEqual(r['weights'][1],0.);self.assertTrue(math.isfinite(r['log_weights'][1]))
        self.assertTrue(np.isfinite(log_density(r,[[100.]])).all())
        with self.assertRaises(ValueError):condition_raw(m,[0],[0.],[0.,0.],[0.,1.])
        with self.assertRaises(ValueError):log_density(r,[[float('inf')]])

    def test_cli_scope_and_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'input.json'
            p.write_text(json.dumps(dict(observed=[2,0],values=[2.43,14.23])))
            result=subprocess.run([sys.executable,'-m','mixturelab','condition','--input',str(p)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            out=json.loads(result.stdout);self.assertEqual(out['conditional']['observed'],[2,0]);self.assertEqual(out['conditional']['coordinate_space'],'original')
            for bad in [dict(observed=[True],values=[1]),dict(observed=[0],values=[True]),dict(observed=[0,0],values=[1,2]),dict(observed=[0],values=[1],hidden=[8])]:
                p.write_text(json.dumps(bad));r=subprocess.run([sys.executable,'-m','mixturelab','condition','--input',str(p)],capture_output=True,text=True)
                self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')
            p.write_text('{"observed":[0],"observed":[1],"values":[0]}')
            r=subprocess.run([sys.executable,'-m','mixturelab','condition','--input',str(p)],capture_output=True,text=True)
            self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')

if __name__=='__main__':unittest.main()

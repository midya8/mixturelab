import copy
import math
import unittest
import numpy as np
from mixturelab.core import Model,expectation
from mixturelab.geometry import geometry,geometry_raw,trace,controls
from mixturelab.experiment import saved
from mixturelab.dataset import load

class GeometryTests(unittest.TestCase):
    def test_all_dimensions_against_independent_finite_differences(self):
        rng=np.random.default_rng(41);report=saved();probes=0
        for config in report['configurations']:
            fit=next(a for a in report['candidates'] if a['id']==config['id']);m=Model.from_dict(fit['model'])
            for _ in range(3):
                x=rng.normal(size=13);a=geometry(x,m);h=1e-5
                for j in range(13):
                    e=np.eye(13)[j]*h;lp,rp=expectation((x+e)[None],m);lm,rm=expectation((x-e)[None],m)
                    self.assertAlmostEqual((lp[0]-lm[0])/(2*h),a['gradient'][j],delta=3e-7)
                    np.testing.assert_allclose((rp[0]-rm[0])/(2*h),np.array(a['responsibility_jacobian'])[:,j],atol=3e-7,rtol=2e-6)
                    gp=geometry(x+e,m)['gradient'];gm=geometry(x-e,m)['gradient']
                    np.testing.assert_allclose((np.array(gp)-gm)/(2*h),np.array(a['hessian'])[:,j],atol=4e-6,rtol=2e-6)
                    probes+=1
        self.assertEqual(probes,312)

    def test_independent_directional_second_derivatives(self):
        rng=np.random.default_rng(17)
        for _ in range(50):
            means=rng.normal(size=(3,2));a=rng.normal(size=(3,2,2));cov=a@a.swapaxes(1,2)+np.eye(2)*.5
            model=Model(np.array([.2,.3,.5]),means,cov,'full',.05);x=rng.normal(size=2);v=rng.normal(size=2);v/=math.hypot(*v);g=geometry(x,model);h=2e-4
            lp=expectation((x+h*v)[None],model)[0][0];lm=expectation((x-h*v)[None],model)[0][0];l0=expectation(x[None],model)[0][0]
            self.assertAlmostEqual((lp-2*l0+lm)/h**2,float(v@np.array(g['hessian'])@v),delta=2e-6)

    def test_decomposition_normalization_and_affine_coordinates(self):
        report=saved();m=Model.from_dict(report['candidates'][-1]['model']);x=np.linspace(-1,1,13);g=geometry(x,m)
        H=np.array(g['hessian']);J=np.array(g['responsibility_jacobian']);V=np.array(g['posterior_gradient_covariance'])
        np.testing.assert_allclose(H,np.array(g['precision_term'])+V,atol=1e-14);np.testing.assert_allclose(H,H.T,atol=1e-13);np.testing.assert_allclose(J.sum(axis=0),0,atol=1e-13)
        self.assertGreaterEqual(np.linalg.eigvalsh(V).min(),-1e-12)
        mean=np.arange(13)*3.;scale=np.linspace(.1,10,13);raw=x*scale+mean;out=geometry_raw(raw,m,mean,scale)
        np.testing.assert_allclose(out['gradient_original'],np.array(g['gradient'])/scale,atol=1e-12)
        np.testing.assert_allclose(out['hessian_original'],H/scale[:,None]/scale[None,:],atol=2e-11)
        np.testing.assert_allclose(out['responsibility_jacobian_original'],J/scale,atol=1e-12)

    def test_translation_rotation_and_label_permutation(self):
        m=Model(np.array([.4,.6]),np.array([[-2.,1.],[1.,3.]]),np.array([[[1.,.3],[.3,2.]],[[2.,-.2],[-.2,.6]]]),'full',.05)
        x=np.array([.2,1.1]);R=np.array([[.6,-.8],[.8,.6]]);b=np.array([2.,4.]);q=Model(m.weights,(R@m.means.T).T+b,np.array([R@c@R.T for c in m.covariances]),'full',.05)
        a=geometry(x,m);c=geometry(R@x+b,q)
        np.testing.assert_allclose(c['gradient'],R@np.array(a['gradient']),atol=1e-13)
        np.testing.assert_allclose(c['hessian'],R@np.array(a['hessian'])@R.T,atol=1e-13)
        p=Model(m.weights[::-1],m.means[::-1],m.covariances[::-1],'full',.05);s=geometry(x,p)
        np.testing.assert_allclose(s['gradient'],a['gradient'],atol=1e-13);np.testing.assert_allclose(s['responsibility_jacobian'],a['responsibility_jacobian'][::-1],atol=1e-13)

    def test_stationary_minimum_is_not_a_mode_certificate(self):
        c=controls()['stationary_minimum'];a=c['geometry'];self.assertEqual(a['gradient'],[0.]);self.assertAlmostEqual(a['hessian'][0][0],3.,places=14)
        self.assertEqual(c['trace']['status'],'gradient_norm_criterion');self.assertEqual(c['trace']['updates'],0);self.assertGreater(c['trace']['final_hessian_eigenvalues'][0],0)
        m=Model.from_dict(c['model']);self.assertGreater(geometry([.01],m)['log_density'],a['log_density'])
        escaped=c['escaping_start'];self.assertGreater(escaped['final']['log_density'],a['log_density']+1)
        # At displacement 2 the local quadratic is a poor finite-change forecast.
        actual=geometry([2.],m)['log_density'];quadratic=a['log_density']+.5*3*4
        self.assertGreater(abs(actual-quadratic),1)

    def test_single_and_duplicate_gaussian_closed_forms(self):
        c=controls();m=Model.from_dict(c['single_gaussian']['model']);a=c['single_gaussian']['geometry'];K=np.linalg.inv(m.covariances[0])
        np.testing.assert_allclose(a['gradient'],K@m.means[0],atol=1e-14);np.testing.assert_allclose(a['hessian'],-K,atol=1e-14);np.testing.assert_array_equal(a['responsibility_jacobian'],np.zeros((1,2)))
        d=c['duplicate']['geometry'];np.testing.assert_allclose(d['responsibility_jacobian'],0,atol=1e-14);np.testing.assert_allclose(d['hessian'],[[-1/1.2]],atol=1e-14)

    def test_bounded_trace_monotonicity_and_every_armijo_step(self):
        report=saved();raw=load()[0];x=(raw-np.array(report['mean']))/np.array(report['scale'])
        for config in report['configurations']:
            m=Model.from_dict(next(a for a in report['candidates'] if a['id']==config['id'])['model']);a=trace(x,m,max_iter=40)
            self.assertLessEqual(a['updates'],40)
            for before,after in zip(a['history'],a['history'][1:]):
                self.assertGreater(after['actual_gain'],0);self.assertGreaterEqual(after['actual_gain'],after['required_gain']);self.assertAlmostEqual(after['log_density']-before['log_density'],after['actual_gain'],places=12)
                self.assertAlmostEqual(math.dist(before['point'],after['point']),after['step'],places=11)
                self.assertLessEqual(after['halvings'],32)
                self.assertAlmostEqual(after['required_gain'],1e-4*after['step']*before['gradient_norm'],places=12)
            if a['status']=='gradient_norm_criterion':self.assertLessEqual(a['final']['gradient_norm'],1e-5)
            else:self.assertIn(a['status'],('budget_exhausted','line_search_exhausted'))

    def test_budget_zero_and_invalid_inputs(self):
        c=controls()['single_gaussian'];m=Model.from_dict(c['model']);a=trace([0,0],m,0);self.assertEqual(a['updates'],0);self.assertEqual(a['status'],'budget_exhausted')
        for kwargs in ({'max_iter':-1},{'max_iter':4097},{'max_iter':True},{'tolerance':0},{'tolerance':float('nan')}):
            with self.assertRaises(ValueError):trace([0,0],m,**kwargs)
        for value in ([0],[0,float('nan')],[0,float('inf')]):
            with self.assertRaises(ValueError):geometry(value,m)
        with self.assertRaises(ValueError):geometry_raw([0,0],m,[0,0],[0,1])

if __name__=='__main__':unittest.main()

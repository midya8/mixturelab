import copy
import math
import unittest
import numpy as np
from mixturelab.core import Model, expectation, component_logs, constrained_covariance, maximization, em, initialize, score
from mixturelab.controls import controls


class GaussianTests(unittest.TestCase):
    def test_independent_two_dimensional_density(self):
        rng=np.random.default_rng(8)
        for _ in range(30):
            a=rng.normal(size=(2,2));c=a@a.T+.1*np.eye(2)
            mu=rng.normal(size=2);x=rng.normal(size=(7,2))
            m=Model(np.array([1.]),mu[None],c[None],'full',.05)
            det=c[0,0]*c[1,1]-c[0,1]**2
            inverse=np.array([[c[1,1],-c[0,1]],[-c[0,1],c[0,0]]])/det
            expected=[-.5*(2*math.log(2*math.pi)+math.log(det)+float((v-mu)@inverse@(v-mu))) for v in x]
            ll,r=expectation(x,m)
            np.testing.assert_allclose(ll,expected,atol=2e-12,rtol=1e-12)
            np.testing.assert_array_equal(r,np.ones((7,1)))

    def test_independent_one_dimensional_mixture(self):
        m=Model(np.array([.2,.8]),np.array([[-2.],[3.]]),np.array([[[.3]],[[2.]]]),'full',.05)
        for value in (-20.,-1.,0.,5.,30.):
            terms=[w*math.exp(-.5*(value-mu)**2/var)/math.sqrt(2*math.pi*var) for w,mu,var in zip(m.weights,m.means[:,0],m.covariances[:,0,0])]
            log,r=expectation([[value]],m)
            self.assertAlmostEqual(log[0],math.log(sum(terms)),places=12)
            np.testing.assert_allclose(r[0],np.array(terms)/sum(terms),atol=1e-13)

    def test_stable_far_tail_and_underflow(self):
        m=Model(np.array([.1,.9]),np.array([[0.],[1000.]]),np.array([[[1.]],[[1.]]]),'full',.05)
        ll,r=expectation([[0.],[1000.],[-10000.]],m)
        self.assertTrue(np.isfinite(ll).all());np.testing.assert_allclose(r.sum(axis=1),1)
        self.assertEqual(r[0,1],0)

    def test_permutation_invariance(self):
        x=np.array([[-2.,1.],[1.,3.],[4.,0.]])
        m=initialize(x,3,'full',.05,11)
        p=[2,0,1];q=Model(m.weights[p],m.means[p],m.covariances[p],'full',.05)
        a,r=expectation(x,m);b,s=expectation(x,q)
        np.testing.assert_allclose(a,b,atol=1e-12);np.testing.assert_allclose(r[:,p],s,atol=1e-12)

    def test_duplicate_components_do_not_identify(self):
        c=controls()['duplicate']
        np.testing.assert_allclose(c['single_log_density'],c['duplicate_log_density'],atol=1e-14)
        np.testing.assert_allclose(c['responsibilities'],np.tile([.2,.8],(3,1)),atol=1e-14)

    def test_collapse_path_and_finite_floor_bound(self):
        c=controls()['collapse'];a=[v['mean_log_likelihood'] for v in c['path']]
        self.assertGreater(a[-1],a[0]+4)
        m=Model(np.array([.4,.6]),np.array([[0.],[0.]]),np.array([[[.05]],[[9.]]]),'full',.05)
        ll,_=expectation(np.linspace(-10,10,1001)[:,None],m)
        self.assertLessEqual(ll.max(),c['constrained_log_density_upper_bound']+1e-12)

    def test_reject_invalid_models_and_samples(self):
        m=Model(np.array([1.]),np.array([[0.]]),np.array([[[1.]]]),'full',.05)
        for bad in ([],[[math.nan]],[[math.inf]],[[1.,2.]]):
            with self.assertRaises(ValueError):expectation(bad,m)
        for weights,cov,family in [([0.],[[[1.]]],'full'),([.5],[[[1.]]],'full'),([1.],[[[.001]]],'full'),([1.],[[[1.]]],'bad')]:
            with self.assertRaises(ValueError):Model(np.array(weights),np.array([[0.]]),np.array(cov),family,.05).validate()
        with self.assertRaises(ValueError):Model(np.array([1.]),np.zeros((1,2)),np.array([[[1.,.1],[.1,1.]]]),'diagonal',.05).validate()


class MStepTests(unittest.TestCase):
    def test_weighted_scalar_oracle(self):
        x=np.array([[-2.],[1.],[4.]])
        r=np.array([[.1,.9],[.4,.6],[.8,.2]])
        m,_=maximization(x,r,'diagonal',.05)
        for j in range(2):
            n=sum(float(row[j]) for row in r)
            mu=sum(float(row[j])*float(v[0]) for row,v in zip(r,x))/n
            variance=sum(float(row[j])*(float(v[0])-mu)**2 for row,v in zip(r,x))/n
            self.assertAlmostEqual(m.weights[j],n/3,places=14)
            self.assertAlmostEqual(m.means[j,0],mu,places=14)
            self.assertAlmostEqual(m.covariances[j,0,0],max(.05,variance),places=14)

    def test_precision_kkt_global_conditional_optimum(self):
        # Independent convex precision-space certificate: C-S=M >=0,
        # K=C^-1 <= I/floor, M*(I/floor-K)=0. Concavity of logdet(K)
        # makes these sufficient for the constrained covariance subproblem.
        rng=np.random.default_rng(42)
        for d in range(1,6):
            for _ in range(20):
                a=rng.normal(size=(d,d));s=a@a.T
                s*=10**rng.uniform(-3,1)
                floor=10**rng.uniform(-2,1)
                c,_=constrained_covariance(s,'full',floor)
                k=np.linalg.inv(c);dual=c-s;slack=np.eye(d)/floor-k
                self.assertGreaterEqual(np.linalg.eigvalsh(dual).min(),-1e-9)
                self.assertGreaterEqual(np.linalg.eigvalsh(slack).min(),-1e-8)
                np.testing.assert_allclose(dual@slack,np.zeros((d,d)),atol=1e-8,rtol=0)
                # Compare random feasible matrices too; no optimizer under test.
                def objective(v):return np.linalg.slogdet(v)[1]+np.trace(np.linalg.solve(v,s))
                for _ in range(4):
                    b=rng.normal(size=(d,d));candidate=b@b.T+floor*np.eye(d)
                    self.assertLessEqual(objective(c),objective(candidate)+1e-8)

    def test_closed_form_rotated_two_dimensional_clip(self):
        # Analytic 2x2 eigenvalues and spectral projector, not eigh().
        rng=np.random.default_rng(9)
        for _ in range(40):
            a=rng.normal(size=(2,2));s=.01*a@a.T
            lo=(np.trace(s)-math.sqrt((s[0,0]-s[1,1])**2+4*s[0,1]**2))/2
            hi=np.trace(s)-lo;f=.03
            if hi-lo<1e-12:continue
            p=(s-lo*np.eye(2))/(hi-lo)
            expected=max(lo,f)*(np.eye(2)-p)+max(hi,f)*p
            c,_=constrained_covariance(s,'full',f)
            np.testing.assert_allclose(c,expected,atol=1e-14)

    def test_diagonal_floor_and_additive_difference(self):
        s=np.array([[.01,.02],[.02,2.]])
        c,n=constrained_covariance(s,'diagonal',.05)
        np.testing.assert_array_equal(c,np.diag([.05,2.]));self.assertEqual(n,1)
        control=controls()['covariance_update']
        self.assertLess(control['clipped_logdet_plus_trace'],control['additive_logdet_plus_trace'])

    def test_all_coincident_rows_and_dead_component(self):
        x=np.ones((4,3))
        m,_=maximization(x,np.ones((4,2))*.5,'full',.05)
        np.testing.assert_allclose(m.covariances,np.tile(.05*np.eye(3),(2,1,1)),atol=1e-14)
        with self.assertRaises(ValueError):maximization(x,np.tile([1.,0.],(4,1)),'full',.05)
        for r in (np.ones((4,2)),np.array([[.5,-.5]]*4),np.ones((3,1))):
            with self.assertRaises(ValueError):maximization(x,r,'full',.05)
        with self.assertRaises(ValueError):constrained_covariance(np.diag([-1.,1.]),'full',.05)


class EMTests(unittest.TestCase):
    def test_full_decomposition_and_final_transition(self):
        rng=np.random.default_rng(5);x=np.r_[rng.normal(-2,.4,(15,2)),rng.normal(2,.8,(15,2))]
        for family in ('full','diagonal'):
            r=em(x,2,family,.05,11,max_iter=60)
            self.assertEqual(r['status'],'likelihood_increment_tolerance')
            for i,h in enumerate(r['history'][1:],1):
                self.assertGreaterEqual(h['gain'],-1e-10);self.assertGreaterEqual(h['q_gain'],-1e-10);self.assertGreaterEqual(h['posterior_kl'],-1e-10)
                self.assertAlmostEqual(h['gain'],h['q_gain']+h['posterior_kl'],places=11)
            w=r['last_transition'];a=Model.from_dict(w['before']);b=Model.from_dict(w['after'])
            ll,q=expectation(x,a);newll,newq=expectation(x,b)
            # Positive probabilities here permit a direct posterior KL oracle.
            kl=sum(float(u)*math.log(float(u/v)) for row,s in zip(q,newq) for u,v in zip(row,s))/len(x)
            self.assertAlmostEqual(kl,r['history'][-1]['posterior_kl'],places=12)
            self.assertAlmostEqual(-score(x,b),r['history'][-1]['mean_log_likelihood'],places=12)

    def test_zero_budget_and_single_update(self):
        x=np.array([[-2.],[0.],[3.]])
        a=em(x,2,max_iter=0);self.assertEqual(a['iterations'],0);self.assertEqual(a['status'],'budget_exhausted');self.assertIsNone(a['last_transition'])
        b=em(x,2,max_iter=1,tolerance=0);self.assertEqual(b['iterations'],1);self.assertEqual(b['status'],'budget_exhausted')
        for key in ('means','weights','covariances'):np.testing.assert_array_equal(a['model'][key],a['initial'][key])

    def test_one_component_has_exact_mle(self):
        x=np.array([[-2.,1.],[0.,-1.],[4.,2.],[1.,-3.]])
        r=em(x,1,'full',.05,23)
        m=Model.from_dict(r['model']);delta=x-x.mean(axis=0)
        np.testing.assert_allclose(m.means[0],x.mean(axis=0),atol=1e-14)
        np.testing.assert_allclose(m.covariances[0],delta.T@delta/len(x),atol=1e-14)

    def test_reproducible_and_invalid_controls(self):
        x=np.array([[-2.],[0.],[3.]])
        self.assertEqual(em(x,2,max_iter=2),em(x,2,max_iter=2))
        for kwargs in ({'k':0},{'k':4},{'max_iter':-1},{'tolerance':-1},{'floor':0}):
            with self.assertRaises(ValueError):em(x,**kwargs)

if __name__=='__main__':unittest.main()

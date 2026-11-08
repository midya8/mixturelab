"""Small, inspectable counterexamples. Not empirical claims about all mixtures."""
import math
import numpy as np
from .core import Model, expectation, constrained_covariance


def controls():
    x = np.array([[-2.], [0.], [3.]])
    collapse = []
    for sigma in (1., .1, .01, .001, 1e-5, 1e-8):
        # This is a supplied parameter path, not an EM fit. Its second component
        # stays broad while the first collapses exactly onto the middle sample.
        terms = []
        for v in x[:, 0]:
            a = math.log(.4) - math.log(sigma) - .5*math.log(2*math.pi) - .5*(v/sigma)**2
            b = math.log(.6) - math.log(3.) - .5*math.log(2*math.pi) - .5*(v/3.)**2
            top = max(a,b)
            terms.append(top+math.log(math.exp(a-top)+math.exp(b-top)))
        collapse.append(dict(sigma=sigma, mean_log_likelihood=sum(terms)/len(terms)))
    one = Model(np.array([1.]), np.array([[.5]]), np.array([[[1.2]]]), 'full', .05)
    duplicate = Model(np.array([.2,.8]), np.array([[.5],[.5]]), np.array([[[1.2]],[[1.2]]]), 'full', .05)
    a, ra = expectation(x, one)
    b, rb = expectation(x, duplicate)
    rotation = np.array([[math.cos(.7), -math.sin(.7)], [math.sin(.7), math.cos(.7)]])
    scatter = rotation @ np.diag([.01,2.]) @ rotation.T
    clipped, _ = constrained_covariance(scatter, 'full', .05)
    additive = scatter + .05*np.eye(2)
    def objective(c):
        return float(np.linalg.slogdet(c)[1] + np.trace(np.linalg.solve(c,scatter)))
    return dict(collapse=dict(points=x[:,0].tolist(), path=collapse, limit='unbounded above as sigma tends to zero; no finite unconstrained global maximum on this path', constrained_log_density_upper_bound=-.5*math.log(2*math.pi*.05)), duplicate=dict(points=x[:,0].tolist(), one=one.to_dict(), split=duplicate.to_dict(), single_log_density=a.tolist(), duplicate_log_density=b.tolist(), responsibilities=rb.tolist(), maximum_density_difference=float(abs(a-b).max())), covariance_update=dict(scatter=scatter.tolist(), floor=.05, clipped=clipped.tolist(), additive=additive.tolist(), clipped_logdet_plus_trace=objective(clipped), additive_logdet_plus_trace=objective(additive), meaning='Smaller is the exact fixed-responsibility covariance objective. Eigenvalue clipping solves this constrained subproblem; adding floor*I generally does not.'))

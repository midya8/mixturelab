"""Finite Gaussian mixtures with a fixed, explicit covariance constraint.
No sklearn, scipy, automatic differentiation, or hidden reseeding.
"""
from dataclasses import dataclass
import math
import numpy as np


def matrix(x):
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or not a.shape[0] or not a.shape[1] or not np.isfinite(a).all():
        raise ValueError('samples must be a nonempty finite matrix')
    return a


@dataclass
class Model:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    family: str
    floor: float

    def validate(self):
        self.weights = np.asarray(self.weights, dtype=float)
        self.means = matrix(self.means)
        self.covariances = np.asarray(self.covariances, dtype=float)
        k, d = self.means.shape
        if self.family not in ('full', 'diagonal') or not math.isfinite(self.floor) or self.floor <= 0:
            raise ValueError('invalid covariance family or floor')
        if self.weights.shape != (k,) or not np.isfinite(self.weights).all() or np.any(self.weights <= 0) or abs(self.weights.sum() - 1) > 1e-10:
            raise ValueError('weights must be positive and sum to one')
        c = self.covariances
        if c.shape != (k, d, d) or not np.isfinite(c).all() or not np.allclose(c, c.swapaxes(1, 2), atol=1e-12, rtol=0):
            raise ValueError('covariances must be finite symmetric matrices')
        if np.linalg.eigvalsh(c).min() < self.floor - 1e-10 * max(1., self.floor):
            raise ValueError('covariance violates floor')
        if self.family == 'diagonal' and not np.array_equal(c, np.array([np.diag(np.diag(v)) for v in c])):
            raise ValueError('diagonal family has off-diagonal entries')
        return self

    def to_dict(self):
        self.validate()
        return dict(weights=self.weights.tolist(), means=self.means.tolist(), covariances=self.covariances.tolist(), family=self.family, floor=self.floor)

    @classmethod
    def from_dict(cls, value):
        return cls(**{k: value[k] for k in ('weights', 'means', 'covariances', 'family', 'floor')}).validate()


def component_logs(x, model):
    x = matrix(x)
    model.validate()
    if x.shape[1] != model.means.shape[1]:
        raise ValueError('sample dimension differs from model')
    columns = []
    for w, mu, cov in zip(model.weights, model.means, model.covariances):
        chol = np.linalg.cholesky(cov)
        z = np.linalg.solve(chol, (x - mu).T)
        columns.append(math.log(w) - .5 * (x.shape[1] * math.log(2 * math.pi) + 2 * np.log(np.diag(chol)).sum() + (z * z).sum(axis=0)))
    a = np.array(columns).T
    if not np.isfinite(a).all():
        raise ValueError('input exceeds finite numerical range')
    return a


def expectation(x, model):
    logs = component_logs(x, model)
    top = logs.max(axis=1)
    shifted = np.exp(logs - top[:, None])
    den = shifted.sum(axis=1)
    return top + np.log(den), shifted / den[:, None]


def constrained_covariance(scatter, family, floor):
    s = np.asarray(scatter, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or not np.isfinite(s).all() or not np.allclose(s, s.T, atol=1e-12):
        raise ValueError('scatter must be a finite symmetric matrix')
    if family not in ('full', 'diagonal') or not math.isfinite(floor) or floor <= 0:
        raise ValueError('invalid covariance constraint')
    s = .5 * (s + s.T)
    values, vectors = np.linalg.eigh(s)
    if values.min() < -1e-10 * max(1., abs(values).max()):
        raise ValueError('scatter must be positive semidefinite')
    if family == 'diagonal':
        diagonal = np.diag(s)
        return np.diag(np.maximum(diagonal, floor)), int(np.count_nonzero(diagonal < floor))
    return (vectors * np.maximum(values, floor)) @ vectors.T, int(np.count_nonzero(values < floor))


def maximization(x, responsibilities, family, floor):
    x = matrix(x)
    r = np.asarray(responsibilities, dtype=float)
    if r.ndim != 2 or r.shape[0] != len(x) or not r.shape[1] or not np.isfinite(r).all() or np.any(r < 0) or not np.allclose(r.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError('responsibilities must be finite nonnegative normalized rows')
    mass = r.sum(axis=0)
    if np.any(mass <= np.finfo(float).tiny):
        raise ValueError('component has zero numerical mass; no hidden reseeding')
    means = r.T @ x / mass[:, None]
    covs, clipped = [], []
    for j in range(len(mass)):
        delta = x - means[j]
        scatter = (delta.T * r[:, j]) @ delta / mass[j]
        cov, count = constrained_covariance(scatter, family, floor)
        covs.append(cov)
        clipped.append(count)
    return Model(mass / len(x), means, np.array(covs), family, floor).validate(), clipped


def initialize(x, k, family, floor, seed):
    x = matrix(x)
    if not isinstance(k, int) or not 1 <= k <= len(x):
        raise ValueError('invalid component count')
    # Random distinct row centers, same pooled covariance. No label information.
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(x), k, replace=False)
    delta = x - x.mean(axis=0)
    cov, _ = constrained_covariance(delta.T @ delta / len(x), family, floor)
    return Model(np.ones(k) / k, x[ids].copy(), np.array([cov.copy() for _ in range(k)]), family, floor).validate()


def em(x, k=3, family='full', floor=.05, seed=11, max_iter=500, tolerance=1e-7):
    x = matrix(x)
    if not isinstance(max_iter, int) or max_iter < 0 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('invalid stopping controls')
    model = initialize(x, k, family, floor, seed)
    initial = model.to_dict()
    ll, r = expectation(x, model)
    history = [dict(iteration=0, mean_log_likelihood=float(ll.mean()), min_mass=float(r.sum(axis=0).min()), min_eigenvalue=float(np.linalg.eigvalsh(model.covariances).min()))]
    stopped = 'budget_exhausted'
    witness = None
    for iteration in range(1, max_iter + 1):
        new_model, clipped = maximization(x, r, family, floor)
        new_ll, new_r = expectation(x, new_model)
        old_logs = component_logs(x, model)
        new_logs = component_logs(x, new_model)
        q_gain = float(np.sum(r * (new_logs - old_logs)) / len(x))
        # Use log responsibilities from logs, not log(underflowed zero).
        kl = float(np.sum(r * ((old_logs - ll[:, None]) - (new_logs - new_ll[:, None]))) / len(x))
        gain = float(np.mean(new_ll - ll))
        if gain < -1e-9 or q_gain < -1e-9 or kl < -1e-9 or abs(gain - q_gain - kl) > 1e-8:
            raise ArithmeticError('EM monotonicity/decomposition numerical audit failed')
        history.append(dict(iteration=iteration, mean_log_likelihood=float(new_ll.mean()), gain=gain, q_gain=q_gain, posterior_kl=kl, identity_error=abs(gain-q_gain-kl), clipped_eigenvalues=clipped, min_mass=float(new_r.sum(axis=0).min()), min_eigenvalue=float(np.linalg.eigvalsh(new_model.covariances).min())))
        witness = dict(before=model.to_dict(), after=new_model.to_dict())
        model, ll, r = new_model, new_ll, new_r
        if gain <= tolerance:
            stopped = 'likelihood_increment_tolerance'
            break
    return dict(model=model.to_dict(), initial=initial, history=history, status=stopped, iterations=len(history)-1, last_transition=witness)


def score(x, model):
    return float(-expectation(x, model)[0].mean())

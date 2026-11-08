"""Frozen mixture conditioning, not missing-data fitting or causal inference."""
import copy
import math
import numpy as np


def _indices(observed, d):
    if not isinstance(observed, (list, tuple)) or any(type(i) is not int or not 0 <= i < d for i in observed) or len(set(observed)) != len(observed):
        raise ValueError('observed must be distinct integer coordinate indices')
    return list(observed), [i for i in range(d) if i not in observed]


def _values(values, n):
    if isinstance(values, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in values):
        raise ValueError('values must be numbers, not booleans')
    raw = np.asarray(values)
    if raw.dtype.kind not in 'iuf':
        raise ValueError('values must be numbers, not booleans or strings')
    a = raw.astype(float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError('observed values must be a matching finite vector')
    return a


def _normal_logs(samples, mean, covariance):
    n, d = samples.shape
    if d == 0:
        return np.zeros(n)
    chol = np.linalg.cholesky(covariance)
    z = np.linalg.solve(chol, (samples - mean).T)
    with np.errstate(over='ignore', invalid='ignore'):
        out = -.5 * (d * math.log(2 * math.pi) + 2 * np.log(np.diag(chol)).sum() + (z * z).sum(axis=0))
    if not np.isfinite(out).all():
        raise ValueError('input exceeds finite numerical range')
    return out


def _normalize(logs):
    top = logs.max()
    evidence = top + math.log(float(np.exp(logs - top).sum()))
    if not math.isfinite(evidence):
        raise ValueError('conditioning exceeds finite numerical range')
    return evidence, logs - evidence


def condition(model, observed, values):
    """Condition on only the supplied coordinates, preserving their order.

    Free coordinates use sorted complement order. No hidden values, labels,
    random draws, fitting, new ridge or covariance clipping enter this result.
    Empty observed/free sets have zero-dimensional unit density by convention.
    """
    model = copy.deepcopy(model).validate()
    k, d = model.means.shape
    obs, free = _indices(observed, d)
    values = _values(values, len(obs))
    means, covs, logs = [], [], []
    for weight, mu, covariance in zip(model.weights, model.means, model.covariances):
        coo = covariance[np.ix_(obs, obs)]
        cmo = covariance[np.ix_(free, obs)]
        cmm = covariance[np.ix_(free, free)]
        marginal = float(_normal_logs(values[None, :], mu[obs], coo)[0])
        if obs:
            chol = np.linalg.cholesky(coo)
            delta = np.linalg.solve(chol.T, np.linalg.solve(chol, values - mu[obs]))
            gain_transpose = np.linalg.solve(chol.T, np.linalg.solve(chol, cmo.T))
            cm = mu[free] + cmo @ delta
            cc = cmm - cmo @ gain_transpose
        else:
            cm, cc = mu[free].copy(), cmm.copy()
        cc = .5 * (cc + cc.T)
        if free:
            np.linalg.cholesky(cc)  # Refuse breakdown, never repair a changed density.
        means.append(cm)
        covs.append(cc)
        logs.append(math.log(float(weight)) + marginal)
    means, covs, logs = np.array(means), np.array(covs), np.array(logs)
    evidence, log_weights = _normalize(logs)
    if not obs:
        evidence = 0.0  # Unit mass for the zero-dimensional observed margin.
    weights = np.exp(log_weights)
    mean = weights @ means
    centered = means - mean
    within = np.einsum('k,kij->ij', weights, covs)
    between = (centered.T * weights) @ centered
    covariance = within + between
    for a in (means, covs, weights, log_weights, mean, within, between, covariance):
        if not np.isfinite(a).all():
            raise ValueError('conditional moments exceed finite numerical range')
    return dict(dimension=d, observed=obs, free=free, values=values.tolist(),
                log_evidence=float(evidence), log_weights=log_weights.tolist(),
                weights=weights.tolist(), means=means.tolist(), covariances=covs.tolist(),
                mean=mean.tolist(), within_covariance=within.tolist(),
                between_covariance=between.tolist(), covariance=covariance.tolist(),
                coordinate_space='standardized')


def condition_raw(model, observed, values, center, scale):
    """Original-unit conditionals using fixed training-only affine scaling."""
    model = copy.deepcopy(model).validate()
    d = model.means.shape[1]
    obs, free = _indices(observed, d)
    center, scale = _values(center, d), _values(scale, d)
    if np.any(scale <= 0):
        raise ValueError('scale must be positive')
    values = _values(values, len(obs))
    r = condition(model, obs, ((values - center[obs]) / scale[obs]).tolist())
    sf = scale[free]
    r['means'] = (np.array(r['means']) * sf + center[free]).tolist()
    r['mean'] = (np.array(r['mean']) * sf + center[free]).tolist()
    for key in ('covariances', 'within_covariance', 'between_covariance', 'covariance'):
        a = np.array(r[key]).reshape((-1, len(free), len(free))) if key == 'covariances' and free else np.array(r[key])
        if free:
            a = a * sf[:, None] * sf[None, :]
        r[key] = a.tolist()
    r['values'] = values.tolist()
    r['log_evidence'] -= float(np.log(scale[obs]).sum())
    r['coordinate_space'] = 'original'
    for key in ('means', 'mean', 'covariances', 'within_covariance', 'between_covariance', 'covariance'):
        if not np.isfinite(np.array(r[key])).all():
            raise ValueError('original-unit moments exceed finite numerical range')
    if not math.isfinite(r['log_evidence']):
        raise ValueError('original-unit evidence exceeds finite numerical range')
    return r


def log_density(result, samples):
    """Evaluate a result made by condition/condition_raw, not a source verifier.

    Uses normalized log weights even if displayed floating weights underflow.
    """
    free = result['free']
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or not len(x) or x.shape[1] != len(free) or not np.isfinite(x).all():
        raise ValueError('targets must be a nonempty finite free-coordinate matrix')
    if not free:
        return np.zeros(len(x))
    means, covs, logs = np.array(result['means']), np.array(result['covariances']), np.array(result['log_weights'])
    columns = np.array([w + _normal_logs(x, m, c) for w, m, c in zip(logs, means, covs)]).T
    top = columns.max(axis=1)
    out = top + np.log(np.exp(columns - top[:, None]).sum(axis=1))
    if not np.isfinite(out).all():
        raise ValueError('conditional density exceeds finite numerical range')
    return out


def controls():
    from .core import Model
    def model(weights, means, variances):
        return Model(np.array(weights), np.array(means), np.array([np.diag(v) for v in variances]), 'diagonal', .01).validate()
    gap = model([.5,.5], [[0.,-4.],[0.,4.]], [[1.,.25],[1.,.25]])
    increase = model([.99,.01], [[0.,0.],[4.,10.]], [[1.,.1],[1.,.1]])
    before = condition(increase, [], [])
    after = condition(increase, [0], [(8+math.log(99))/4])
    g = condition(gap, [0], [0.])
    return dict(mean_in_gap=dict(model=gap.to_dict(), conditional=g,
                log_density_at_mean=float(log_density(g, [[0.]])[0]),
                log_density_at_component_mean=float(log_density(g, [[4.]])[0])),
                increasing_variance=dict(model=increase.to_dict(), before=before, after=after,
                unconditional_target_variance=before['covariance'][1][1],
                conditional_target_variance=after['covariance'][0][0]))

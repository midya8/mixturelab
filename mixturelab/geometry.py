"""Analytic INPUT derivatives and bounded log-density ascent, fixed model only."""
import math
import numpy as np
from .core import Model, expectation


def point(x,d):
    a=np.asarray(x,dtype=float)
    if a.shape!=(d,) or not np.isfinite(a).all():raise ValueError('point must have the model dimension and finite values')
    return a


def geometry(x,model):
    model.validate();d=model.means.shape[1];x=point(x,d)
    ll,r=expectation(x[None,:],model);r=r[0]
    precision=np.array([np.linalg.solve(c,np.eye(d)) for c in model.covariances])
    precision=.5*(precision+precision.swapaxes(1,2))
    component_gradients=-np.einsum('kij,kj->ki',precision,x-model.means)
    gradient=r@component_gradients
    centered=component_gradients-gradient
    # Centered covariance avoids subtracting two large second moments.
    precision_term=-np.einsum('k,kij->ij',r,precision)
    posterior_covariance=(centered.T*r)@centered
    hessian=precision_term+posterior_covariance
    jacobian=r[:,None]*centered
    for a in (gradient,hessian,jacobian,precision_term,posterior_covariance):
        if not np.isfinite(a).all():raise ValueError('input derivatives exceed finite range')
    return dict(point=x.tolist(),log_density=float(ll[0]),responsibilities=r.tolist(),gradient=gradient.tolist(),hessian=hessian.tolist(),responsibility_jacobian=jacobian.tolist(),precision_term=precision_term.tolist(),posterior_gradient_covariance=posterior_covariance.tolist(),gradient_norm=float(math.hypot(*gradient)))


def geometry_raw(raw,model,mean,scale):
    d=model.means.shape[1];raw=point(raw,d);mean=point(mean,d);scale=point(scale,d)
    if np.any(scale<=0):raise ValueError('scale must be positive')
    result=geometry((raw-mean)/scale,model)
    result.update(raw_point=raw.tolist(),gradient_original=(np.array(result['gradient'])/scale).tolist(),hessian_original=(np.array(result['hessian'])/scale[:,None]/scale[None,:]).tolist(),responsibility_jacobian_original=(np.array(result['responsibility_jacobian'])/scale).tolist())
    return result


def trace(x,model,max_iter=128,tolerance=1e-5):
    if not isinstance(max_iter,int) or isinstance(max_iter,bool) or not 0<=max_iter<=4096 or not math.isfinite(tolerance) or tolerance<=0:
        raise ValueError('invalid trace budget or tolerance')
    model.validate();x=point(x,model.means.shape[1]);history=[]
    current=geometry(x,model)
    history.append(dict(point=x.tolist(),log_density=current['log_density'],gradient_norm=current['gradient_norm']))
    status='budget_exhausted'
    for _ in range(max_iter):
        norm=current['gradient_norm']
        if norm<=tolerance:
            status='gradient_norm_criterion';break
        if not math.isfinite(norm) or norm==0:raise ValueError('invalid gradient norm')
        # Cap the starting physical step at one standardized Euclidean unit.
        direction=np.array(current['gradient'])/norm
        accepted=False
        for halves in range(33):
            step=2.**(-halves)
            proposed=x+step*direction
            ll=float(expectation(proposed[None,:],model)[0][0])
            required=1e-4*step*norm
            gain=ll-current['log_density']
            if gain>=required:
                x=proposed
                current=geometry(x,model)
                history.append(dict(point=x.tolist(),log_density=ll,gradient_norm=current['gradient_norm'],step=step,halvings=halves,required_gain=required,actual_gain=gain))
                accepted=True;break
        if not accepted:
            status='line_search_exhausted';break
    if current['gradient_norm']<=tolerance:status='gradient_norm_criterion'
    eigenvalues=np.linalg.eigvalsh(np.array(current['hessian']))
    return dict(history=history,status=status,updates=len(history)-1,budget=max_iter,tolerance=tolerance,armijo=1e-4,initial_step=1.,max_halvings=32,final=current,final_hessian_eigenvalues=eigenvalues.tolist(),scope='bounded fixed-density log-likelihood ascent in standardized coordinates; no parameter training, exhaustive mode search, global optimum or exact stationary/mode certificate')


def controls():
    # Equal unit-variance Gaussians at +/-2: zero score at x=0 but H=3.
    minimum=Model(np.array([.5,.5]),np.array([[-2.],[2.]]),np.array([[[1.]],[[1.]]]),'full',.05)
    single=Model(np.array([1.]),np.array([[2.,-1.]]),np.array([[[1.,.2],[.2,2.]]]),'full',.05)
    duplicate=Model(np.array([.2,.8]),np.array([[.5],[.5]]),np.array([[[1.2]],[[1.2]]]),'full',.05)
    return dict(stationary_minimum=dict(model=minimum.to_dict(),geometry=geometry([0.],minimum),trace=trace([0.],minimum),escaping_start=trace([.01],minimum)),single_gaussian=dict(model=single.to_dict(),geometry=geometry([0.,0.],single),trace=trace([0.,0.],single)),duplicate=dict(model=duplicate.to_dict(),geometry=geometry([4.],duplicate)))

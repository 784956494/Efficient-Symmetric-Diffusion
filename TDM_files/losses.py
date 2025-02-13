import numpy as np
import torch
import util
from sde import vectorlize_g, matrixlize_xi
from tqdm import tqdm
from ipdb import set_trace as debug
from torch.func import vjp, jvp, vmap, jacrev

def compute_SSM_train(opt, label,dyn, ts, gs,xis, net, return_z=False):
    dt      = dyn.dt
    label = label.reshape(label.shape[0],-1)
    g_ts    = dyn.sigma(ts)
    g_ts    = g_ts[:,None,None,None] if util.is_image_dataset(opt) else g_ts[:,None]
    _ts     = ts.reshape(label.shape[0], *([1,]*(label.dim()-1)))
    reweight=1/(dyn.sigma(_ts)*np.sqrt(dt)/np.sqrt(2))
    func = lambda x: reweight*g_ts*dt * net(gs, x, ts)
    # zs      = net(gs,xis,ts)
    zs, div_val      = output_and_div(func, xis)
    loss    = torch.nn.functional.mse_loss(reweight*g_ts*dt*zs,reweight*label) + 2 * div_val
    return loss, zs if return_z else loss

def compute_DSM_train(reweight,pred,label):
    return torch.nn.functional.mse_loss(reweight*pred,reweight*label)


def div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u)
    return lambda x: torch.trace(J(x).squeeze(-2).squeeze(-2).squeeze(0)).unsqueeze(0)

def get_exact_div_fn(fn):
    """Flatten all but the last axis and compute the true divergence."""

    def div_fun(y, gs, ts):
        y_shape = y.shape
        dim = np.prod(y_shape[1:])
        y = y.unsqueeze(1)
        jac = torch.vmap(torch.func.jacrev(fn, argnums=0))(gs, y, ts)
        # jac = torch.autograd.functional.jacobian(fn, (y, t), create_graph=True)[0]
        jac = jac.view(y_shape[0], dim, dim)
        return jac.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

    return div_fun

def get_div_fn(drift_fn):
    """Euclidean divergence of the drift function."""
    f = get_exact_div_fn(drift_fn)
    return lambda y, g, t: f(y, g, t)
def output_and_div(vecfield, x, gs, ts,v=None, div_mode="exact"):
    if div_mode == "exact":
        dx = vecfield(x)
        div_func = get_div_fn(vecfield)
        # div = vmap(div_fn(vecfield))(x)
        div = div_func(x, gs, ts)
    else:
        dx, vjpfunc = vjp(vecfield, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    # return dx, -div-x.shape[-1]
    return dx, div

def get_exact_div_fn(fn):
    """Flatten all but the last axis and compute the true divergence."""

    def funcation(y):
        y_shape = y.shape
        print(y)
        dim = int(np.prod(y_shape[1:]))
        # Compute Jacobian for each input
        jac = vmap(jacrev(fn, argnums=0))(y)
        jac = jac.reshape(y_shape[0], dim, dim)
        
        # Calculate divergence (sum of diagonal elements)
        divergence = jac.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        return divergence

    return funcation

def get_div_fn(drift_fn):
    """Euclidean divergence of the drift function."""
    f = get_exact_div_fn(drift_fn)
    return lambda y: f(y)

def output_and_div(vecfield, x, v=None, div_mode="exact"):
    if div_mode == "exact":
        dx = vecfield(x)
        div_func = get_div_fn(vecfield)
        div = div_func(x)
    else:
        dx, vjpfunc = torch.autograd.functional.vjp(vecfield, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, div

def compute_NLL(opt,net,dyn,eval_loader):
    n,dim,_=opt.data_dim
    loglik = 0
    num_data= eval_loader.dataset.shape[0]
    
    if opt.problem_name in ["Protein", "RNA", "Checkerboard", "Pacman", "HighdimTorus"]:
        offset  = n*np.log(2*np.pi)
    elif opt.problem_name in ["GmmAlgebra"]:
        offset = n * np.log(8 * np.pi ** 2)
    else:
        offset = 0
    for idx, g in enumerate(tqdm(eval_loader,desc=util.yellow("Evaluating NLL"))):
        g           = g.to(opt.device)
        bs,n,dim,_  = g.shape
        xi_dim      = dim ** 2  if opt.mode=='u' else int((dim*(dim-1))/2)
        xi          = torch.randn(bs,n,xi_dim)
        logp0xi     = util.log_density_multivariate_normal(xi.reshape(bs,-1))
        _ts         = torch.linspace(opt.t0, opt.T, opt.interval)
        acc_Logprob = 0
        vv          = torch.randn_like(xi.reshape(bs,-1))
        for idx, t in enumerate(_ts):
            _t = t.repeat(bs)[...,None]
            vecfield = lambda z: net(vectorlize_g(g, mode = opt.mode),z, _t)
            score, div = output_and_div(vecfield, xi.reshape(bs,-1),v=vv,div_mode='approx')
            acc_Logprob = acc_Logprob+div*opt.dt
            g, xi       = dyn.propagate(t, g, xi, score, 'forward', dw=0,ode=True)
        logpTg  = 0
        logpTxi = util.log_density_multivariate_normal(xi.reshape(bs,-1))
        logp0g  = acc_Logprob+logpTg+logpTxi-logp0xi
        logp0g  = logp0g-offset # divide over dim
        loglik  = loglik+logp0g.sum() #sum the loglikelihood for all batch
    avg_loglik= loglik/num_data
    return -avg_loglik # average over batch
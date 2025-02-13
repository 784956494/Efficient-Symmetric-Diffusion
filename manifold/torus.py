import torch
import numpy as np
class Torus:
    name = 'Torus'
    @classmethod
    def exp(cls, x, v):
        return (x + v).remainder(2 * np.pi)
    @classmethod
    def log(cls, x, y):
        v = (y - x).remainder(2 * np.pi)
        v[v > np.pi] -= 2 * np.pi
        return v
    @classmethod
    def expt(cls, x, v, t):
        return cls.exp(x, t[:, None] * v), v
    @classmethod
    def projx(cls, x):
        return x.remainder(2 * np.pi)
    @classmethod
    def proju(cls, x, v):
        return v 
    @classmethod
    def phi(cls, x):
        return torch.remainder(x, 2 * torch.pi)
    @classmethod
    def psi(cls, x):
        return x
    @classmethod
    def grad_phi(cls, x):
        out = torch.eye(x.size(-1), device=x.device, dtype=x.dtype)
        out = out.reshape((1, x.size(-1), x.size(-1)))
        return out.repeat(x.size(0), 1, 1)
    @classmethod
    def tr_hess_phi(cls, x):
        return 0
    @classmethod
    def loss(cls, x, model_drift, model_diff=None, beta_scheduler=None, sample_phase=False, x_t_eig=None):
        t = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device).clamp(1e-8, 1 - 1e-8)
        if beta_scheduler:
            int_beta_r = beta_scheduler.integrate(1 - t)
        else:
            int_beta_r = 1 - t
        xe = cls.psi(x)
        mu_t = xe * torch.exp(-0.5 * (int_beta_r))
        var_t = -torch.expm1(-(int_beta_r))
        xe_t = torch.randn_like(xe) * var_t ** 0.5 + mu_t
        x_t = cls.phi(xe_t)
        beta_t = beta_scheduler.step(1 - t)
        drift_target = -(xe_t - mu_t) / var_t.clamp_min(1e-10)
        drift_target = 0.5 * beta_t * xe_t + 2 * beta_t * drift_target
        if sample_phase:
            return drift_target, x_t
        else:
            f_x_t = model_drift(x_t, t)
        drift_loss = torch.mean(var_t * ((drift_target - f_x_t) ** 2))
        return drift_loss, None
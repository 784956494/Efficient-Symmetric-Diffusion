import torch
import numpy as np
from utils.math_utils import Divsin, mul, get_exact_jac_fn
from .torus import Torus
divsin = Divsin.apply

class SOn:
    name = 'Special Orthogonal'
    @classmethod
    def exp(cls, x, v):
        v_base = x.transpose(-1, -2) @ v
        # if x.shape[-1] == 3:
        # # use rodrigues formula, which is numerically stable
        #     theta = v_base[..., [0, 0, 1], [1, 2, 2]].norm(dim=-1).unsqueeze(-1).unsqueeze(-1)
        #     k = v_base / theta
        #     r = torch.matrix_power(k, 0) + theta.sin() * k + (1 - theta.cos()) * torch.matrix_power(k, 2)

        #     return x @ r
        # else:
        r = torch.linalg.matrix_exp(v_base)
        return x @ r
    
    def log(cls, x, y, move_back=True):
        r = x.transpose(-1, -2) @ y
        if x.shape[-1] == 3:
            val = (((r[..., range(3), range(3)]).sum(dim=-1) - 1) / 2).clip(min=-1, max=1)
            theta = val.acos()
            log_val = divsin(theta)[..., None, None] / 2 * (r - r.transpose(-1, -2))

            if move_back:
                return x @ log_val
            else:
                return log_val
        else:
            log_val = matrix_logarithm(r)
            return x @ log_val
    @classmethod  
    def grad_phi(cls, gamma, U, Z):
        # This is 4D tensor transposed
        batch_size, n, n = U.shape
        dV = []
        for i in range(n):
            lambda_i = gamma[:, i].unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
            diag_matrix = lambda_i * torch.eye(Z.shape[1]).to(Z.device)  # (batch_size, n, n)
            diag_matrix = diag_matrix - Z
            diag_matrix_pinv = torch.linalg.pinv(diag_matrix)  # (batch_size, n, n)
            VLV = diag_matrix_pinv
            
            delA = torch.zeros((batch_size, n, n, n, n), dtype=torch.float32)
            
            indices = torch.arange(n)
            
            delA[:, indices[:, None], indices, indices[:, None], indices] = 1
            delA = delA.reshape(batch_size, n * n, n, n)            
            VLV_expanded = VLV.unsqueeze(1).expand(batch_size, n * n, n, n)
            VLV_delA_reshaped = torch.bmm(VLV_expanded.reshape(batch_size * n * n, n, n),
                                delA.reshape(batch_size * n * n, n, n))
            VLV_delA = VLV_delA_reshaped.reshape(batch_size, n, n, n, n)
            v_i = U[:, :, i]
            dvi = torch.einsum('bijkl,bl->bijk', VLV_delA, v_i).transpose(-1, -2)
            dvi = dvi.transpose(1, 2)

            dV.append(dvi)

        grad_f = torch.stack(dV, dim=-3)
        
        return grad_f

    @classmethod
    def proju(cls, x, v):
        return (v - x @ v.transpose(-1, -2) @ x) / 2
    @classmethod
    def projx(cls, x):
            """
            Use svd nearest projection.
            """
            U, _, Vh = torch.linalg.svd(x, full_matrices=False)
            return U @ Vh
    @classmethod
    def psi(cls, U, sign):
        batch_size, n, _ = U.shape
        diag_entries = torch.arange(n, 0, -1, dtype=U.dtype, device=U.device) 
        Lambda = 1 / torch.tensor(n, dtype=U.dtype, device=U.device) * torch.diag(diag_entries)
        
        U_star = U.mT
        result = torch.bmm(U_star, torch.matmul(Lambda, U))  
        
        result_diag = torch.diagonal(result, dim1=-2, dim2=-1)  
        result = result - torch.diag_embed(result_diag) + torch.diag_embed(result_diag / 2)
        upper_tri = torch.triu(result)
        return upper_tri

    @classmethod
    def phi(cls, X):
        A = X + X.mT   
        # A = X          
        eigvals, eigvecs = torch.linalg.eigh(A) 
        #ensure ordering of eigvenctors are correct
        sorted_indices = torch.argsort(eigvals, dim=-1, descending=True)
        eigvals = torch.gather(eigvals, -1, sorted_indices)
        eigvecs = torch.gather(eigvecs, -1, sorted_indices.unsqueeze(-2).expand_as(eigvecs))
        #ensures sign
        signs = torch.sign(eigvecs[..., 0:1, :])
        eigvecs = eigvecs.mT
        # eigvecs = eigvecs * signs.squeeze(1).unsqueeze(-1)
        #ensures rotational
        # eigvecs = eigvecs * eigvecs.det().unsqueeze(-1).unsqueeze(-1)
        return (eigvals, eigvecs, signs)
    
    @classmethod
    def proju(cls, x, v):
        return (v - x @ v.transpose(-1, -2) @ x) / 2
    
    @classmethod
    def sym_noise(cls, batch_size, n, device, dtype):
        xe_t = torch.randn((batch_size, n ** 2), device=device, dtype=dtype).reshape(-1, n, n)
        xe_t_diagonal = torch.randn(batch_size, n).to(device)
        xe_t = torch.triu(xe_t, diagonal=1)
        xe_t = (xe_t + xe_t.mT)
        xe_t = xe_t + torch.diag_embed(xe_t_diagonal)
        return xe_t

    @classmethod
    def to_skew_symm(cls, v, n):
        batch_size = v.shape[0]
        A = torch.zeros((batch_size, n, n), dtype=v.dtype, device=v.device)
        indices = torch.triu_indices(n, n, offset=1)
        A[:, indices[0], indices[1]] = v
        A[:, indices[1], indices[0]] = -v
        return A

    @classmethod
    def tangent_to_son(cls, tangent_vec, n):
        tangent_skew = cls.to_skew_symm(tangent_vec.unsqueeze(0), n)
        return SOn.exp(torch.eye(n).to(tangent_vec.device), tangent_skew.squeeze(0))
    
    @classmethod
    def sample_son(cls, tangent_vec, mean, n):
        mean = mean.broadcast_to([tangent_vec.shape[0], mean.shape[1], mean.shape[1]])
        tangent_skew = cls.to_skew_symm(tangent_vec, n)
        tangent_skew = mean.matmul(tangent_skew)
        return SOn.exp(mean, tangent_skew)
    
    @classmethod
    def loss(cls, x, model_drift, model_diff, beta_scheduler=None):
        batch_size, n, n = x.shape
        t = torch.rand((x.shape[0], 1, 1), dtype=x.dtype, device=x.device).clamp(1e-4, 1 - 1e-4)
        if beta_scheduler:
            int_beta_r = beta_scheduler.integrate(1 - t)
        else:
            int_beta_r = 1 - t
        x_signs = torch.sign(x[..., :, 0:1]).squeeze(1)
        beta_t = beta_scheduler.step(1 - t)
        xe = cls.psi(x, x_signs)
        # xe = xe + xe.mT
        mu_t = (xe) * (torch.exp(-0.5 * (int_beta_r)))
        var_t = -torch.expm1(-(int_beta_r))
        # G = cls.sym_noise(batch_size, n, x.device, x.dtype)
        G = torch.triu(torch.rand_like(xe))
        # G = torch.rand_like(xe)
        xe_t = (G * var_t ** 0.5 + mu_t)
        gamma, x_t, sign = cls.phi(xe_t)

        x_torus = torch.angle(x_signs.squeeze(-1))
        torus_drift, x_torus_t = Torus().loss(x_torus, model_drift, None, beta_scheduler=beta_scheduler, sample_phase=True)

        f_x_t, f_torus_t  = model_drift(x_t, t.squeeze(-1), x_torus_t)
        g_x_t = model_diff(x_t, t.squeeze(-1))
        grad_log_p = -(xe_t - mu_t) / var_t
        drift_target = 0.5 * beta_t * xe_t + 2 * beta_t * grad_log_p
        with torch.enable_grad():
            xe_t.requires_grad_(True)
            projection = lambda X: cls.phi(X)[1]
            jac_fn = get_exact_jac_fn(projection)
            grad = jac_fn(xe_t).squeeze(1)
        grad = grad.detach()
        # grad = self.manifold.grad_phi(gamma, x_t.mT, xe_t)
        # grad = torch.triu(grad)
        # xe_t.requires_grad_(False)
        xe_t = xe_t.detach()
        #compute drift term loss
        drift_target = cls.proju(x_t, mul(drift_target, grad))
        drift_loss = torch.mean(var_t * (((drift_target - f_x_t) ** 2)))
        gamma_i = gamma.unsqueeze(-1) 
        gamma_j = gamma.unsqueeze(-2) 
        diff_target = (gamma_i - gamma_j)
        
        diff_loss = torch.mean(var_t * ((diff_target - g_x_t) ** 2))

        #account for phase
        
        torus_loss = torch.mean(var_t * ((torus_drift - f_torus_t) ** 2))
        drift_loss = drift_loss + torus_loss
        return (drift_loss, diff_loss)

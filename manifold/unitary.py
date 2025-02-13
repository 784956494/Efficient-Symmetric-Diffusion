import torch
from utils.math_utils import complex_grad, mul, get_exact_jac_fn

class Un:
    name = 'Unitary'
    @classmethod
    def exp(cls, x, v):
        v_base = x.conj().transpose(-1, -2) @ v
        r = torch.linalg.matrix_exp(v_base)
        return x @ r

    @classmethod
    def proju(cls, x, v):
        return (v - x @ v.conj().transpose(-1, -2) @ x) / 2
    @classmethod
    def projx(cls, x):
        U, _, Vh = torch.linalg.svd(x, full_matrices=False)
        val = U @ Vh
        return val
    @classmethod
    def psi(cls, U):
        batch_size, n, _ = U.shape
        diag_entries = torch.arange(n, 0, -1, device=U.device).to(dtype=U.dtype)
        Lambda = 1 / torch.tensor(n, dtype=U.dtype, device=U.device) * torch.diag(diag_entries)
        
        U_star = U.conj().mT
        result = torch.bmm(U_star, torch.matmul(Lambda, U))  
        
        result_diag = torch.diagonal(result, dim1=-2, dim2=-1)  
        result = result - torch.diag_embed(result_diag) / 2
        upper_tri = torch.triu(result)
        return upper_tri

    @classmethod
    def phi(cls, X):
        # A = X + X.mT   
        A = torch.view_as_complex(X)  
        A = A + A.conj().mT
        eigvals, eigvecs = torch.linalg.eigh(A) 
        #ensure ordering of eigvenctors are correct
        sorted_indices = torch.argsort(eigvals.real, dim=-1, descending=True)
        eigvals = torch.gather(eigvals, -1, sorted_indices)
        eigvecs = torch.gather(eigvecs, -1, sorted_indices.unsqueeze(-2).expand_as(eigvecs))
        #ensures sign
        # first_component = eigvecs[..., 0:1, :]
        # phase = first_component / torch.abs(first_component)
        # eigvecs = eigvecs / phase
        eigvecs = (eigvecs.resolve_conj().mT)
        return (eigvals, eigvecs)
    
    @classmethod
    def sym_noise(cls, batch_size, n, device, dtype):
        xe_t = torch.randn(batch_size, n, n, dtype=dtype).reshape(-1, n, n).to(device)
        xe_t_diagonal = torch.randn(batch_size, n, device=device) + 1j * torch.randn(batch_size, n, device=device)
        xe_t = torch.triu(xe_t, diagonal=1)
        xe_t = (xe_t + xe_t.conj().mT)
        xe_t = xe_t + torch.diag_embed(xe_t_diagonal)
        return xe_t

    @classmethod
    def loss(cls, x, model_drift, model_diff, beta_scheduler=None):
        batch_size, n, n = x.shape
        t = torch.rand((x.shape[0], 1, 1), device=x.device).clamp(3e-4, 1 - 3e-4)
        if beta_scheduler:
            int_beta_r = beta_scheduler.integrate(1 - t)
        else:
            int_beta_r = 1 - t

        beta_t = beta_scheduler.step(1 - t)
        xe = cls.psi(x)
        # xe = xe + xe.conj().mT
        mu_t = (xe) * (torch.exp(-0.5 * (int_beta_r)))
        var_t = -torch.expm1(-(int_beta_r))
        # G = self.manifold.sym_noise(batch_size, n, self.device, xe)
        G = torch.randn_like(xe).triu()
        xe_t = G * var_t ** 0.5 + mu_t
        gamma, x_t = cls.phi(torch.view_as_real(xe_t))
        f_x_t = model_drift(x_t, t.squeeze(-1))
        g_x_t = model_diff(x_t, t.squeeze(-1))
        grad_log_p = -(xe_t - mu_t) / var_t
        drift_target = 0.5 * beta_t * xe_t + 2 * beta_t * grad_log_p
        with torch.enable_grad():
            xe_t = torch.view_as_real(xe_t)
            xe_t.requires_grad_(True)         
            projection = lambda X: torch.view_as_real(cls.phi(X)[1])
            grad_fn = complex_grad(projection)
            grad = grad_fn(xe_t)
        grad = grad.detach()
        xe_t = xe_t.detach()
        xe_t = torch.view_as_complex(xe_t)
        #compute drift term loss
        drift = cls.proju(x_t, mul(drift_target, grad))
        drift_loss = torch.mean(var_t * (((drift - f_x_t).abs() ** 2)))

        gamma_i = gamma.unsqueeze(-1) 
        gamma_j = gamma.unsqueeze(-2) 
        diff_target = (gamma_i - gamma_j)
        
        diff_loss = torch.mean(var_t * ((diff_target - g_x_t).abs() ** 2))
        return (drift_loss, diff_loss)

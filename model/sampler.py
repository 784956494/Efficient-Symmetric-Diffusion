import torch
from tqdm import tqdm
from manifold import Torus

class Sampler(object):
    def __init__(self, manifold, beta_scheduler, model_drift, model_diff=None, ts=0.0, tf=1.0, device='cpu'):
        self.manifold = manifold
        self.beta_scheduler = beta_scheduler
        self.model_drift = model_drift
        self.model_diff = model_diff
        self.ts = ts
        self.tf = tf
        self.device = device
        self.torus = Torus()

    def sample_torus(self, num_samples, num_time, dims, sample_phase=False):
        xe_t = torch.randn((num_samples, dims)).to(self.device)
        x_t = self.torus.phi(xe_t)
        time_pts = torch.linspace(self.ts, self.tf, num_time, device=xe_t.device)
        for i in tqdm(range(num_time - 1), total=num_time, desc="Sampling"):
            dW = torch.randn_like(xe_t, device=xe_t.device)
            t = time_pts[i].expand(xe_t.shape[0], 1)
            dt = time_pts[i + 1] - t
            if sample_phase:
                drift = self.model_drift.torus_model(torch.cat([x_t, t], dim=-1)).detach()
            else:
                drift = self.model_drift(x_t, t).detach()        
            x_h = drift * dt
            x_h = x_h + dW * (dt).sqrt() * self.beta_scheduler.step(1 - t).sqrt()
            x_t = self.torus.exp(x_t, x_h)
        return x_t
    
    def sample_orthogonal(self, num_samples, num_time, dims):
        def diff_compute(gamma, U, B):
            batch, n, n = U.shape
            A = torch.zeros_like(B)
            for i in range(n):
                col_sum = torch.zeros_like(U[:, :, i])
                for j in range(n):
                    if j != i:
                        col_sum += (B[:, i, j] / (gamma[:, i, j])).unsqueeze(-1) * U[:, :, j]
                A[:, :, i] = col_sum
            return A
        # xe_t = self.manifold.sym_noise(num_samples, dims, device=self.device, dtype=torch.float32)
        xe_t = torch.triu(torch.randn(num_samples, dims, dims)).to(self.device)
        xe_torus = torch.randn(num_samples, dims).to(self.device)
        x_torus_t = self.torus.phi(xe_torus)
        # xe_t = torch.randn(num_samples, dims, dims)
        # xe_t = xe_t.mT + xe_t
        gamma, x_t, sign = self.manifold.phi(xe_t)
        # x_t = x_t.mT
        time_pts = torch.linspace(0, 1, num_time, device=xe_t.device)
        for i in tqdm(range(num_time - 1), total=num_time, desc="Sampling"):
            # dW = self.manifold.sym_noise(num_samples, dims, device=self.device, dtype=torch.float32)
            dW = torch.randn(num_samples, dims, dims).triu()
            dW = dW + dW.mT

            dW_torus = torch.randn(num_samples, dims)

            t = time_pts[i]
            dt = time_pts[i + 1] - t

            drift, drift_torus = self.model_drift(x_t, t.expand(xe_t.shape[0], 1), x_torus_t)
            drift = drift.detach()
            drift_torus = drift_torus.detach()

            gamma = self.model_diff(x_t, t.expand(xe_t.shape[0], 1)).detach()
            diff = diff_compute(gamma, x_t, dW)
            x_h = self.manifold.proju(x_t, drift * dt)
            x_h = x_h + self.manifold.proju(x_t, diff) * (dt).sqrt() * self.beta_scheduler.step(1 - t).sqrt() 
            x_t = self.manifold.exp(x_t, x_h)
            x_t = self.manifold.projx(x_t)
            # x_t = x_t * x_t.det().unsqueeze(-1).unsqueeze(-1)

            x_h_torus = drift_torus * dt
            x_h_torus = x_h_torus + dW_torus * (dt).sqrt() * self.beta_scheduler.step(1 - t).sqrt()
            x_torus_t = self.torus.exp(x_torus_t, x_h_torus)

        x_t_sign = torch.sign(x_t[..., :, 0:1]).squeeze(1)
        x_phase = torch.cos(x_torus_t)
        x_phase = (x_phase >= 0).float() * 2 - 1
        x_t = x_t * (x_t_sign * x_phase.unsqueeze(-1))
        x_t_sign = torch.sign(x_t[..., :, 0:1]).squeeze(1)
        # print((x_t_sign - x_phase.unsqueeze(-1)).abs().max())
        return x_t
    
    def sample_unitary(self, num_samples, num_time, dims):
        def diff_compute(gamma, U, B):
            A = torch.zeros_like(B)
            for i in range(3):
                col_sum = torch.zeros_like(U[:, :, i])
                for j in range(3):
                    if j != i:
                        col_sum = col_sum + (B[:, i, j] / (gamma[:, i, j])).unsqueeze(-1) * U[:, :, j]
                A[:, :, i] = col_sum
            return A
        # xe_t = torch.view_as_real(manifold.sym_noise(num_samples, dims, args.device, v[:num_samples]))
        # xe_t = torch.view_as_real(torch.rand_like(v[:num_samples])).triu()
        xe_t = torch.randn(num_samples, dims, dims, dtype=torch.complex64).triu()
        xe_t = torch.view_as_real(xe_t)
        gamma, x_t, = self.manifold.phi(xe_t)
        time_pts = torch.linspace(0, 1, num_time, device=xe_t.device)
        beta = lambda t: 0.01 + (1.0 - 0.01) * t
        for i in tqdm(range(num_time - 1), total=num_time, desc="Sampling"):
            dW = self.manifold.sym_noise(num_samples, dims, self.device, dtype=torch.complex64)
            t = time_pts[i]
            dt = time_pts[i + 1] - t
            drift = self.model_drift(x_t, t.expand(xe_t.shape[0], 1)).detach()
            gamma = self.model_diff(x_t, t.expand(xe_t.shape[0], 1)).detach()
            diff = diff_compute(gamma, x_t, dW)
            x_h = drift * dt
            x_h = x_h + self.manifold.proju(x_t, diff) * (torch.abs(dt).sqrt()) * beta(1 - t).sqrt()
            x_t = self.manifold.exp(x_t, x_h)
            x_t = self.manifold.projx(x_t)
        return x_t
    
    def sample(self, num_samples, num_time, dims):
        if self.manifold.name == 'Torus':
            return self.sample_torus(num_samples, num_time, dims)
        elif self.manifold.name == 'Special Orthogonal':
            return self.sample_orthogonal(num_samples, num_time, dims)
        elif self.manifold.name == 'Unitary':
            return self.sample_unitary(num_samples, num_time, dims)
        else:
            raise NotImplementedError('Sampling only supported for Torus, SO(n), and U(n)...')
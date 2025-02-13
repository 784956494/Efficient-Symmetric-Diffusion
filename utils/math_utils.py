import torch
import torch.nn as nn
import math
import numpy as np

class Sin_Act(torch.nn.Module):
	def __init__(self):
		super(Sin_Act, self).__init__()

	def forward(self, x):
		return torch.sin(x)

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0,
                                             end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def get_activation(act, **kwargs):
    if act=='swish':
        return torch.nn.SiLU()
    elif act=='sin':
        return Sin_Act()
    elif act=='relu':
         return torch.nn.ReLU()
    elif act == 'log_sigmoid':
         return torch.nn.LogSigmoid()
    else:
        raise NotImplementedError(f'Activation {act} not implemented.')
    
class Divsin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x / x.sin()
        y_stable = torch.ones_like(x)
        ctx.save_for_backward(x)
        return torch.where(x.abs() < 1e-6, y_stable, y)
    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        y = (1 - x * x.cos() / x.sin()) / x.sin()
        y_stable = torch.zeros_like(x)
        return torch.where(x.abs() < 1e-6, y_stable, y) * g

# def mul(A, B):
#     C = torch.einsum('bijkl,bkl->bij', A, B)
#     return C

def mul(A, B):
    batch_size = B.size(0)
    n = B.size(1)
    C = torch.zeros((batch_size, n, n), device=A.device, dtype=A.dtype)
    for i in range(n):
        for j in range(n):
            D = A.conj().transpose(-1, -2) @ B[:, i, j, :, :]
            C[:, i, j] = D.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    return C

def get_exact_jac_fn(fn):
    def jac_fn(y):
        jac = torch.vmap(torch.func.jacrev(fn, argnums=0))(y)
        return jac
    return jac_fn

def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])

def complex_grad(fn):
    def single_fn(x_2d):
        out_2d = fn(x_2d.unsqueeze(0)).squeeze(0)
        return out_2d
    jac_single_fn = torch.func.jacrev(single_fn)
    def batched_grad_fn(X_4d):
        J = torch.vmap(jac_single_fn)(X_4d)
        partial_x_u = J[..., 0, :, :, 0]
        partial_y_u = J[..., 0, :, :, 1]
        partial_x_v = J[..., 1, :, :, 0]
        partial_y_v = J[..., 1, :, :, 1]
        df_dz_real = 0.5 * (partial_x_u + partial_y_v)
        df_dz_imag = 0.5 * (partial_x_v - partial_y_u)
        grad = torch.complex(df_dz_real, df_dz_imag)
        return grad

    return batched_grad_fn

def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result
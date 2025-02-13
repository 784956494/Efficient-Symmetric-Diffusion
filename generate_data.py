import argparse
import torch
import numpy as np
from manifold import Torus, SOn
from utils.initialize import create_dir
from utils.math_utils import matrix_diag
import abc

def generate_data(args):
    if args.manifold == 'torus':
        generate_gaussian_torus(args)
    if args.manifold == 'special_orthogonal':
        generate_gaussian_orthogonal(args)
    elif args.manifold == 'unitary':
        generate_electron(args)
    else:
        raise NotImplementedError('manifold not supported')
    return

def generate_gaussian_torus(args):
    manifold = Torus()
    num_samples = args.num_samples
    cov = 0.2 * torch.eye(args.dim)
    if args.fix_mean:
        mean = torch.zeros(args.dim)
    else:
        mean = torch.randn(args.dim)
        mean = torch.remainder(mean, 2 * torch.pi)
    data = np.random.multivariate_normal(mean, cov, num_samples)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    mean_tensor = mean
    wrapped_data_tensor = manifold.exp(mean_tensor, data_tensor).detach().cpu().numpy()
    np.save('{}T{}'.format(args.data_path, args.dim), wrapped_data_tensor)

def generate_gaussian_orthogonal(args):
    manifold = SOn()
    num_samples = args.num_samples
    n = args.dim
    dim = (n * (n - 1)) // 2
    cov = torch.eye(dim) * 0.2
    if args.fix_mean:
        tan_mean_1 = torch.zeros(dim) # mean on SO(n), given on the tangent space of identity
        tan_mean_1[-1] = torch.pi/2
        tan_mean_2 = torch.zeros(dim)
        tan_mean_2[-1] = torch.pi/2
    else:
        tan_mean_1 = torch.randn(dim)
        tan_mean_2 = torch.randn(dim)
    SO_means = []
    SO_means.append(manifold.tangent_to_son(tan_mean_1, n)) #lift mean to SO(n)
    SO_means.append(manifold.tangent_to_son(tan_mean_2, n))
    mean_in_r = torch.zeros(dim)
    all_samples = []
    for i in range(2):
        gaussian_samples = torch.from_numpy(np.random.multivariate_normal(mean_in_r, cov, size=num_samples//2)).to(dtype = torch.float32)
        so_samples = manifold.sample_son(gaussian_samples, SO_means[i], n)
        all_samples.append(so_samples)
    data_tensor = torch.cat(all_samples, dim=0)
    data_tensor = data_tensor.detach().cpu().numpy()
    np.save('{}SO{}'.format(args.data_path, args.dim), data_tensor)

def generate_electron(args):
    center_mean = 0.0
    center_var = 1.0
    angular_min = 2.0
    angular_max = 3.0
    x_span = np.linspace(0, 1, num = 1 + args.dim)[: args.dim]

    DFT_1D = np.zeros((args.dim, args.dim), dtype=np.cfloat)
    for k in range(args.dim):
        DFT_1D[k, :] = 1/np.sqrt(args.dim) * np.array([np.exp(-1j * 2 * np.pi * k/args.dim * n) for n in range(args.dim)])

    DFT_1D_dag = DFT_1D.T.conj()
    frequency = np.concatenate([np.arange(0, args.dim//2 + 1), np.arange(-args.dim//2 + 1, 0)])
    laplacian = np.diag(np.array([- 4 * np.pi ** 2 * (ii ** 2) for ii in frequency]))
    delta_h = DFT_1D_dag @ laplacian @ DFT_1D

    center = center_mean + center_var * np.random.randn(args.num_samples)
    angular = np.random.uniform(low = angular_min, high = angular_max, size = args.num_samples)
    angular = 1/2 * np.square(angular)

    batch_V_diag = x_span.reshape(-1, 1).T - center.reshape(-1,1)
    batch_V_diag = angular[:, None] * np.square(batch_V_diag)

    delta_h = torch.tensor(delta_h, dtype=torch.complex64)
    batch_V_diag = torch.tensor(batch_V_diag, dtype = torch.complex64)
    batch_V = matrix_diag(batch_V_diag)

    hamiltonian = delta_h - batch_V
    sample = 1j * 1.0 * hamiltonian
    sample = sample.to(torch.complex64)
    lie_group_element = torch.matrix_exp(sample)
    np.save('{}U{}'.format(args.data_path, args.dim), lie_group_element)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--manifold', choices=['torus', 'unitary', 'special_orthogonal'], type=str, required=True)
    parser.add_argument('--fix_mean', type=bool, default=True)
    parser.add_argument('--distribution', type=str, required=True, choices=['gaussian', 'electron'])

    args = parser.parse_args()
    create_dir(args.data_path)
    np.random.seed(args.seed)
    generate_data(args)
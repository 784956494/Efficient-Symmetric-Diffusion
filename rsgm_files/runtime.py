import torch
import torch.nn.functional as F
from typing import Callable, Tuple, Sequence
import numpy as np

def get_ode_drift_fn(model, params, states):
    def drift_fn(y: torch.Tensor, t: float) -> torch.Tensor:
        model_out, _ = model(params, states, y=y, t=t)
        return model_out

    return drift_fn

def get_exact_div_fn(fn):
    """Flatten all but the last axis and compute the true divergence."""

    def div_fn(y: torch.Tensor, t: float):
        y_shape = y.shape
        dim = np.prod(y_shape[1:])
        t = t.reshape(-1).unsqueeze(-1)
        y = y.unsqueeze(1)
        print(y.shape)
        t = t.unsqueeze(1)
        jac = torch.vmap(torch.func.jacrev(fn, argnums=0))(y, t)
        # jac = torch.autograd.functional.jacobian(fn, (y, t), create_graph=True)[0]
        jac = jac.view(y_shape[0], dim, dim)
        return jac.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


    return div_fn

def div_noise(shape: Sequence[int], hutchinson_type: str
) -> torch.Tensor:
    """Sample noise for the Hutchinson estimator."""
    if hutchinson_type == "Gaussian":
        epsilon = torch.randn(shape)
    elif hutchinson_type == "Rademacher":
        epsilon = (torch.randint(0, 2, shape, dtype=torch.float32) * 2 - 1)
    elif hutchinson_type == "None":
        epsilon = None
    else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
    
    return epsilon

def get_estimate_div_fn(fn: callable):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(y: torch.Tensor, t: float, context: torch.Tensor, eps: torch.Tensor):
        eps = eps.view(eps.shape[0], -1)
        def grad_fn(y):
            return torch.sum(fn(y, t, context) * eps)
        grad_fn_eps = torch.autograd.grad(grad_fn(y), y, create_graph=True)[0].view(y.shape[0], -1)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(eps.shape))))

    return div_fn


def get_div_fn(drift_fn, hutchinson_type: str = "None"):
    """Euclidean divergence of the drift function."""
    if hutchinson_type == "None":
        return lambda y, t, eps: get_exact_div_fn(drift_fn)(y, t)
    else:
        return lambda y, t, eps: get_estimate_div_fn(drift_fn)(
            y, t, eps
        )

def get_riemannian_div_fn(func, hutchinson_type: str = "None", manifold=None):
    """Divergence of the drift function.
    If M is submersion with Euclidean ambient metric: div = div_E
    Else (in a chart) div f = 1/sqrt(g) \sum_i \partial_i(sqrt(g) f_i)
    """
    def sqrt_g(x):
        if manifold is None or not hasattr(manifold.metric, "lambda_x"):
            return 1.0
        else:
            return manifold.metric.lambda_x(x)
    drift_fn = lambda y, t: sqrt_g(y) * func(y, t)
    div_fn = get_div_fn(drift_fn, hutchinson_type)
    return lambda y, t, eps: div_fn(y, t, eps) / sqrt_g(y)

def get_ism_loss_fn(
    pushforward,
    model,
    train: bool,
    like_w: bool = True,
    hutchinson_type="Rademacher",
    eps: float = 1e-3,
):
    sde = pushforward.sde

    def loss_fn(batch
    ) -> Tuple[float, dict]:
        score_fn = sde.reparametrise_score_fn(model, train, True)
        y_0 = pushforward.transform.inv(batch)

        t = torch.rand(y_0.shape[0]) * (sde.tf - sde.t0 - eps) + (sde.t0 + eps)

        y_t = sde.marginal_sample(y_0, t)
        score = score_fn(y_t, t)
        score = score.view(y_t.shape)

        # ISM loss
        epsilon = div_noise(y_0.shape, hutchinson_type)
        drift_fn = lambda y, t: score_fn(y, t)
        tx = time.time()
        div_fn = get_riemannian_div_fn(drift_fn, hutchinson_type='None', manifold = sde.manifold)
        div_score = div_fn(y_t, t, epsilon)
        print(time.time() - tx)
        sq_norm_score = sde.manifold.metric.squared_norm(score, y_t)
        losses = 0.5 * sq_norm_score + div_score
        if like_w:
            g2 = sde.beta_schedule.beta_t(t)
            losses = losses * g2

        loss = torch.mean(losses)
        return loss

    return loss_fn
import torch

import abc

class Transform(abc.ABC):
    def __init__(self, domain, codomain):
        self.domain = domain
        self.codomain = codomain

    @abc.abstractmethod
    def __call__(self, x):
        """Computes the transform `x => y`."""
        pass

    @abc.abstractmethod
    def inv(self, y):
        """Inverts the transform `y => x`."""
        pass

    @abc.abstractmethod
    def log_abs_det_jacobian(self, x, y):
        """Computes the log det jacobian `log |dy/dx|` given input and output."""
        pass


class Id(Transform):
    def __init__(self, domain, **kwargs):
        super().__init__(domain, domain)

    def __call__(self, x):
        return x

    def inv(self, y):
        return y

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros(x.shape[0])

class PushForward:
    """
    A density estimator able to evaluate log_prob and generate samples.
    Requires specifying a base distribution.
    Generative model: z -> y -> x ∈ M
    """

    def __init__(self, flow, base, transform=Id):
        self.flow = flow  # NOTE: Convention is that flow: data -> base
        self.base = base
        self.transform = transform

    def __repr__(self):
        return "PushForward: base:{} flow:{}".format(self.base, self.flow)

    def get_log_prob(self, model_w_dicts, train=False, transform=True, **kwargs):
        def log_prob(x, context=None):
            y = self.transform.inv(x) if transform else x
            tf = kwargs.pop("tf", 1.0)
            t0 = kwargs.pop("t0", 0.0)

            flow = self.flow.get_forward(model_w_dicts, train, augmented=True, **kwargs)
            z, inv_logdets, nfe = flow(
                y, context, tf=tf, t0=t0
            )  # NOTE: flow is not reversed
            log_prob = self.base.log_prob(z).reshape(-1)
            log_prob += inv_logdets
            if transform:
                log_prob -= self.transform.log_abs_det_jacobian(y, x)
            return torch.clamp(log_prob, -1e38, 1e38), nfe

        return log_prob

    def get_sampler(
        self, model_w_dicts, train=False, reverse=True, transform=True, **kwargs
    ):
        def sample(shape, context, z=None):
            z = self.base.sample(shape) if z is None else z
            flow = self.flow.get_forward(model_w_dicts, train, **kwargs)
            y, nfe = flow(z, context, reverse=reverse)  # NOTE: flow is reversed
            x = self.transform(y) if transform else y
            return x

        return sample
from typing import Callable, Dict
import torch

# ParametrisedScoreFunction = Callable[[Dict, Dict, torch.Tensor, float], torch.Tensor]

def get_score_fn(
    sde,
    model,
    train=False,
    std_trick=True,
    residual_trick=True,
):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
    
    Args:
      sde: An `sde.SDE` object that represents the forward SDE.
      model: A PyTorch function representing the score function model.
      params: A dictionary that contains all trainable parameters.
      states: A dictionary that contains all other mutable parameters.
      train: `True` for training and `False` for evaluation.
      return_state: If `True`, return the new mutable states alongside the model output.
      
    Returns:
      A score function.
    """

    def score_fn(y, t, rng=None):
        # Apply the model with parameters, state, and inputs
        model_out = model(y, t)
        score = model_out

        if std_trick:
            # Scaling the output with 1.0 / std
            std = sde.marginal_prob(torch.zeros_like(y), t)[1]
            score = score / std.unsqueeze(-1)  # Element-wise division
        if residual_trick:
            # Ensure time-reversal aligns with forward if NN = 0
            fwd_drift = sde.drift(y, t)
            residual = 2 * fwd_drift / sde.beta_schedule.beta_t(t).unsqueeze(-1)
            score += residual
        return score

    return score_fn
from abc import ABC, abstractmethod
class BetaSchedule(ABC):
    @abstractmethod
    def beta_t(self, t):
        pass

    @abstractmethod
    def log_mean_coeff(self, t):
        pass

    @abstractmethod
    def reverse(self):
        pass

class LinearBetaSchedule(BetaSchedule):
    def __init__(
        self,
        tf: float = 1,
        t0: float = 0,
        beta_0: float = 0.001,
        beta_f: float = 15,
    ):
        self.tf = tf
        self.t0 = t0
        self.beta_0 = beta_0
        self.beta_f = beta_f

    def log_mean_coeff(self, t):
        normed_t = (t - self.t0) / (self.tf - self.t0)
        return -0.5 * (
            0.5 * normed_t**2 * (self.beta_f - self.beta_0) + normed_t * self.beta_0
        )

    def rescale_t(self, t):
        return -2 * self.log_mean_coeff(t)

    def beta_t(self, t):
        normed_t = (t - self.t0) / (self.tf - self.t0)
        return self.beta_0 + normed_t * (self.beta_f - self.beta_0)

    def reverse(self):
        return LinearBetaSchedule(
            tf=self.t0, t0=self.tf, beta_f=self.beta_0, beta_0=self.beta_f
        )
import torch
from abc import ABC, abstractmethod
import math
class ConstantBetaSchedule(LinearBetaSchedule):
    def __init__(
        self,
        tf: float = 1,
        value: float = 1,
    ):
        super().__init__(tf=tf, t0=0.0, beta_0=value, beta_f=value)

class SDE(ABC):
    # Specify if the SDE returns full diffusion matrix, or just a scalar indicating diagonal variance

    full_diffusion_matrix = False
    spatial_dependent_diffusion = False

    def __init__(self, beta_schedule=ConstantBetaSchedule()):
        """Abstract definition of an SDE"""
        self.beta_schedule = beta_schedule
        self.tf = beta_schedule.tf
        self.t0 = beta_schedule.t0

    @abstractmethod
    def drift(self, x, t):
        """Compute the drift coefficients of the SDE at (x, t)

        Parameters
        ----------
        x : torch.Tensor
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at
        """
        pass

    def diffusion(self, x, t):
        """Compute the diffusion coefficients of the SDE at (x, t)

        Parameters
        ----------
        x : torch.Tensor
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at
        """
        beta_t = self.beta_schedule.beta_t(t)
        return torch.sqrt(beta_t)

    def coefficients(self, x, t):
        """Compute the drift and diffusion coefficients of the SDE at (x, t)

        Parameters
        ----------
        x : torch.Tensor
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at
        """
        return self.drift(x, t), self.diffusion(x, t)

    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x | x_0 = 0)$.

        Parameters
        ----------
        x : torch.Tensor
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at
        """
        raise NotImplementedError()

    def marginal_log_prob(self, x0, x, t):
        """Compute the log marginal distribution of the SDE, $log p_t(x | x_0 = 0)$.

        Parameters
        ----------
        x0: torch.Tensor
            Location of the start of the diffusion
        x : torch.Tensor
            Location of the end of the diffusion
        t : float
            Time of diffusion
        """
        raise NotImplementedError()

    def grad_marginal_log_prob(self, x0, x, t, **kwargs):
        """Compute the log marginal distribution and its gradient

        Parameters
        ----------
        x0: torch.Tensor
            Location of the start of the diffusion
        x : torch.Tensor
            Location of the end of the diffusion
        t : float
            Time of diffusion
        """
        marginal_log_prob = lambda x0, x, t: self.marginal_log_prob(x0, x, t, **kwargs)
        logp_grad_fn = torch.autograd.grad(marginal_log_prob(x0, x, t), x, create_graph=True)
        logp = marginal_log_prob(x0, x, t)
        logp_grad = logp_grad_fn[0]
        logp_grad = self.manifold.to_tangent(logp_grad, x)
        return logp, logp_grad

    def sample_limiting_distribution(self, shape):
        """Generate samples from the limiting distribution, $p_{t_f}(x)$.
        (distribution may not exist / be inexact)

        Parameters
        ----------
        shape : Tuple
            Shape of the samples to sample.
        """
        return self.limiting.sample(shape)

    def limiting_distribution_logp(self, z):
        """Compute log-density of the limiting distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: limiting distribution sample
        Returns:
          log probability density
        """
        return self.limiting.log_prob(z)

    def discretize(self, x, t, dt):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probability flow sampling.
        Defaults to Euler-Maruyama discretization.

        Parameters
        ----------
        x : torch.Tensor
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at

        Returns:
            f, G - the discretized SDE drift and diffusion coefficients
        """
        drift, diffusion = self.coefficients(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.abs(dt))
        return f, G

    def reparametrise_score_fn(self, score_fn, *args):
        return get_score_fn(self, score_fn, std_trick=True, residual_trick=True)

    def reverse(self, score_fn):
        return RSDE(self, score_fn)

    def probability_ode(self, score_fn):
        return ProbabilityFlowODE(self, score_fn)
    
class ProbabilityFlowODE:
    def __init__(self, sde: SDE, score_fn=None):
        self.sde = sde

        self.t0 = sde.t0
        self.tf = sde.tf

        if score_fn is None and not isinstance(sde, RSDE):
            raise ValueError(
                "Score function must not be None or SDE must be a reversed SDE"
            )
        elif score_fn is not None:
            self.score_fn = score_fn
        elif isinstance(sde, RSDE):
            self.score_fn = sde.score_fn

    def coefficients(self, x, t, z=None):
        drift, diffusion = self.sde.coefficients(x, t)
        score_fn = self.score_fn(x, t, z)

        # Compute G G^T score_fn
        if self.sde.full_diffusion_matrix:
            # If square matrix diffusion coefficients
            ode_drift = drift - 0.5 * torch.einsum(
                "...ij,...kj,...k->...i", diffusion, diffusion, score_fn
            )
        else:
            # If scalar diffusion coefficients (i.e., no extra dims on the diffusion)
            ode_drift = drift - 0.5 * torch.einsum(
                "...,...,...i->...i", diffusion, diffusion, score_fn
            )

        return ode_drift, torch.zeros(drift.shape[:-1])


def get_matrix_div_fn(func):
    def matrix_div_fn(x, t, context):
        # Define a function that returns div of nth column matrix function
        f = lambda n: get_exact_div_fn(lambda x, t, context: func(x, t, context)[..., n])(
            x, t, context
        )
        matrix = func(x, t, context)
        div_term = torch.stack([f(n) for n in range(matrix.shape[-1])], dim=-1)
        return div_term

    return matrix_div_fn

class RSDEBase(SDE):
    """Reverse time SDE, assuming the diffusion coefficient is spatially homogenous"""

    def __init__(self, sde: SDE, score_fn):
        super().__init__(sde.beta_schedule.reverse())
        self.sde = sde
        self.score_fn = score_fn

    def diffusion(self, x, t):
        return self.sde.diffusion(x, t)

    def drift(self, x, t):
        forward_drift, diffusion = self.sde.coefficients(x, t)
        score_fn = self.score_fn(x, t)
        
        # Compute G G^T score_fn
        if self.sde.full_diffusion_matrix:
            # If square matrix diffusion coeffs
            reverse_drift = forward_drift - torch.einsum(
                "...ij,...kj,...k->...i", diffusion, diffusion, score_fn
            )
        else:
            # If scalar diffusion coeffs (i.e., no extra dims on the diffusion)
            reverse_drift = forward_drift - torch.einsum(
                "...,...,...i->...i", diffusion, diffusion, score_fn
            )

        if self.sde.spatial_dependent_diffusion:
            # NOTE: this has not been tested
            if self.sde.full_diffusion_matrix:
                # ∇·(G G^t) = (∇· G_i G_i^t)_i =
                G_G_tr = lambda x, t, _: torch.einsum(
                    "...ij,...kj->...ik",
                    self.sde.diffusion(x, t),
                    self.sde.diffusion(x, t),
                )
                matrix_div_fn = get_matrix_div_fn(G_G_tr)
                div_term = matrix_div_fn(x, t, None)
            else:
                # ∇·(g^2 I) = (∇·g^2 1_d)_i = (||∇ g^2||_1)_i = 2 g ||∇ g||_1 1_d
                grad = torch.autograd.grad(
                    self.sde.diffusion(x, t).sum(), x, create_graph=True
                )[0]
                ones = torch.ones_like(forward_drift)
                div_term = 2 * diffusion[..., None] * grad.sum(dim=-1)[..., None] * ones
            reverse_drift += div_term

        return reverse_drift

    def reverse(self):
        return self.sde
    
class RSDE(RSDEBase):
    def __init__(self, sde: SDE, score_fn):
        super().__init__(sde, score_fn)
        self.manifold = sde.manifold

import torch

import torch
import abc
from typing import Tuple

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(
        self,
        sde: SDE,
    ):
        super().__init__()
        self.sde = sde

    @abc.abstractmethod
    def update_fn(
        self, x: torch.Tensor, t: float, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One update of the predictor.

        Args:
          rng: A PyTorch random state (here represented as a tensor for compatibility).
          x: A PyTorch tensor representing the current state.
          t: A float representing the current time step.
          dt: A float representing the time step increment.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        raise NotImplementedError()


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(
        self, x: torch.Tensor, t: float, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn_like(x)  # Equivalent to jax.random.normal
        drift, diffusion = self.sde.coefficients(x, t)
        x_mean = x + drift * dt[..., None]

        if len(diffusion.shape) > 1 and diffusion.shape[-1] == diffusion.shape[-2]:
            # If square matrix diffusion coefficients
            diffusion_term = torch.einsum(
                "...ij,j,...->...i", diffusion, z, torch.sqrt(torch.abs(dt))
            )
        else:
            # If scalar diffusion coefficients (i.e. no extra dims on the diffusion)
            diffusion_term = torch.einsum(
                "...,...i,...->...i", diffusion, z, torch.sqrt(torch.abs(dt))
            )
        x = x_mean + diffusion_term
        return x, x_mean

class EulerMaruyamaManifoldPredictor(Predictor):
    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(self, x: torch.Tensor, t: float, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        # Generate random normal tangent
        z = self.sde.manifold.random_normal_tangent(base_point=x, n_samples=x.shape[0]
        ).reshape(shape[0], -1)
        # Get drift and diffusion coefficients from SDE
        drift, diffusion = self.sde.coefficients(x.reshape(shape[0], -1), t)
        drift = drift * dt[..., None]
        # Check the shape of the diffusion matrix
        if len(diffusion.shape) > 1 and diffusion.shape[-1] == diffusion.shape[-2]:
            # If diffusion is a square matrix
            tangent_vector = drift + torch.einsum(
                "...ij,...j,...->...i", diffusion, z, torch.sqrt(torch.abs(dt))
            )
        else:
            # If diffusion is a scalar
            tangent_vector = drift + torch.einsum(
                "...,...i,...->...i", diffusion, z, torch.sqrt(torch.abs(dt))
            )
        tangent_vector = tangent_vector.reshape(shape)
        x = self.sde.manifold.exp(tangent_vec=tangent_vector, base_point=x)
        
        return x, x
class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(
        self,
        sde: SDE,
        snr: float,
        n_steps: int,
    ):
        super().__init__()
        self.sde = sde
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(
        self, x: torch.Tensor, t: float, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One update of the corrector.

        Args:
          rng: A PyTorch random state (represented as a tensor for compatibility).
          x: A PyTorch tensor representing the current state.
          t: A float representing the current time step.
          dt: A float representing the time step increment.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        raise NotImplementedError()


class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(
        self,
        sde: SDE,
        snr: float,
        n_steps: int,
    ):
        pass

    def update_fn(
        self, x: torch.Tensor, t: float, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, x
import torch

def linspace(start, stop, num, device):
    steps = torch.arange(num, dtype=torch.float32, device=device) / (num - 1)
    if not isinstance(start, torch.Tensor):
        start = start * torch.ones_like(stop)
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    return out

def expand_dims_pytorch(x, axes):
    for axis in sorted(axes):
        x = x.unsqueeze(axis)
    return x

def get_pc_sampler(
    sde: SDE,
    N: int,
    predictor: str = "EulerMaruyamaPredictor",
    corrector: str = None,
    inverse_scaler=lambda x: x,
    snr: float = 0.2,
    n_steps: int = 1,
    denoise: bool = True,
    eps: float = 1e-3,
    return_hist=False,
):
    """Create a Predictor-Corrector (PC) sampler."""
    if predictor == 'EulerMaruyamaPredictor':
        predictor = EulerMaruyamaPredictor(sde)
    elif predictor == 'GRW':
        predictor = EulerMaruyamaManifoldPredictor(sde)
    corrector = NoneCorrector(sde, snr, n_steps)

    def pc_sampler(device, x, t0=None, tf=None):
        """The PC sampler function."""
        
        t0 = sde.t0 if t0 is None else t0
        tf = sde.tf if tf is None else tf
        if not torch.is_tensor(t0):
            t0 = torch.broadcast_to(torch.tensor([t0]), (x.shape[0],))
        if not torch.is_tensor(tf):
            tf = torch.broadcast_to(torch.tensor([tf]), (x.shape[0],))

        # Only integrate to eps off the forward start time for numerical stability
        if isinstance(sde, RSDE):
            tf = tf + eps
        else:
            t0 = t0 + eps
        timesteps = linspace(t0, tf, N, device=device)
        dt = (tf - t0) / N

        x_hist = torch.zeros((N, *x.shape), device=device)

        def loop_body(i, x, x_mean, x_hist):
            t = timesteps[i]

            x, x_mean = corrector.update_fn(x, t, dt)
            x, x_mean = predictor.update_fn(x, t, dt)

            x_hist[i] = x
            return x, x_mean, x_hist
        x_mean = x.clone()  
        for i in range(N):
            x, x_mean, x_hist = loop_body(i, x, x_mean, x_hist)

        if return_hist:
            return (
                inverse_scaler(x_mean if denoise else x),
                inverse_scaler(x_hist),
                timesteps,
            )
        else:
            return inverse_scaler(x_mean if denoise else x)

    return pc_sampler
import torch
import numpy as np
from torchdiffeq import odeint  # Assuming you're using torchdiffeq for ODE integration
from typing import Sequence

class ReverseAugWrapper:
    def __init__(self, module, tf):
        self.module = module
        self.tf = tf

    def __call__(
        self, y: torch.Tensor, t: torch.Tensor, context: torch.Tensor, *args, **kwargs
    ):
        states = self.module(y, self.tf - t, context, *args, **kwargs)
        
        return torch.cat([-states[..., :-1], states[..., [-1]]], dim=1)
    
class ReverseWrapper:
    def __init__(self, module, tf):
        self.module = module
        self.tf = tf

    def __call__(
        self, y: torch.Tensor, t: torch.Tensor, context: torch.Tensor, *args, **kwargs
    ):
        states = self.module(y, self.tf - t, context, *args, **kwargs)
        return -states

class CNF:
    def __init__(
        self,
        t0: float = 0,
        tf: float = 1,
        hutchinson_type: str = "None",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        get_drift_fn=get_ode_drift_fn,
        manifold=None,
        **kwargs,
    ):
        self.get_drift_fn = get_drift_fn
        self.t0 = t0
        self.tf = tf
        self.ode_kwargs = dict(atol=atol, rtol=rtol)
        self.test_ode_kwargs = dict(atol=1e-5, rtol=1e-5)
        self.hutchinson_type = hutchinson_type
        self.manifold = manifold

    def get_forward(self, model_w_dicts, train, augmented=False, **kwargs):
        model, params, states = model_w_dicts

        def forward(data, context=None, t0=None, tf=None, rng=None, reverse=False):
            hutchinson_type = self.hutchinson_type if train else "None"
            
            shape = data.shape
            epsilon = div_noise(rng, shape, hutchinson_type)
            t0 = self.t0 if t0 is None else t0
            tf = self.tf if tf is None else tf
            eps = kwargs.get("eps", 1e-3)
            ts = torch.tensor([t0 + eps, tf], device=data.device)
            ode_kwargs = self.ode_kwargs if train else self.test_ode_kwargs

            if augmented:  # Solving for the change in log-likelihood

                def ode_func(
                    y: torch.Tensor, t: torch.Tensor, context: torch.Tensor, params, states
                ) -> torch.Tensor:
                    sample = y[:, :-1]
                    vec_t = torch.ones((sample.shape[0],), device=t.device) * t
                    drift_fn = self.get_drift_fn(model, params, states)
                    drift = drift_fn(sample, vec_t, context)
                    div_fn = get_riemannian_div_fn(
                        drift_fn, hutchinson_type, self.manifold
                    )
                    logp_grad = div_fn(sample, vec_t, context, epsilon).reshape(
                        [shape[0], 1]
                    )
                    return torch.cat([drift, logp_grad], dim=1)

                data = data.view(shape[0], -1)
                init = torch.cat([data, torch.zeros((shape[0], 1), device=data.device)], dim=1)
                ode_func = ReverseAugWrapper(ode_func, tf) if reverse else ode_func
                y, nfe = odeint(
                    ode_func, init, ts, context, params, states, **ode_kwargs
                )
                z = y[-1, ..., :-1].view(shape)
                delta_logp = y[-1, ..., -1]
                return z, delta_logp, nfe
            else:

                def ode_func(
                    y: torch.Tensor, t: torch.Tensor, context: torch.Tensor, params, states
                ) -> torch.Tensor:
                    sample = y
                    vec_t = torch.ones((sample.shape[0],), device=t.device) * t
                    drift_fn = self.get_drift_fn(model, params, states)
                    drift = drift_fn(sample, vec_t, context)
                    return drift

                data = data.view(shape[0], -1)
                init = data
                ode_func = ReverseWrapper(ode_func, tf) if reverse else ode_func
                y, nfe = odeint(
                    ode_func, init, ts, context, params, states, **ode_kwargs
                )
                z = y[-1].view(shape)
                return z, nfe

        return forward
import torch
from functools import partial
import torch.nn as nn
def get_sde_drift_from_fn(sde: SDE, model, params, states):
    def drift_fn(y: torch.Tensor, t: float, context: torch.Tensor) -> torch.Tensor:
        """The drift function of the reverse-time SDE."""
        score_fn = sde.reparametrise_score_fn(model, train=False)
        pode = sde.probability_ode(score_fn)
        return pode.coefficients(y, t, context)[0]

    return drift_fn


class SDEPushForward(PushForward):
    def __init__(self, flow: SDE, base, diffeq="sde", transform=Id):
        self.sde = flow
        self.diffeq = diffeq
        flow = CNF(
            t0=self.sde.t0,
            tf=self.sde.tf,
            get_drift_fn=partial(get_sde_drift_from_fn, self.sde),
            manifold=flow.manifold,
        )
        super(SDEPushForward, self).__init__(flow, base, transform)

    def get_sampler(
        self, model, train=False, reverse=True, transform=True, **kwargs
    ):
        if self.diffeq == "ode":  # via probability flow
            sample = super().get_sampler(model, train, reverse)
        elif self.diffeq == "sde":  # via stochastic process

            def sample(device, shape, z=None):
                z = self.base.sample(shape).to(device) if z is None else z
                score_fn = self.sde.reparametrise_score_fn(model)
                score_fn = partial(score_fn)
                sde = self.sde.reverse(score_fn) if reverse else self.sde
                sampler = get_pc_sampler(sde, **kwargs)

                y = sampler(device, z)
                x = self.transform(y) if transform else y
                return x

        else:
            raise ValueError(self.diffeq)
        return sample
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class MultivariateNormalDiag(MultivariateNormal):
    def __init__(self, dim, mean=None, scale=None, **kwargs):
        mean = torch.zeros((dim)) if mean is None else mean
        scale = torch.ones((dim)) if scale is None else scale
        covariance_matrix = torch.diag(scale**2)
        super().__init__(mean, covariance_matrix)

    def sample(self, rng, shape):
        return super().sample(sample_shape=shape)

    def log_prob(self, z):
        return super().log_prob(z)

    def grad_U(self, x):
        return x / (self.covariance_matrix.diag()**2)


class Langevin(SDE):
    """Construct Langevin dynamics on a manifold"""

    def __init__(self, beta_schedule, manifold, ref_scale=0.5, ref_mean=None, N=100):
        self.beta_schedule = beta_schedule
        self.manifold = manifold
        self.limiting = MultivariateNormalDiag(dim=2)
        self.N = N

    def fixed_grad(self, grad):
        """Replace NaNs or Infs in the gradient with zeros."""
        is_nan_or_inf = torch.isnan(grad) | (torch.abs(grad) == float('inf'))
        return torch.where(is_nan_or_inf, torch.zeros_like(grad), grad)

    def drift(self, x, t):
        """dX_t =-0.5 beta(t) grad U(X_t)dt + sqrt(beta(t)) dB_t"""
        # Vectorized function for drift computation
        drift_fn = lambda x: -0.5 * self.fixed_grad(self.limiting.grad_U(x))
        beta_t = self.beta_schedule.beta_t(t)

        # Apply the vectorized drift function to each element in the batch
        drift = beta_t.unsqueeze(-1) * torch.stack([drift_fn(xi) for xi in x], dim=0)
        return drift

    def marginal_sample(self, x, t, return_hist=False):
        out = self.manifold.random_walk(x, self.beta_schedule.rescale_t(t))
        if return_hist or out is None:
            sampler = get_pc_sampler(self, self.N, predictor="GRW", return_hist=return_hist)
            out = sampler('cpu', x, tf=t)
        return out

    def marginal_prob(self, x, t):
        """Proxy for the marginal probability (mean and standard deviation)."""
        log_mean_coeff = self.beta_schedule.log_mean_coeff(t)
        axis_to_expand = tuple(range(-1, -len(x.shape), -1))  # (-1) or (-1, -2)
        mean_coeff = expand_dims_pytorch(torch.exp(log_mean_coeff), axis_to_expand)
        # mean_coeff = torch.exp(log_mean_coeff).expand(axis_to_expand)
        mean = mean_coeff * x
        std = torch.sqrt(1 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def varhadan_exp(self, xs, xt, s, t):
        delta_t = self.beta_schedule.rescale_t(t) - self.beta_schedule.rescale_t(s)
        axis_to_expand = tuple(range(-1, -len(xt.shape), -1))  # (-1) or (-1, -2)
        delta_t = expand_dims_pytorch(delta_t, axis_to_expand)
        # delta_t = delta_t.unsqueeze(axis_to_expand)
        grad = self.manifold.log(xs, xt) / delta_t
        return delta_t, grad

    def reverse(self, score_fn):
        return RSDE(self, score_fn)
class UniformDistribution:
    """Uniform density on compact manifold"""

    def __init__(self, manifold, **kwargs):
        self.manifold = manifold

    def sample(self, shape):
        return self.manifold.random_uniform(n_samples=shape[0])

    def log_prob(self, z):
        return -torch.ones(z.shape[0]) * self.manifold.log_volume

    def grad_U(self, x):
        return torch.zeros_like(x)
class Brownian(Langevin):
    def __init__(self, manifold, beta_schedule, N=100):
        """Construct a Brownian motion on a compact manifold"""
        # super().__init__(beta_schedule, manifold, N=N)
        self.manifold = manifold
        self.limiting = UniformDistribution(manifold)
        self.N = N

        self.beta_schedule = beta_schedule
        self.tf = beta_schedule.tf
        self.t0 = beta_schedule.t0

    def grad_marginal_log_prob(self, x0, x, t, **kwargs):
        s = self.beta_schedule.rescale_t(t)
        logp_grad = self.manifold.grad_marginal_log_prob(x0, x, s, thresh=0.5, n_max=5)
        return None, logp_grad

    def reparametrise_score_fn(self, score_fn, *args):
        return get_score_fn(self, score_fn, std_trick=True, residual_trick=False)
class DefaultDistribution:
    def __new__(cls, manifold, flow, **kwargs):
        if isinstance(flow, SDE):
            return flow.limiting
        else:
            if hasattr(manifold, "random_uniform"):
                return UniformDistribution(manifold)
            else:
                # TODO: WrappedNormal (if applicable)
                raise NotImplementedError(f"No default distribution for {manifold}")
def get_activation(act, **kwargs):
    if act=='swish':
        return torch.nn.SiLU()
    elif act=='sin':
        return Sin_Act()
    elif act=='relu':
         return torch.nn.ReLU()
    elif act == 'log sigmoid':
         return torch.nn.LogSigmoid()
    else:
        raise NotImplementedError(f'Activation {act} not implemented.')

class Sin_Act(torch.nn.Module):
	def __init__(self):
		super(Sin_Act, self).__init__()

	def forward(self, x):
		return torch.sin(x)


class MLP(nn.Module):
    def __init__(self, hidden_shapes, output_shape, act, bias=True):
        super(MLP, self).__init__()
        self.hidden_shapes = hidden_shapes
        self.output_shape = output_shape
        self.act = act
        self.bias = bias

        # Define layers using nn.ModuleList
        layers = []
        for i in range(len(self.hidden_shapes) - 1):
            layers.append(nn.Linear(hidden_shapes[i], hidden_shapes[i + 1], bias=self.bias))
            layers.append(get_activation(self.act))
        
        # Final layer without activation
        layers.append(nn.Linear(hidden_shapes[-1], output_shape, bias=self.bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)    

class Concat(nn.Module):
    def __init__(self, output_shape, hidden_shapes, act):
        super().__init__()
        self._layer = MLP(hidden_shapes=hidden_shapes, output_shape=output_shape, act=act)

    def forward(self, x, t):
        if len(t.shape) == 0:
            t = t * torch.ones(x.shape[:-1], device=x.device)
        if len(t.shape) == len(x.shape) - 1:
            t = torch.unsqueeze(t, dim=-1)
            
        return self._layer(torch.cat([x, t], dim=-1))
class VectorFieldGenerator(nn.Module, abc.ABC):
    def __init__(self, model, embedding, manifold):
        """X = fi * Xi with fi weights and Xi generators"""
        super(VectorFieldGenerator, self).__init__()
        self.net = model
        self.embedding = embedding
        self.manifold = manifold

    @staticmethod
    @abc.abstractmethod
    def output_shape(manifold):
        """Cardinality of the generating set."""

    def _weights(self, x, t):
        """shape=[..., card=n]"""
        return self.net(*self.embedding(x, t))

    @abc.abstractmethod
    def _generators(self, x):
        """Set of generating vector fields: shape=[..., d, card=n]"""

    @property
    def decomposition(self):
        return lambda x, t: self._weights(x, t), lambda x: self._generators(x)

    def forward(self, x, t):
        fi_fn, Xi_fn = self.decomposition
        fi, Xi = fi_fn(x, t), Xi_fn(x)
        out = torch.einsum("...n,...dn->...d", fi, Xi)
        # NOTE: seems that extra projection is required for generator=eigen
        out = self.manifold.to_tangent(out, x)
        return out

    def div_generators(self, x):
        """Divergence of the generating vector fields: shape=[..., card=n]"""

class TorusGenerator(VectorFieldGenerator):
    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, manifold)

        self.rot_mat = torch.tensor([[0, -1], [1, 0]], dtype=torch.float64)

    @staticmethod
    def output_shape(manifold):
        return manifold.dim

    def _generators(self, x):
        return (
            torch.matmul(self.rot_mat, x.reshape((*x.shape[:-1], self.manifold.dim, 2)))[..., None]
        )[..., 0]

    def forward(self, x, t):
        weights_fn, fields_fn = self.decomposition
        weights = weights_fn(x, t)
        fields = fields_fn(x)
        return (fields * weights[..., None]).reshape(
            (*x.shape[:-1], self.manifold.dim * 2)
        )
    
class LieAlgebraGenerator(VectorFieldGenerator):
    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, manifold)

    @staticmethod
    def output_shape(manifold):
        return manifold.dim

    def _generators(self, x):
        return self.manifold.lie_algebra.basis

    def __call__(self, x, t):
        x = x.reshape((x.shape[0], self.manifold.n, self.manifold.n))
        fi_fn, Xi_fn = self.decomposition
        x_input = x.reshape((*x.shape[:-2], -1))
        fi, Xi = fi_fn(x_input, t), Xi_fn(x)
        out = torch.einsum("...i,ijk ->...jk", fi, Xi)
        out = self.manifold.compose(x, out)
        # out = self.manifold.to_tangent(out, x)
        return out.reshape((x.shape[0], -1))
class Embedding(nn.Module, abc.ABC):
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold


class NoneEmbedding(Embedding):
    def __call__(self, x, t):
        return x, t
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def get_loss_step_fn(
    loss_fn,
    train: bool,
):
    def step_fn(train_state, batch: dict):
        # Compute gradients with torch.autograd.grad
        if train:
            model_state = train_state.model_state
            optimizer = train_state.opt_state

            # Compute loss and gradients
            loss = loss_fn(batch)
            optimizer.zero_grad()
            loss.backward()
            print('here')
            # Perform optimizer step
            torch.nn.utils.clip_grad_norm_(model_state.parameters(), 1.0)
            optimizer.step()

            # Update training state
            step = train_state.step + 1
            new_train_state = train_state._replace(
                step=step,
                model_state=model_state,
                opt_state=optimizer
            )
        else:
            loss, _ = loss_fn(batch)
            new_train_state = train_state

        # Return new carry state and loss
        new_carry_state = new_train_state
        return new_carry_state, loss.detach()

    return step_fn

def to_skew_symm(v, n):
    batch_size = v.shape[0]
    A = torch.zeros((batch_size, n, n), dtype=v.dtype, device=v.device)
    indices = torch.triu_indices(n, n, offset=1)
    A[:, indices[0], indices[1]] = v
    A[:, indices[1], indices[0]] = -v
    return A

from geomstats.geometry.special_orthogonal import SpecialOrthogonal
beta_schedule = LinearBetaSchedule()
n = 50
manifold = SpecialOrthogonal(n=n, point_type='matrix')
transform = Id(manifold)
flow = Brownian(manifold=manifold, beta_schedule=beta_schedule)
base = DefaultDistribution(manifold=manifold, flow=flow)
pushforward = SDEPushForward(flow, base, diffeq='sde', transform=transform)
import torch.optim as optim
embedding = NoneEmbedding(manifold)
dim = n * (n - 1) // 2
model = Concat(dim, [n ** 2 + 1, 512, 512, 512, 512], 'sin')
score = LieAlgebraGenerator(model, embedding, dim, manifold)
optimizer = optim.Adam(score.parameters(), lr=2e-4)
from collections import namedtuple


TrainState = namedtuple(
    "TrainState",
    [
        "opt_state",
        "model_state",
        "step",
    ],
)

train_state = TrainState(
            opt_state=optimizer,
            model_state=score,
            step=0
        )
loss = get_ism_loss_fn(pushforward=pushforward, model=score, eps=1e-8, train=True, hutchinson_type="None")
train_step_fn = get_loss_step_fn(loss, True)
v = torch.rand(1, dim)
MMM = to_skew_symm(v, n)

from tqdm import tqdm
from timeit import default_timer as timer
import sys
import time

# file = open("output_log.txt", "a")
# sys.stdout = file

try:
    batch = MMM
    print(batch.shape)
    train_time = time.time()
    train_state, loss_val = train_step_fn(train_state, batch)
    step = train_state.step
    time_passed = time.time() - train_time 
    print(str(time_passed))
except Exception as e:
    # Open a text file in write mode
    with open("error_log.txt", "w") as file:
        # Write the error message to the file
        file.write(str(e))
        # Optionally, write the traceback for more detailed info
        import traceback
        file.write("\n\n")
        traceback.print_exc(file=file)
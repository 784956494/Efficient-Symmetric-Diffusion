import abc
import torch
import numpy as np
import geomstats.backend as gs
from functools import partial
from geomstats.geometry.manifold import Manifold

class VectorSpace(Manifold, abc.ABC):
    """Abstract class for vector spaces.

    Parameters
    ----------
    shape : tuple
        Shape of the elements of the vector space. The dimension is the
        product of these values by default.
    default_point_type : str, {'vector', 'matrix'}
        Point type.
        Optional, default: 'vector'.
    """

    def __init__(self, shape, default_point_type="vector", **kwargs):
        if "dim" not in kwargs.keys():
            kwargs["dim"] = int(np.prod(np.array(shape)))
        super(VectorSpace, self).__init__(default_point_type=default_point_type, **kwargs)
        self.shape = shape

    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the vector space.

        This method checks the shape of the input point.

        Parameters
        ----------
        point : array-like, shape=[.., {dim, [n, n]}]
            Point to test.
        atol : float
            Unused here.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        if self.default_point_type == "vector":
            point_shape = point.shape[-1:]
            minimal_ndim = 1
        else:
            point_shape = point.shape[-2:]
            minimal_ndim = 2
        belongs = point_shape == self.shape
        if point.ndim == minimal_ndim:
            return belongs
        return gs.tile(gs.array([belongs]), [point.shape[0]])

    def belongs_vmap(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the vector space.

        This method checks the shape of the input point.

        Parameters
        ----------
        point : array-like, shape=[.., {dim, [n, n]}]
            Point to test.
        atol : float
            Unused here.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        if self.default_point_type == "vector":
            point_shape = point.shape[-1:]
            minimal_ndim = 1
        else:
            point_shape = point.shape[-2:]
            minimal_ndim = 2
        belongs = point_shape == self.shape
        if point.ndim == minimal_ndim:
            return belongs
        return gs.tile(torch.tensor([belongs], device=point.device), [point.shape[0]])

    @staticmethod
    def projection(point):
        """Project a point to the vector space.

        This method is for compatibility and returns `point`. `point` should
        have the right shape,

        Parameters
        ----------
        point: array-like, shape[..., {dim, [n, n]}]
            Point.

        Returns
        -------
        point: array-like, shape[..., {dim, [n, n]}]
            Point.
        """
        return point

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Tangent vectors are identified with points of the vector space so
        this checks the shape of the input vector.

        Parameters
        ----------
        vector : array-like, shape=[..., {dim, [n, n]}]
            Vector.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Point in the vector space.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        return self.belongs(vector)

    def is_tangent_vmap(self, vector, base_point=None, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Tangent vectors are identified with points of the vector space so
        this checks the shape of the input vector.

        Parameters
        ----------
        vector : array-like, shape=[..., {dim, [n, n]}]
            Vector.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Point in the vector space.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        return self.belongs_vmap(vector)

    def to_tangent(self, vector, base_point=None):
        """Project a vector to a tangent space of the vector space.

        This method is for compatibility and returns vector.

        Parameters
        ----------
        vector : array-like, shape=[..., {dim, [n, n]}]
            Vector.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Point in the vector space

        Returns
        -------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at base point.
        """
        return vector

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in the vector space with a uniform distribution in a box.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1.0

        Returns
        -------
        point : array-like, shape=[..., dim]
           Sample.
        """
        size = self.shape
        if n_samples != 1:
            size = (n_samples,) + self.shape
        point = bound * (gs.random.rand(*size) - 0.5) * 2
        return point

    def random_normal_tangent(self, base_point, n_samples=1):
        """Sample in the tangent space from the standard normal distribution.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        # return gs.random.normal(size=(n_samples, self.dim))
        return torch.randn((n_samples, self.dim), device=base_point.device)


class EmbeddedManifold(Manifold, abc.ABC):
    """Class for manifolds embedded in a vector space.

    Parameters
    ----------
    dim : int
        Dimension of the embedded manifold.
    embedding_space : VectorSpace
        Embedding space.
    default_coords_type : str, {'intrinsic', 'extrinsic', etc}
        Coordinate type.
        Optional, default: 'intrinsic'.
    """

    def __init__(
        self,
        dim,
        embedding_space,
        submersion,
        value,
        tangent_submersion,
        default_coords_type="intrinsic",
        **kwargs
    ):
        super(EmbeddedManifold, self).__init__(
            dim=dim,
            default_point_type=embedding_space.default_point_type,
            default_coords_type=default_coords_type,
            **kwargs
        )
        self.embedding_space = embedding_space
        self.embedding_metric = embedding_space.metric
        self.submersion = submersion
        if isinstance(value, float):
            value = gs.array(value)
        self.value = value
        self.tangent_submersion = tangent_submersion

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        value = self.value
        if isinstance(point, torch.Tensor):
            belongs = self.embedding_space.belongs_vmap(point, atol)
            value = torch.tensor(value).float().to(point.device)
        else:
            belongs = self.embedding_space.belongs(point, atol)
        if not gs.any(belongs):
            return belongs
        constraint = gs.isclose(self.submersion(point), value, atol=atol)
        if value.ndim == 2:
            constraint = gs.all(constraint, axis=(-2, -1))
        elif value.ndim == 1:
            constraint = gs.all(constraint, axis=-1)
        return gs.logical_and(belongs, constraint)

    def belongs_vmap(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        belongs = self.embedding_space.belongs_vmap(point, atol)
        if not gs.any(belongs):
            return belongs
        value = self.value
        constraint = gs.isclose(self.submersion(point), value, atol=atol)
        if value.ndim == 2:
            constraint = gs.all(constraint, axis=(-2, -1))
        elif value.ndim == 1:
            constraint = gs.all(constraint, axis=-1)
        return gs.logical_and(belongs, constraint)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        belongs = self.embedding_space.is_tangent(vector, base_point, atol)
        tangent_sub_applied = self.tangent_submersion(vector, base_point)
        constraint = gs.isclose(tangent_sub_applied, 0.0, atol=atol)
        value = self.value
        if value.ndim == 2:
            constraint = gs.all(constraint, axis=(-2, -1))
        elif value.ndim == 1:
            constraint = gs.all(constraint, axis=-1)
        return gs.logical_and(belongs, constraint)

    def is_tangent_vmap(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        belongs = self.embedding_space.is_tangent_vmap(vector, base_point, atol)
        tangent_sub_applied = self.tangent_submersion(vector, base_point)
        constraint = gs.isclose(tangent_sub_applied, 0.0, atol=atol)
        value = self.value
        if value.ndim == 2:
            constraint = gs.all(constraint, axis=(-2, -1))
        elif value.ndim == 1:
            constraint = gs.all(constraint, axis=-1)
        return gs.logical_and(belongs, constraint)

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """Convert from intrinsic to extrinsic coordinates.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[..., dim]
            Point in the embedded manifold in intrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[..., dim_embedding]
            Point in the embedded manifold in extrinsic coordinates.
        """
        raise NotImplementedError("intrinsic_to_extrinsic_coords is not implemented.")

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """Convert from extrinsic to intrinsic coordinates.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[..., dim_embedding]
            Point in the embedded manifold in extrinsic coordinates,
            i. e. in the coordinates of the embedding manifold.

        Returns
        -------
        point_intrinsic : array-lie, shape=[..., dim]
            Point in the embedded manifold in intrinsic coordinates.
        """
        raise NotImplementedError("extrinsic_to_intrinsic_coords is not implemented.")

    @abc.abstractmethod
    def projection(self, point):
        """Project a point in embedding manifold on embedded manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim_embedding]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., dim_embedding]
            Projected point.
        """

    @abc.abstractmethod
    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """

    def random_normal_tangent(self, base_point, n_samples=1):
        """Sample in the tangent space from the standard normal distribution.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        # ambiant_noise = gs.random.normal(
        #     size=(n_samples, self.embedding_space.dim)
        # ).to(base_point.device)
        ambiant_noise = torch.randn((n_samples, self.embedding_space.dim), 
                            device=base_point.device)
        return self.to_tangent(vector=ambiant_noise, base_point=base_point)

    def log_heat_kernel_exp(self, x0, x, t):
        t = t / 2  # NOTE: to match random walk
        r_squared = self.metric.squared_dist(x0, x)
        log_u_0 = -0.5 * self.metric.log_metric_polar(r_squared)
        return (
            -self.dim / 2 * gs.log(4 * gs.pi)
            - self.dim / 2 * gs.log(t)
            - r_squared / (4 * t)
            + log_u_0
        )

    def log_heat_kernel(self, x0, x, t, thresh, n_max):
        # TODO: How to choose condition? should condition be on radius not on time?
        print("ahhhhhhhhhhhh")
        cond = t <= thresh
        approx = self.log_heat_kernel_exp(x0, x, t)
        print('log heat kernel approx:')
        print(approx.isinf().any())
        print(approx.isnan().any())
        exact = self._log_heat_kernel(x0, x, t, n_max=n_max)
        print('log heat kernel exact:')
        print(exact.isinf().any())
        print(exact.isnan().any())
        ret = gs.where(cond, approx, exact)
        print('log heat kernel ret:')
        print(ret.isinf().any())
        print(ret.isnan().any())
        return ret

    # Varadhan asymptotic
    def grad_log_heat_kernel_exp(self, x0, x, t):
        exp_inv = self.metric.log(x0, x)
        return exp_inv / gs.expand_dims(t, -1)
    
    def grad_marginal_helper(self, x0, x, t, thresh, n_max):
        # x.requires_grad_(True)
        # print(x)
        # assert x.value.requires_grad
        log_heat_kernel = lambda y: (self._log_heat_kernel(x0, y, t, n_max=n_max)).sum()
        # logp_grad = torch.autograd.grad(a, x, retain_graph=True, create_graph=True)[0]
        logp_grad = torch.func.jacrev(log_heat_kernel)(x)
        print('grad marginal helper logp_grad:')
        print(logp_grad.isinf().any())
        print(logp_grad.isnan().any())
        cond = gs.expand_dims(t <= thresh, -1)
        print('grad marginal helper cond:')
        print(cond.isinf().any())
        print(cond.isnan().any())
        approx = self.grad_log_heat_kernel_exp(x0, x, t)
        print('grad marginal helper approx:')
        print(approx.isinf().any())
        print(approx.isnan().any())
        exact = self.to_tangent(logp_grad, x)
        print('grad marginal helper exact:')
        print(exact.isinf().any())
        print(exact.isnan().any())
        ret = gs.where(cond, approx, exact)
        print('grad marginal helper ret:')
        print(ret.isinf().any())
        print(ret.isnan().any())
        return ret

    def grad_marginal_log_prob(self, x0, x, t, thresh, n_max):
        log_heat_kernel = lambda y: self._log_heat_kernel(x0, y, t, n_max=n_max)
        with torch.enable_grad():
            x.requires_grad_()
            logp_grad = torch.autograd.grad(log_heat_kernel(x).sum(), x)[0]
        cond = gs.expand_dims(t <= thresh, -1)
        approx = self.grad_log_heat_kernel_exp(x0, x, t)
        x.requires_grad_(False)
        exact = self.to_tangent(logp_grad, x)
        return gs.where(cond, approx, exact)


class OpenSet(Manifold, abc.ABC):
    """Class for manifolds that are open sets of a vector space.

    In this case, tangent vectors are identified with vectors of the ambient
    space.

    Parameters
    ----------
    dim: int
        Dimension of the manifold. It is often the same as the ambient space
        dimension but may differ in some cases.
    ambient_space: VectorSpace
        Ambient space that contains the manifold.
    """

    def __init__(self, dim, ambient_space, **kwargs):
        if "default_point_type" not in kwargs:
            kwargs["default_point_type"] = ambient_space.default_point_type
        super().__init__(dim=dim, **kwargs)
        self.ambient_space = ambient_space

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        return self.ambient_space.belongs(vector, atol)

    def to_tangent(self, vector, base_point=None):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        return self.ambient_space.projection(vector)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample random points on the manifold.

        If the manifold is compact, a uniform distribution is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample for non compact manifolds.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., {dim, [n, n]}]
            Points sampled on the hypersphere.
        """
        sample = self.ambient_space.random_point(n_samples, bound)
        return self.projection(sample)

    @abc.abstractmethod
    def projection(self, point):
        """Project a point in ambient manifold on manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in ambient manifold.

        Returns
        -------
        projected : array-like, shape=[..., dim]
            Projected point.
        """

    def random_normal_tangent(self, base_point, n_samples=1):
        """Sample in the tangent space from the standard normal distribution.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        # ambiant_noise = gs.random.normal(
        #     size=(n_samples, self.ambient_space.dim)
        # ).to(base_point.device)
        ambiant_noise = torch.randn((n_samples, self.ambient_space.dim), 
                            device=base_point.device)
        return ambiant_noise
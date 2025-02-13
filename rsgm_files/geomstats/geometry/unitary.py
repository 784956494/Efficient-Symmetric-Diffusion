"""Exposes the `SpecialOrthogonal` group class."""

import torch
import math
import numpy as np
import geomstats.backend as gs
import geomstats.errors
import geomstats.vectorization
import geomstats.algebra_utils as utils

from geomstats.geometry.base import EmbeddedManifold
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.invariant_metric import BiInvariantMetric
from geomstats.geometry.lie_group import LieGroup, MatrixLieGroup
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.geometry.utils import set_diag

ATOL = 1e-5

TAYLOR_COEFFS_1_AT_PI = [
    0.0,
    -gs.pi / 4.0,
    -1.0 / 4.0,
    -gs.pi / 48.0,
    -1.0 / 48.0,
    -gs.pi / 480.0,
    -1.0 / 480.0,
]


class _UnitaryMatrices(MatrixLieGroup, EmbeddedManifold):
    """Class for uintary groups in matrix representation.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        matrices = Matrices(n, n)
        gln = GeneralLinear(n, positive_det=True)
        super(_UnitaryMatrices, self).__init__(
            dim=int((n * (n - 1)) / 2),
            n=n,
            value=gs.eye(n),
            lie_algebra=SkewHermitianMatrices(n=n),
            embedding_space=gln,
            submersion=lambda x: matrices.mul(matrices.transpose(x), x),
            tangent_submersion=lambda v, x: 2
            * matrices.to_symmetric(matrices.mul(matrices.transpose(x), v)),
        )
        self.bi_invariant_metric = BiInvariantMetric(group=self)
        self.metric = self.bi_invariant_metric

    @classmethod
    def inverse(cls, point):
        """Return the transpose matrix of point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]

        Returns
        -------
        inverse : array-like, shape=[..., n, n]
            Inverse.
        """
        return Matrices.transpose(point)

    #NOTE: modified to incorporate flattened data
    def projection(self, point):
        """Project a matrix on U(n) by minimizing the Frobenius norm.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        NOTE: the input vector and base_point may be flattened

        Returns
        -------
        rot_mat : array-like, shape=[..., n, n]
            Rotation matrix.
        """
        shape = point.shape
        point = point.reshape(-1, self.n, self.n)
        aux_mat = self.submersion(point)
        # aux_mat = Matrices.mul(Matrices.transpose(point), point)
        inv_sqrt_mat = SymmetricMatrices.powerm(aux_mat, -1 / 2)
        rotation_mat = Matrices.mul(point, inv_sqrt_mat)
        det = gs.linalg.det(rotation_mat)
        return utils.flip_determinant(rotation_mat, det).reshape(shape)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in U(n) from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Unused.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
\        """
        return self.random_uniform(n_samples)

    def random_uniform(self, n_samples=1, device='cpu'):
        """

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
        """
        if n_samples == 1:
            size = (self.n, self.n)
        else:
            size = (n_samples, self.n, self.n)
        random_mat = torch.randn(size, device=device)
        Q, R = gs.linalg.qr(random_mat)
        eye = gs.zeros((n_samples, self.n, self.n)).to(device)
        R = set_diag(eye, gs.sign(gs.diagonal(R, axis1=-2, axis2=-1)))

        y = Q @ R
        det = gs.linalg.det(y)
        return utils.flip_determinant(y, det)

    def skew_matrix_from_vector(self, vec):
        """Get the skew-symmetric matrix derived from the vector.

        In nD, fill a skew-symmetric matrix with the values of the vector.

        Parameters
        ----------
        vec : array-like, shape=[..., dim]
            Vector.

        Returns
        -------
        skew_mat : array-like, shape=[..., n, n]
            Skew-symmetric matrix.
        """
        return self.lie_algebra.matrix_representation(vec)

    def vector_from_skew_matrix(self, skew_mat):
        """Derive a vector from the skew-symmetric matrix.

        In 3D, compute the vector defining the cross product
        associated to the skew-symmetric matrix skew mat.

        Parameters
        ----------
        skew_mat : array-like, shape=[..., n, n]
            Skew-symmetric matrix.

        Returns
        -------
        vec : array-like, shape=[..., dim]
            Vector.
        """
        return self.lie_algebra.basis_representation(skew_mat)

    def exp_from_identity(self, tangent_vec):
        """Compute the group exponential of the tangent vector at the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dimension]
            Tangent vector at base point.

        Returns
        -------
        point : array-like, shape=[..., dimension]
            Point.
        """
        # return gs.linalg.expm(tangent_vec)
        return torch.matrix_exp(tangent_vec)

    def log_from_identity(self, point):
        """Compute the group logarithm of the point at the identity.

        Parameters
        ----------
        point : array-like, shape=[..., dimension]
            Point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dimension]
            Group logarithm.
        """
        # return gs.linalg.logm(point)
        return self.super().log_from_identity(self, point)

    @property
    def log_volume(self):
        if self.n == 2:
            return math.log(2) + math.log(math.pi)
        elif self.n == 3:
            return math.log(8) + 2 * math.log(math.pi)
        else:
            out = (self.n - 1) * math.log(2)
            out += ((self.n - 1) * (self.n + 2) / 4) * math.log(math.pi)
            k = gs.expand_dims(gs.arange(2, self.n + 1), axis=-1)
            out += gs.sum(gs.gammaln(k / 2), axis=0)
            return out

    def logdetexp(self, x, y):
        x_norm = torch.linalg.norm(x, dim=(-1,-2))
        mask = x_norm > 1e-10
        x_norm = gs.where(mask, x_norm, gs.ones_like(x_norm))
        out = math.log(2) + gs.log(1. -gs.cos(x_norm)) - 2 * gs.log(x_norm)
        return out

    @property
    def injectivity_radius(self):
        return math.pi

    #NOTE: modified to incorporate flattened data
    def random_normal_tangent(self, base_point, n_samples=1):
        """Sample in the tangent space from the standard normal distribution.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        n_samples : int
            Number of samples.
            Optional, default: 1.

        NOTE: the input vector and base_point may be flattened

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        shape = base_point.shape
        base_point = base_point.reshape(-1, self.n, self.n)
        ambiant_noise = torch.randn((n_samples, self.dim), device=base_point.device)
        samples = self.lie_algebra.matrix_representation(ambiant_noise, normed=True)
        samples = self.compose(base_point, samples)
        return samples.reshape(shape)

    #NOTE: modified to incorporate flattened data
    def to_tangent(self, vector, base_point=None):
        """Project a vector onto the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., {dim, [n, n]}]
            Vector to project. Its shape must match the shape of base_point.
        base_point : array-like, shape=[..., {dim, [n, n]}], optional
            Point of the group.
            Optional, default: identity.
        
        NOTE: the input vector and base_point may be flattened

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        """
        if base_point is None:
            return self.lie_algebra.projection(vector)
        #NOTE: Reshape to [..., n, n]
        shape = vector.shape
        vector = vector.reshape(-1, self.n, self.n)
        base_point = base_point.reshape(-1, self.n, self.n)
        tangent_vec_at_id = self.compose(self.inverse(base_point), vector)
        regularized = self.lie_algebra.projection(tangent_vec_at_id)
        return self.compose(base_point, regularized).reshape(shape)

    #NOTE: modified to incorporate flattened data
    def exp_not_from_identity(self, tangent_vec, base_point):
        """Calculate the group exponential at base_point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at base point.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Base point.

        NOTE: the input vector and base_point may be flattened

        Returns
        -------
        exp : array-like, shape=[..., {dim, [n, n]}]
            Group exponential.
        """
        shape = tangent_vec.shape
        tangent_vec = tangent_vec.reshape(-1, self.n, self.n)
        base_point = base_point.reshape(-1, self.n, self.n)

        if self.default_point_type == "vector":
            tangent_translation = self.tangent_translation_map(
                point=base_point, left_or_right="left", inverse=True
            )

            tangent_vec_at_id = tangent_translation(tangent_vec)
            exp_from_identity = self.exp_from_identity(tangent_vec=tangent_vec_at_id)
            exp = self.compose(base_point, exp_from_identity)
            exp = self.regularize(exp)
            return exp
            
        lie_vec = self.compose(self.inverse(base_point), tangent_vec)
        # NOTE: wo this Euler-Maruyama diverges outside of the manifold
        lie_vec = self.to_tangent(lie_vec)
        return self.compose(base_point, self.exp_from_identity(lie_vec)).reshape(shape)

    #NOTE: modified to incorporate flattened data
    def log_not_from_identity(self, point, base_point):
        """Compute the group logarithm of `point` from `base_point`.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]
            Point.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Base point.

        NOTE: the input vector and base_point may be flattened

        Returns
        -------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
            Group logarithm.
        """
        shape = point.shape
        point = point.reshape(-1, self.n, self.n)
        base_point = base_point.reshape(-1, self.n, self.n)

        if self.default_point_type == "vector":
            tangent_translation = self.tangent_translation_map(
                point=base_point, left_or_right="left"
            )
            point_near_id = self.compose(self.inverse(base_point), point)
            log_from_id = self.log_from_identity(point=point_near_id)
            log = tangent_translation(log_from_id)
            return log

        lie_point = self.compose(self.inverse(base_point), point)
        return self.compose(base_point, self.log_from_identity(lie_point)).reshape(shape)


class _SpecialOrthogonalVectors(LieGroup):
    """

    Parameters
    ----------
    epsilon : float
        Precision to use for calculations involving potential divison by 0 in
        rotations.
        Optional, default: 0.
    """

    def __init__(self, n, epsilon=0.0):
        dim = n * (n - 1) // 2
        LieGroup.__init__(self, dim=dim, default_point_type="vector")

        self.n = n
        self.epsilon = epsilon

    def get_identity(self, point_type="vector"):
        """Get the identity of the group.

        Parameters
        ----------
        point_type : str, {'vector', 'matrix'}
            Point_type of the returned value. Unused here.

        Returns
        -------
        identity : array-like, shape=[1,]
            Identity.
        """
        return gs.zeros(self.dim)

    identity = property(get_identity)

    @geomstats.vectorization.decorator(["else", "matrix"])
    def projection(self, point):
        """

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        rot_mat : array-like, shape=[..., n, n]
            Rotation matrix.
        """
        mat = point
        n_mats, _, _ = mat.shape

        mat_unitary_u, _, mat_unitary_v = gs.linalg.svd(mat)
        rot_mat = Matrices.mul(mat_unitary_u, mat_unitary_v)
        mask = gs.less(gs.linalg.det(rot_mat), 0.0)
        mask_float = gs.cast(mask, gs.float32) + self.epsilon
        diag = gs.concatenate((gs.ones(self.n - 1), -gs.ones(1)), axis=0)
        diag = gs.to_ndarray(diag, to_ndim=2)
        diag = (
            gs.to_ndarray(utils.from_vector_to_diagonal_matrix(diag), to_ndim=3)
            + self.epsilon
        )
        new_mat_diag_s = gs.tile(diag, [n_mats, 1, 1])

        aux_mat = Matrices.mul(mat_unitary_u, new_mat_diag_s)
        rot_mat = rot_mat + gs.einsum(
            "...,...jk->...jk", mask_float, Matrices.mul(aux_mat, mat_unitary_v)
        )
        return rot_mat

    def inverse(self, point):
        """

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point.

        Returns
        -------
        inv_point : array-like, shape=[..., 3]
            Inverse.
        """
        return -self.regularize(point)

    def random_point(self, n_samples=1, bound=1.0):
        return gs.random.rand(n_samples, 3)

    def exp_from_identity(self, tangent_vec):
        """Compute the group exponential of the tangent vector at the identity.

        As rotations are represented by their rotation vector,
        which corresponds to the element `X` in the Lie Algebra such that
        `exp(X) = R`, this methods returns its input without change.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dimension]
            Tangent vector at base point.

        Returns
        -------
        point : array-like, shape=[..., dimension]
            Point.
        """
        return self.regularize(tangent_vec)

    def log_from_identity(self, point):
        """Compute the group logarithm of the point at the identity.

        As rotations are represented by their rotation vector,
        which corresponds to the element `X` in the Lie Algebra such that
        `exp(X) = R`, this methods returns its input after regularization.


        Parameters
        ----------
        point : array-like, shape=[..., dimension]
            Point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dimension]
            Group logarithm.
        """
        return self.regularize(point)

    def skew_matrix_from_vector(self, vec):
        """Get the skew-symmetric matrix derived from the vector.

        In 3D, compute the skew-symmetric matrix,known as the cross-product of
        a vector, associated to the vector `vec`.

        Parameters
        ----------
        vec : array-like, shape=[..., dim]
            Vector.

        Returns
        -------
        skew_mat : array-like, shape=[..., n, n]
            Skew-symmetric matrix.
        """
        return SkewSymmetricMatrices(self.n).matrix_representation(vec)

    def vector_from_skew_matrix(self, skew_mat):
        """Derive a vector from the skew-symmetric matrix.

        In 3D, compute the vector defining the cross product
        associated to the skew-symmetric matrix skew mat.

        Parameters
        ----------
        skew_mat : array-like, shape=[..., n, n]
            Skew-symmetric matrix.

        Returns
        -------
        vec : array-like, shape=[..., dim]
            Vector.
        """
        return SkewSymmetricMatrices(self.n).basis_representation(skew_mat)

    def to_tangent(self, vector, base_point=None):
        return self.regularize_tangent_vec(vector, base_point)

    def regularize_tangent_vec_at_identity(self, tangent_vec, metric=None):
        """Regularize a tangent vector at the identity.

        In 2D, regularize a tangent_vector by getting its norm at the identity,
        to be less than pi.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        metric : RiemannianMetric
            Metric to compute the norm of the tangent vector.
            Optional, default is the Euclidean metric.

        Returns
        -------
        regularized_vec : array-like, shape=[..., 1]
            Regularized tangent vector.
        """
        return self.regularize(tangent_vec)

    def regularize_tangent_vec(self, tangent_vec, base_point, metric=None):
        """Regularize tangent vector at a base point.

        In 2D, regularize a tangent_vector by getting the norm of its parallel
        transport to the identity, determined by the metric, less than pi.

        Parameters
        ----------
        tangent_vec : array-like, shape=[...,1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Point on the manifold.
        metric : RiemannianMetric
            Metric to compute the norm of the tangent vector.
            Optional, default is the Euclidean metric.

        Returns
        -------
        regularized_tangent_vec : array-like, shape=[..., 1]
            Regularized tangent vector.
        """
        return self.regularize_tangent_vec_at_identity(tangent_vec)

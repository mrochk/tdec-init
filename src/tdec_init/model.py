import equinox as eqx
import jax, jax.numpy as jnp
from jax.lax import stop_gradient
from decoupling._splines import design_matrices

from jaxtyping import Array
from typing import Callable

from tdec_init.utils import wrap_angles

class Internals(eqx.Module):
    '''b-spline internals extended linearly outside of their knots'''

    coefs: Array
    knots: Array
    degree: int

    def __call__(self, us):

        def internal(u, c, k):
            lo, hi = k[0], k[-1]
            u_clipped = jnp.clip(u, lo, hi)[None]
            B, dB = design_matrices(u_clipped, k, self.degree)
            value, slope = B @ c, dB @ c
            return (value + slope * (u - u_clipped)).squeeze()

        return jax.vmap(internal)(us, *stop_gradient((self.coefs, self.knots)))

class MLP(eqx.Module):
    W0: Array
    W1: Array
    b0: Array
    b1: Array
    activations: Callable 

    scale: Array
    angles_mask: Array # mask for the inputs that are angles

    def __init__(self, W0, W1, activations, scale, angles_mask):
        self.W0, self.W1, self.activations = W0, W1, activations
        self.scale, self.angles_mask = scale, angles_mask
        self.b0, self.b1 = jnp.zeros(W0.shape[0]), jnp.zeros(W1.shape[0])

    def _normalized(self, x):
        x_wrapped = wrap_angles(x, self.angles_mask)
        return self.W1 @ self.activations(self.W0 @ x_wrapped + self.b0) + self.b1

    def __call__(self, x): return self._normalized(x) * stop_gradient(self.scale)

def loss(model: MLP, inputs, outputs):
    scale = jax.lax.stop_gradient(model.scale)
    predictions = jax.vmap(model)(inputs)
    errors = (predictions - outputs) / scale
    return jnp.mean(errors**2)

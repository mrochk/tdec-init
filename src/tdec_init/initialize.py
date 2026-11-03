import optax
import equinox as eqx
from tqdm import tqdm
import jax, jax.numpy as jnp
from jax.lax import stop_gradient
from jax.random import split, fold_in
from jaxtyping import Array

from tdec_init.decoupling import cmtf_bsd
from tdec_init.model import Internals, MLP

def values_loss(model: MLP, inputs, outputs, _):
    '''mse on the normalized targets'''
    scale = stop_gradient(model.scale)
    errors = (jax.vmap(model)(inputs) - outputs) / scale
    return jnp.mean(errors**2)

def sobolev_loss(model: MLP, inputs, outputs, jacobians):
    '''mse on the normalized targets and on their jacobians, with equal weights'''
    scale = stop_gradient(model.scale)
    errors = (jax.vmap(model)(inputs) - outputs) / scale
    slope_errors = (jax.vmap(jax.jacfwd(model))(inputs) - jacobians) / scale[:, None]
    return jnp.mean(errors**2) + jnp.mean(slope_errors**2)

class Initializer:
    name: str
    trained_steps = 0 # gradient steps of the initialization, set by the pretraining

    def __init__(self, key: Array, num_hidden: int, inputs: Array, outputs: Array, jacobians: Array, angles_mask: Array):
        self.key = key
        self.num_hidden = num_hidden
        self.inputs, self.outputs, self.jacobians = inputs, outputs, jacobians
        self.angles_mask = angles_mask

    def initialize(self) -> MLP: raise NotImplementedError

class RandomInitializer(Initializer):
    '''lecun normal on standardized inputs, zero biases'''
    name = 'random'

    def initialize(self):
        num_inputs, num_outputs = self.inputs.shape[1], self.outputs.shape[1]
        key_W0, key_W1 = split(self.key)

        in_scale = jnp.std(self.inputs, axis=0) + 1e-8
        out_scale = jnp.std(self.outputs, axis=0) + 1e-8

        W0 = jax.random.normal(key_W0, (self.num_hidden, num_inputs)) / (jnp.sqrt(num_inputs) * in_scale)
        W1 = jax.random.normal(key_W1, (num_outputs, self.num_hidden)) / jnp.sqrt(self.num_hidden)

        return MLP(W0, W1, jax.nn.gelu, out_scale, self.angles_mask)

class DecouplingInitializer(Initializer):
    '''weights and frozen spline internals from the decoupling of the jacobians'''
    name = 'decoupling'

    def initialize(self):
        scale = jnp.std(self.outputs, axis=0) + 1e-8

        decoupling = cmtf_bsd(self.inputs, self.outputs, self.jacobians, self.num_hidden, self.key)
        internals = Internals(decoupling.coefs, decoupling.knots, decoupling.degree)

        # the mlp multiplies its output by the scale, so that it computes W g(V^T x)
        W0, W1 = decoupling.V.T, decoupling.W / scale[:, None]

        return MLP(W0, W1, internals, scale, self.angles_mask)

class ValuesInitializer(Initializer):
    '''random initialization, then pretraining on the values'''
    name = 'values'
    loss = staticmethod(values_loss)

    def __init__(self, key, num_hidden, inputs, outputs, jacobians, angles_mask, pretrain_steps: int, lr: float = 1e-3, batch_size: int = 32):
        super().__init__(key, num_hidden, inputs, outputs, jacobians, angles_mask)
        self.pretrain_steps = pretrain_steps
        self.lr = lr
        self.batch_size = batch_size

        # the number of steps tells apart the runs with different budgets
        self.name = f'{self.name}-{pretrain_steps}'

    def _random_model(self) -> MLP:
        return RandomInitializer(self.key, self.num_hidden, self.inputs, self.outputs, self.jacobians, self.angles_mask).initialize()

    def initialize(self):
        return self.pretrain(self._random_model(), (self.inputs, self.outputs, self.jacobians))

    def pretrain(self, model: MLP, data: tuple, validation: tuple | None = None, patience: int | None = None) -> MLP:
        '''
        minimize the loss on data = (inputs, outputs, jacobians) for `pretrain_steps` steps
        with a validation set, return the model with the lowest validation loss, checked every 100 steps,
        and stop once it has not improved for `patience` steps
        '''
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        inputs, outputs, jacobians = data
        jacobians = jacobians.transpose(2, 0, 1) # (N, n, m)

        @eqx.filter_jit
        def update(model, opt_state, key):
            index = jax.random.choice(key, inputs.shape[0], (self.batch_size,)) # with replacement
            loss, grads = eqx.filter_value_and_grad(self.loss)(model, inputs[index], outputs[index], jacobians[index])
            updates, opt_state = optimizer.update(grads, opt_state)
            return eqx.apply_updates(model, updates), opt_state, loss

        @eqx.filter_jit
        def validate(model):
            val_inputs, val_outputs, val_jacobians = validation
            return self.loss(model, val_inputs, val_outputs, val_jacobians.transpose(2, 0, 1))

        best, best_loss, best_step = model, validate(model) if validation is not None else None, 0

        # self.key already drew the random weights, derive another one for the batches
        keys = split(fold_in(self.key, 1), self.pretrain_steps)
        bar, losses = tqdm(keys, desc=f'{self.name} pretraining'), []

        for step, key in enumerate(bar, 1):
            model, opt_state, loss = update(model, opt_state, key)
            losses.append(loss)

            # the batch loss is noisy, show its mean over the last 100 steps
            if step % 100 == 0:
                postfix = dict(loss=f'{float(jnp.mean(jnp.array(losses[-100:]))):.6f}')

                if validation is not None:
                    validation_loss = validate(model)
                    if validation_loss < best_loss: best, best_loss, best_step = model, validation_loss, step
                    postfix.update(validation=f'{float(validation_loss):.6f}', best=f'{float(best_loss):.6f}')

                bar.set_postfix(postfix)

                if validation is not None and patience is not None and step - best_step >= patience: break

        bar.close()
        self.trained_steps = step
        return model if validation is None else best

class SobolevInitializer(ValuesInitializer):
    '''random initialization, then pretraining on the values and the jacobians'''
    name = 'sobolev'
    loss = staticmethod(sobolev_loss)

class ValuesHeldoutInitializer(ValuesInitializer):
    '''pretrain on a part of the samples until the loss of the held out rest stops improving'''
    name = 'values-heldout'
    validation = 0.2 # fraction of the samples held out, at least one

    def __init__(self, key, num_hidden, inputs, outputs, jacobians, angles_mask, patience: int = 2000, max_steps: int = 100_000, **kwargs):
        super().__init__(key, num_hidden, inputs, outputs, jacobians, angles_mask, max_steps, **kwargs)
        self.patience = patience
        # trained until convergence, the name has no number of steps
        self.name = type(self).name

    def initialize(self):
        N = self.inputs.shape[0]
        index = jax.random.permutation(fold_in(self.key, 2), N)
        k = max(1, round(self.validation * N))
        subset = lambda i: (self.inputs[i], self.outputs[i], self.jacobians[..., i])

        return self.pretrain(self._random_model(), subset(index[k:]), subset(index[:k]), self.patience)

class SobolevHeldoutInitializer(ValuesHeldoutInitializer):
    name = 'sobolev-heldout'
    loss = staticmethod(sobolev_loss)

INITIALIZERS = [
    RandomInitializer,
    DecouplingInitializer,
    ValuesInitializer,
    SobolevInitializer,
    ValuesHeldoutInitializer,
    SobolevHeldoutInitializer,
]

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

class Random_Init(Initializer):
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

class TDec_Init(Initializer):
    '''weights and frozen spline internals from the decoupling of the jacobians'''
    name = 'decoupling'

    def initialize(self):
        scale = jnp.std(self.outputs, axis=0) + 1e-8

        decoupling = cmtf_bsd(self.inputs, self.outputs, self.jacobians, self.num_hidden, self.key)
        internals = Internals(decoupling.coefs, decoupling.knots, decoupling.degree)

        # the mlp multiplies its output by the scale, so that it computes W g(V^T x)
        W0, W1 = decoupling.V.T, decoupling.W / scale[:, None]

        return MLP(W0, W1, internals, scale, self.angles_mask)

class Values_Init(Initializer):
    name = 'values-heldout'
    loss = staticmethod(values_loss)
    validation = 0.2 # fraction of the samples held out, at least one

    def __init__(
        self, key, num_hidden, inputs, outputs, jacobians, angles_mask,
        patience: int = 2000, max_steps: int = 100_000, lr: float = 1e-3, batch_size: int = 32, eval_every: int = 25,
    ):
        super().__init__(key, num_hidden, inputs, outputs, jacobians, angles_mask)
        self.patience, self.max_steps = patience, max_steps
        self.lr, self.batch_size = lr, batch_size
        self.eval_every = eval_every # steps between the checks of the held-out loss, as between the episodes

    def initialize(self):
        model = Random_Init(self.key, self.num_hidden, self.inputs, self.outputs, self.jacobians, self.angles_mask).initialize()

        N = self.inputs.shape[0]
        index = jax.random.permutation(fold_in(self.key, 2), N)
        k = max(1, round(self.validation * N))
        subset = lambda i: (self.inputs[i], self.outputs[i], self.jacobians[..., i].transpose(2, 0, 1)) # jacobians (N, n, m)

        return self.pretrain(model, subset(index[k:]), subset(index[:k]))

    def pretrain(self, model: MLP, data: tuple, validation: tuple) -> MLP:
        '''minimize the loss on data = (inputs, outputs, jacobians), early stopped on the loss on validation'''
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        inputs, outputs, jacobians = data

        @eqx.filter_jit
        def update(model, opt_state, key):
            index = jax.random.choice(key, inputs.shape[0], (self.batch_size,)) # with replacement
            loss, grads = eqx.filter_value_and_grad(self.loss)(model, inputs[index], outputs[index], jacobians[index])
            updates, opt_state = optimizer.update(grads, opt_state)
            return eqx.apply_updates(model, updates), opt_state, loss

        validate = eqx.filter_jit(lambda model: self.loss(model, *validation))

        # step 0 is the random model
        best, best_loss, best_step = model, validate(model), 0
        validation_loss = best_loss # the last checked, shown in the progress bar

        # self.key already drew the random weights, derive another one for the batches
        keys = split(fold_in(self.key, 1), self.max_steps)
        bar, losses = tqdm(keys, desc=f'{self.name} pretraining'), []

        for step, key in enumerate(bar, 1):
            model, opt_state, loss = update(model, opt_state, key)
            losses.append(loss)

            if step % self.eval_every == 0:
                validation_loss = validate(model)
                if validation_loss < best_loss: best, best_loss, best_step = model, validation_loss, step
                if step - best_step >= self.patience: break

            # the batch loss is noisy, show its mean over the last 100 steps
            if step % 100 == 0:
                bar.set_postfix(
                    loss=f'{float(jnp.mean(jnp.array(losses[-100:]))):.6f}',
                    validation=f'{float(validation_loss):.6f}',
                    best=f'{float(best_loss):.6f}',
                )

        bar.close()
        self.trained_steps = best_step # the steps of the kept model, as recorded between the episodes
        return best

class Sobolev_Init(Values_Init):
    '''as ValuesHeldoutInitializer, pretraining on the values and the jacobians'''
    name = 'sobolev-heldout'
    loss = staticmethod(sobolev_loss)

import optax
import jax, jax.numpy as jnp
import equinox as eqx
from tqdm import tqdm

from tdec_init.model import MLP
from tdec_init.planner import RandomShootingPlanner
from tdec_init.initialize import Initializer
from tdec_init.collect import make_delta
from tdec_init.utils import *

def unstack(states) -> list:
    '''split stacked states into the list of states expected by gymnax.visualize.Visualizer'''
    return [jax.tree.map(lambda x: x[i], states) for i in range(len(states.time))]

class ModelBasedAlgorithm:

    def __init__(
        self,
        env, key,
        initializer: Initializer,
        planner: RandomShootingPlanner,
        params = None,
        fields = None,
        lr: float = 1e-3,
        batch_size: int = 32,
        train_steps: int = 1000,
    ):
        self.env = env
        self.key = key

        self.planner = planner

        self.params = env.default_params if params is None else params
        self.fields = get_fields(env, fields)

        self.planner.post_init(env, self.params, self.fields)

        self.batch_size = batch_size
        self.train_steps = train_steps

        # the episodes have their own keys, so that they start from the same states whatever the training
        self.episode_key = jax.random.fold_in(key, 1)

        self.model = initializer.initialize()

        self.inputs, self.outputs = initializer.inputs, initializer.outputs

        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_inexact_array))

    @staticmethod
    def _loss(model: MLP, inputs, outputs):
        scale = jax.lax.stop_gradient(model.scale)
        predictions = jax.vmap(model)(inputs)
        errors = (predictions - outputs) / scale
        return jnp.mean(errors**2)

    @eqx.filter_jit
    def _update_model(self, model, opt_state, inputs, outputs):
        grads = eqx.filter_grad(self._loss)(model, inputs, outputs)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state

    def _train_model(self):
        for _ in range(self.train_steps):
            self.key, key = jax.random.split(self.key)

            index = jax.random.choice(key, self.inputs.shape[0], (self.batch_size,))
            inputs = self.inputs[index]
            outputs = self.outputs[index]

            self.model, self.opt_state = self._update_model(self.model, self.opt_state, inputs, outputs)

    @eqx.filter_jit
    def episode(self, model: MLP, key):
        '''run one episode in the real environment, planning with the model'''

        def step(carry, key):
            state, done = carry
            key_plan, key_step = jax.random.split(key)
            action = self.planner.plan(model, state, key_plan)
            _, next_state, reward, terminated, _ = self.env.step_env(key_step, state, action, self.params)
            s = pack(state, self.fields)
            x = jnp.concatenate([s, jnp.ravel(action)])
            y = pack(next_state, self.fields) - s
            return (next_state, done | terminated), (x, y, reward, ~done, state)

        key_reset, key_steps = jax.random.split(key)
        _, state = self.env.reset(key_reset, self.params)
        keys = jax.random.split(key_steps, self.params.max_steps_in_episode)
        _, out = jax.lax.scan(step, (state, jnp.array(False)), keys)
        return out

    def run(self, num_episodes: int):

        returns, losses = [], []

        bar = tqdm(range(num_episodes))
        for episode in bar:

            bar.set_description_str(f'(episode)')
            X, Y, rewards, valid, _ = self.episode(self.model, jax.random.fold_in(self.episode_key, episode))
            X, Y = X[valid], Y[valid]

            total_rewards = float(jnp.sum(rewards * valid)) 
            bar.set_postfix_str(f'episode {episode}, rewards {total_rewards:.2f}')

            returns.append(total_rewards)
            losses.append(float(self._loss(self.model, X, Y)))

            self.inputs = jnp.concatenate([self.inputs, X])
            self.outputs = jnp.concatenate([self.outputs, Y])

            bar.set_description_str(f'(training)')
            self._train_model()

        return returns, losses

    def simulate(self, key):
        _, _, rewards, valid, states = self.episode(self.model, key)
        num_steps = int(valid.sum())
        return rewards[:num_steps], unstack(states)[:num_steps]

class Oracle(ModelBasedAlgorithm):
    '''no model to train'''

    def run(self, num_episodes: int):
        returns = []
        for episode in range(num_episodes):
            _, _, rewards, valid, _ = self.episode(self.model, jax.random.fold_in(self.episode_key, episode))
            returns.append(float((rewards * valid).sum()))
        return returns, [float('nan')] * num_episodes

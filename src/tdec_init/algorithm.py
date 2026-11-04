import optax
import equinox as eqx
from tqdm import tqdm
import jax, jax.numpy as jnp

from tdec_init.planner import RandomShootingPlanner
from tdec_init.initialize import Initializer
from tdec_init.model import MLP, loss
from tdec_init.utils import *

def unstack(states) -> list:
    '''split stacked states into the list of states expected by visualizer'''
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
        train_steps: int = 10_000,
        patience: int | None = None,
        validation: float = 0.2,
        eval_every: int = 50,
    ):
        '''
        with `patience`, a `validation` fraction of each episode's data is held out once and for all when it is
        collected, each training round trains on the rest for at most `train_steps` steps, checks the loss on all
        the held-out data every `eval_every` steps, stops once it has not improved for `patience` steps, and keeps
        the model with the lowest held-out loss, the initial samples are never held out, the initializer saw them
        without, each round is `train_steps` steps on all the data
        '''
        self.env = env
        self.key = key

        self.planner = planner

        self.params = env.default_params if params is None else params
        self.fields = get_fields(env, fields)

        self.planner.post_init(env, self.params, self.fields)

        self.batch_size = batch_size
        self.train_steps = train_steps
        self.patience, self.validation, self.eval_every = patience, validation, eval_every
        self.trained_steps = [] # steps of each training round, with early stopping the step of the kept model

        # the episodes have their own keys, so that they start from the same states whatever the training
        self.episode_key = jax.random.fold_in(key, 1)

        self.model = initializer.initialize()

        self.inputs, self.outputs = initializer.inputs, initializer.outputs
        self.held_out = jnp.zeros(self.inputs.shape[0], dtype=bool) # the samples never trained on, for early stopping

        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_inexact_array))

    @eqx.filter_jit
    def _update_model(self, model, opt_state, inputs, outputs):
        L, grads = eqx.filter_value_and_grad(loss)(model, inputs, outputs)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        return L, eqx.apply_updates(model, updates), opt_state

    @staticmethod
    @eqx.filter_jit
    def _validation_loss(model, inputs, outputs): return loss(model, inputs, outputs)

    def _hold_out(self, num: int):
        '''mask of the new samples to hold out, a `validation` fraction drawn at random'''
        self.key, key = jax.random.split(self.key)
        k = round(self.validation * num)
        return jnp.zeros(num, dtype=bool).at[jax.random.permutation(key, num)[:k]].set(True)

    def _train_model(self):
        # the held-out samples were never trained on, so the model before the round competes fairly
        early_stopping = self.patience is not None and bool(self.held_out.any())

        if early_stopping:
            val_inputs, val_outputs = self.inputs[self.held_out], self.outputs[self.held_out]
            inputs, outputs = self.inputs[~self.held_out], self.outputs[~self.held_out]
            # step 0 is the model before the round, kept if the training only makes it worse
            best, best_loss, best_step = self.model, self._validation_loss(self.model, val_inputs, val_outputs), 0
        else: inputs, outputs = self.inputs, self.outputs

        bar = tqdm(range(1, self.train_steps + 1))
        L, step = None, 0
        for step in bar:
            self.key, key = jax.random.split(self.key)
            index = jax.random.choice(key, inputs.shape[0], (self.batch_size,))
            L_prev = L
            L, self.model, self.opt_state = self._update_model(self.model, self.opt_state, inputs[index], outputs[index])
            if L_prev is None: L_prev = L
            L = 0.5*L_prev + 0.5*L
            bar.set_postfix_str(f'loss={L:.4f}')

            if early_stopping and step % self.eval_every == 0:
                validation_loss = self._validation_loss(self.model, val_inputs, val_outputs)
                if validation_loss < best_loss: best, best_loss, best_step = self.model, validation_loss, step
                bar.set_postfix_str(f'loss={L:.6f}, validation={float(validation_loss):.6f}, best={float(best_loss):.6f}')
                if step - best_step >= self.patience: break

        bar.close()
        if early_stopping: self.model, step = best, best_step
        self.trained_steps.append(step)

    @eqx.filter_jit
    def episode(self, model: MLP, key):
        '''run one episode in the real environment, planning with the model'''

        def step(carry, key):
            state, done = carry
            key_plan, key_step = jax.random.split(key)
            action = self.planner.plan(model, state, key_plan)
            _, next_state, reward, terminated, _ = self.env.step_env(key_step, state, action, self.params)
            s = state2vec(state, self.fields)
            x = jnp.concatenate([s, jnp.ravel(action)])
            y = state2vec(next_state, self.fields) - s
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
            losses.append(float(loss(self.model, X, Y)))

            self.inputs = jnp.concatenate([self.inputs, X])
            self.outputs = jnp.concatenate([self.outputs, Y])
            if self.patience is not None: self.held_out = jnp.concatenate([self.held_out, self._hold_out(X.shape[0])])
            else: self.held_out = jnp.concatenate([self.held_out, jnp.zeros(X.shape[0], dtype=bool)])

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

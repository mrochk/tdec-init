from abc import ABC, abstractmethod
import jax, jax.numpy as jnp

from tdec_init.model import MLP
from tdec_init.collect import make_delta
from tdec_init.utils import pack, unpack

class Planner(ABC):
    def __init__(self, candidates, horizon):
        self.candidates = candidates
        self.horizon = horizon

    def post_init(self, env, params, fields):
        self.env = env
        self.params = params
        self.fields = fields

    def _reward(self, state, s, action):
        '''known reward and termination'''
        key = jax.random.key(0)
        _, _, reward, done, _ = self.env.step_env(key, unpack(state, self.fields, s), action, self.params)
        return reward, done

    @abstractmethod
    def plan(self, model, state, key): raise NotImplementedError

class RandomShootingPlanner(Planner):
    def plan(self, model: MLP, state, key):
        '''return the first action of the best random action sequence'''

        action_space = self.env.action_space(self.params)

        keys = jax.random.split(key, (self.candidates, self.horizon))
        actions = jax.vmap(jax.vmap(action_space.sample))(keys)

        def rollout(actions):

            def step(carry, action):
                s, done = carry
                reward, terminal = self._reward(state, s, action)
                # no reward after the episode ends, as in the real return
                reward = jnp.where(done, 0.0, reward)
                return (s + model(jnp.concatenate([s, jnp.ravel(action)])), done | terminal), reward

            _, rewards = jax.lax.scan(step, (pack(state, self.fields), jnp.array(False)), actions)

            return jnp.sum(rewards)

        returns = jax.vmap(rollout)(actions)

        # return the action that led to the highest return
        return actions[jnp.argmax(returns), 0]

class OraclePlanner(RandomShootingPlanner):
    '''plans with the true dynamics instead of the world model'''

    def plan(self, model, state, key):
        f = make_delta(self.env, self.params, self.fields)
        return super().plan(lambda x: f(state, x, jax.random.key(0)), state, key)

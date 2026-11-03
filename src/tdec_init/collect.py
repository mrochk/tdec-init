import jax, jax.numpy as jnp
from jax.random import split
from jaxtyping import Array

from tdec_init.utils import *

def make_delta(env, params, fields):
    env = env.with_transition_gradients()
    action_shape = env.action_space(params).shape

    def delta(state, x, key):
        state_size = pack(state, fields).size
        state_vec, action_vec = x[:state_size], x[state_size:].reshape(action_shape)
        _, next_state, *_ = env.step_env(key, unpack(state, fields, state_vec), action_vec, params)
        return pack(next_state, fields) - state_vec

    return delta

def collect_information(
    env,
    key: Array,
    num_envs: int,
    num_steps: int,
    fields = None,
    params = None,
):
    if params is None: params = env.default_params

    fields = get_fields(env, fields)
    action_space = env.action_space(params)
    delta = make_delta(env, params, fields)

    print(f'collecting {num_envs}x{num_steps} samples...', end=' ', flush=True)

    def step(state, key):
        key_action, key_step = split(key, 2)

        action = action_space.sample(key_action) # sample uniformly

        x = jnp.concatenate([pack(state, fields), jnp.ravel(action)])

        # env.step forwards the first half of its key to step_env
        key_env = split(key_step)[0]

        f = lambda x: delta(state, x, key_env)
        jacobian = jax.jacfwd(f)

        _, state, *_ = env.step(key_step, state, action, params)

        return state, (x, f(x), jacobian(x))

    def rollout(key):
        key_reset, key_steps = split(key, 2)
        keys_states = split(key_steps, num_steps)

        _, state = env.reset(key_reset, params)
        _, out = jax.lax.scan(step, state, keys_states)

        return out

    keys_envs = split(key, num_envs)
    inputs, outputs, jacobians = jax.vmap(rollout)(keys_envs)

    print('done', flush=True)

    N, (n, m) = num_envs * num_steps, jacobians.shape[-2:]
    inputs = wrap_angles(inputs.reshape(N, m), angle_mask(env, fields, params))

    return (inputs, outputs.reshape(N, n), jacobians.reshape(N, n, m).transpose(1, 2, 0))

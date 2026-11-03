import jax, jax.numpy as jnp
from jaxtyping import Array

from tdec_init.envs_info import STATE_FIELDS, STATE_BOUNDS, ANGLE_FIELDS

def pack(state, fields) -> Array:
    '''flatten the state obj into a vector'''
    return jnp.concatenate([jnp.ravel(getattr(state, f)) for f in fields])

def unpack(state, fields, vector):
    '''convert vector back to state obj'''
    updates, i = {}, 0
    for f in fields:
        leaf = jnp.asarray(getattr(state, f))
        updates[f] = vector[i:i+leaf.size].reshape(leaf.shape)
        i += leaf.size
    return state.replace(**updates)

def get_fields(env, fields = None):
    if fields is not None: return fields
    if env.name not in STATE_FIELDS: raise ValueError(f'no default state fields for {env.name}')
    return STATE_FIELDS[env.name]

def wrap_angles(x, mask):
    '''wrap angles of x to [-pi, pi)'''
    return jnp.where(mask, (x + jnp.pi) % (2 * jnp.pi) - jnp.pi, x)

def angle_mask(env, fields=None, params=None) -> jax.Array:
    '''boolean mask for the entries that are angles'''
    if params is None: params = env.default_params
    _, state = env.reset(jax.random.key(0), params)
    angles = ANGLE_FIELDS.get(env.name, ())
    mask = [jnp.full(jnp.size(getattr(state, f)), f in angles) for f in get_fields(env, fields)]
    num_actions = jnp.zeros(env.action_space(params).shape).size
    return jnp.concatenate(mask + [jnp.zeros(num_actions, dtype=bool)])

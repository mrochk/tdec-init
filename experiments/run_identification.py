import warnings; warnings.simplefilter('ignore')

import csv, os, click
import jax, jax.numpy as jnp
import equinox as eqx
import gymnax

from tdec_init.initialize import *
from tdec_init import collect_information
from tdec_init.utils import angle_mask, get_fields, state2vec, wrap_angles

from config import ENVS, PARAMS, SEED, clear_rows

SEEDS = range(5)
NUM_ENVS = [100]
NUM_STEPS = 5

NUM_TEST_ENVS, NUM_TEST_STEPS = 500, NUM_STEPS # one-step test set
NUM_ROLLOUTS, ROLLOUT_HORIZON = 100, 20 # multi-step test set

RESULTS = os.path.join(os.path.dirname(__file__), 'results', 'init.csv')
COLUMNS = ['env', 'seed', 'N', 'init', 'onestep', 'jacobian', 'rollout', 'pretrain_steps']

def trajectories(env, params, fields, key, num: int, horizon: int):
    action_space = env.action_space(params)

    def step(state, key):
        key_action, key_step = jax.random.split(key)
        action = action_space.sample(key_action)
        _, next_state, *_ = env.step_env(key_step, state, action, params)
        return next_state, (state2vec(state, fields), jnp.ravel(action))

    def rollout(key):
        key_reset, key_steps = jax.random.split(key)
        _, state = env.reset(key_reset, params)
        _, (S, A) = jax.lax.scan(step, state, jax.random.split(key_steps, horizon + 1))
        return S, A[:-1]

    return jax.jit(jax.vmap(rollout))(jax.random.split(key, num))

@eqx.filter_jit
def evaluate(model, test, S, A, mask):
    '''normalized errors of the one-step predictions of their jacobians and of the rollouts'''
    inputs, outputs, jacobians = test

    var = jnp.var(outputs, axis=0) + 1e-8
    onestep = jnp.mean((jax.vmap(model)(inputs) - outputs) ** 2 / var)

    J, Jhat = jacobians.transpose(2, 0, 1), jax.vmap(jax.jacfwd(model))(inputs)
    jacobian = jnp.mean((Jhat - J) ** 2 / var[:, None]) / jnp.mean(J ** 2 / var[:, None])

    def rollout(s, actions):
        step = lambda s, a: (s + model(jnp.concatenate([s, a])),) * 2
        return jax.lax.scan(step, s, actions)[1]

    n = S.shape[-1]
    Shat = jax.vmap(rollout)(S[:, 0], A)
    std = jnp.std(inputs[:, :n], axis=0) + 1e-8
    rollout = jnp.mean((wrap_angles(Shat - S[:, 1:], mask[:n]) / std) ** 2)

    return onestep, jacobian, rollout

def make_initializers(*args):
    return (Random_Init(*args), TDec_Init(*args), Values_Init(*args), Sobolev_Init(*args))

@click.command()
@click.option('--env', '-e', 'envs', multiple=True, type=click.Choice(list(ENVS), case_sensitive=False))
def main(envs):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

    for env_name in [ENVS[e] for e in envs or ENVS]:
        hp = PARAMS[env_name]
        env, env_params = gymnax.make(env_name)
        env_params = env_params.replace(**hp.get('env_params', {}))
        fields, mask = get_fields(env), angle_mask(env, params=env_params)

        for seed in SEEDS:
            clear_rows(RESULTS, COLUMNS, env_name, seed)
            key_train, key_test, key_rollouts, key_init = jax.random.split(jax.random.fold_in(jax.random.key(SEED), seed), 4)

            # the test sets are shared by all the N and initializations of a seed
            test = collect_information(env, key_test, NUM_TEST_ENVS, NUM_TEST_STEPS, params=env_params)
            S, A = trajectories(env, env_params, fields, key_rollouts, NUM_ROLLOUTS, ROLLOUT_HORIZON)

            for num_envs in NUM_ENVS:
                samples = collect_information(env, key_train, num_envs, NUM_STEPS, params=env_params)
                N = samples[0].shape[0]

                args = (key_init, hp['num_hidden'], *samples, mask)

                initializers = make_initializers(*args)

                for initializer in initializers:
                    try: model = initializer.initialize()
                    except Exception as e:
                        assert isinstance(initializer, TDec_Init)
                        print(f'{initializer.name} failed, skipping: {e}')
                        continue

                    onestep, jacobian, rollout = map(float, evaluate(model, test, S, A, mask))
                    with open(RESULTS, 'a', newline='') as f:
                        csv.writer(f).writerow([env_name, seed, N, initializer.name, onestep, jacobian, rollout, initializer.trained_steps])

                    print(f'{env_name} seed {seed} N {N:4d} {initializer.name:>16}: onestep {onestep:.2e} jacobian {jacobian:.2e} rollout {rollout:.2e}')

if __name__ == '__main__': main()

import warnings; warnings.simplefilter('ignore')
import csv, os, click
import jax, gymnax
from pprint import pp

from tdec_init import ModelBasedAlgorithm, Oracle, RandomShootingPlanner, OraclePlanner, collect_information
from tdec_init.initialize import *
from tdec_init.utils import angle_mask

from config import ENVS, PARAMS, PRETRAIN_STEPS, SEED, clear_rows

SEEDS = range(5)

RESULTS = os.path.join(os.path.dirname(__file__), 'results', 'mbrl.csv')
COLUMNS = ['env', 'seed', 'init', 'episode', 'return', 'loss']

@click.command()
@click.option('--env', '-e', 'envs', multiple=True, type=click.Choice(list(ENVS), case_sensitive=False))
def main(envs):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

    for env_name in [ENVS[e] for e in envs or ENVS]:
        hp = PARAMS[env_name]
        env, env_params = gymnax.make(env_name)
        env_params = env_params.replace(**hp.get('env_params', {}))
        mask = angle_mask(env, params=env_params)
        pp(env_name)
        pp(hp)

        for seed in SEEDS:
            clear_rows(RESULTS, COLUMNS, env_name, seed)
            key_samples, key_init, key_algorithm = jax.random.split(jax.random.fold_in(jax.random.key(SEED), seed), 3)
            samples = collect_information(env, key_samples, hp['num_envs_collect'], hp['num_steps_collect'], params=env_params)

            args = (key_init, hp['num_hidden'], *samples, mask)
            initializers = [
                RandomInitializer(*args),
                TDec_Init(*args),
                #*[ValuesInitializer(*args, steps) for steps in PRETRAIN_STEPS],
                #*[SobolevInitializer(*args, steps) for steps in PRETRAIN_STEPS],
                #ValuesHeldoutInitializer(*args),
                #SobolevHeldoutInitializer(*args),
            ]

            # same algorithm key for all so that episodes start from same states
            runs = [(i.name, ModelBasedAlgorithm, RandomShootingPlanner, i) for i in initializers]
            runs += [('oracle', Oracle, OraclePlanner, RandomInitializer(*args))]

            for name, Algorithm, Planner, initializer in runs:
                print(f'{env_name} seed {seed} {name}')

                make = lambda initializer: Algorithm(
                    env, key_algorithm, initializer,
                    Planner(hp['num_candidates'], hp['horizon']),
                    params=env_params,
                    train_steps=hp['train_steps'],
                )

                # the decoupling fails when the rank is too high for the samples, the run is skipped and has no rows
                try: algorithm = make(initializer)
                except Exception as e:
                    if not isinstance(initializer, TDec_Init): raise
                    print(f'{name} failed, skipping: {e}')
                    continue

                returns, losses = algorithm.run(hp['num_episodes'])

                with open(RESULTS, 'a', newline='') as f:
                    csv.writer(f).writerows([env_name, seed, name, episode, r, l] for episode, (r, l) in enumerate(zip(returns, losses)))

                print('  returns:', ' '.join(f'{r:8.1f}' for r in returns))

if __name__ == '__main__': main()

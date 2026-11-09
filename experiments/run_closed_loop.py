import warnings; warnings.simplefilter('ignore')
import csv, os, click
import jax, gymnax
from pprint import pp
from pathlib import Path

from tdec_init.initialize import *
from tdec_init.utils import angle_mask
from tdec_init import (
    RandomShootingPlanner,
    ModelBasedAlgorithm,
    collect_information,
    OraclePlanner,
    Oracle,
)

from config import ENVS, PARAMS, SEED, clear_rows

SEEDS = range(5)

PATH_RESULTS = Path(__file__).parent / 'results' / 'mbrl.csv'

COLUMNS = ['env', 'seed', 'init', 'episode', 'return', 'loss', 'train_steps']

@click.command()
@click.option('--env', '-e', 'envs', multiple=True, type=click.Choice(list(ENVS), case_sensitive=False))
def main(envs):
    os.makedirs(os.path.dirname(PATH_RESULTS), exist_ok=True)

    for env_name in [ENVS[e] for e in envs or ENVS]:
        hp = PARAMS[env_name]
        env, env_params = gymnax.make(env_name)
        env_params = env_params.replace(**hp.get('env_params', {}))
        mask = angle_mask(env, params=env_params)
        pp(env_name)
        pp(hp)

        for seed in SEEDS:
            key_samples, key_init, key_algorithm = jax.random.split(jax.random.fold_in(jax.random.key(SEED), seed), 3)
            samples = collect_information(env, key_samples, hp['num_envs_collect'], hp['num_steps_collect'], params=env_params)

            args = (key_init, hp['num_hidden'], *samples, mask)
            initializers = [
                Random_Init(*args),
                TDec_Init(*args),
                Values_Init(*args),
                Sobolev_Init(*args),
            ]

            # same algorithm key for all so that episodes start from same states
            runs = [('oracle', Oracle, OraclePlanner, Random_Init(*args))] + [(i.name, ModelBasedAlgorithm, RandomShootingPlanner, i) for i in initializers]

            clear_rows(PATH_RESULTS, COLUMNS, env_name, seed, [name for name, *_ in runs])

            for name, Algorithm, Planner, initializer in runs:
                print(f'{env_name} seed {seed} {name}')

                make = lambda initializer: Algorithm(
                    env, key_algorithm, initializer,
                    Planner(hp['num_candidates'], hp['horizon']),
                    params=env_params,
                )

                try: algorithm = make(initializer)
                except Exception as e:
                    assert isinstance(initializer, TDec_Init)
                    print(f'{name} failed, skipping: {e}')
                    continue

                returns, losses = algorithm.run(hp['num_episodes'])

                with open(PATH_RESULTS, 'a', newline='') as f:
                    steps = getattr(algorithm, 'trained_steps', []) or [''] * len(returns)
                    csv.writer(f).writerows([env_name, seed, name, episode, r, l, s] for episode, (r, l, s) in enumerate(zip(returns, losses, steps)))

                print('  returns:', ' '.join(f'{r:8.1f}' for r in returns))
                if algorithm.trained_steps: print('  steps:  ', ' '.join(f'{s:8d}' for s in algorithm.trained_steps))

if __name__ == '__main__': main()

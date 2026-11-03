import csv, os

DEFAULT_HORIZON = 20
DEFAULT_EPISODES = 20
DEFAULT_TRAIN_STEPS = 1000
DEFAULT_CANDIDATES = 1000
DEFAULT_NUM_ENVS_COLLECT = 5
DEFAULT_NUM_STEPS_COLLECT = 10

def make_params(dictionary):
    if (k := 'horizon') not in dictionary: dictionary[k] = DEFAULT_HORIZON
    if (k := 'train_steps') not in dictionary: dictionary[k] = DEFAULT_TRAIN_STEPS
    if (k := 'num_episodes') not in dictionary: dictionary[k] = DEFAULT_EPISODES
    if (k := 'num_candidates') not in dictionary: dictionary[k] = DEFAULT_CANDIDATES
    if (k := 'num_envs_collect') not in dictionary: dictionary[k] = DEFAULT_NUM_ENVS_COLLECT
    if (k := 'num_steps_collect') not in dictionary: dictionary[k] = DEFAULT_NUM_STEPS_COLLECT
    return dictionary

SEED = 42

PRETRAIN_STEPS = [10000]

ENVS = {
    'reacher': 'Reacher-misc',
    'pendulum': 'Pendulum-v1',
    'swingup': 'CartPoleSwingUp-v0',
    'acrobot': 'AcrobotContinuous-v0',
    'quadrotor': 'PlanarQuadrotor-misc',
}

def clear_rows(path: str, columns: list, env: str, seed: int):
    rows = []
    if os.path.exists(path):
        with open(path, newline='') as f: rows = list(csv.reader(f))[1:]
    with open(path, 'w', newline='') as f: csv.writer(f).writerows([columns, *(r for r in rows if r[:2] != [env, str(seed)])])

PARAMS = {

    # return at most 0
    # the oracle reaches about -1 to -450
    'Pendulum-v1': make_params({'num_hidden': 8}),

    # return at most 200
    # about 145 to 175 for the oracle
    'CartPoleSwingUp-v0': make_params({
        'env_params': {'tau': 0.05},
        'num_hidden': 16,
    }),

    # return negative until the goal then +100 when crossing the line
    'AcrobotContinuous-v0': make_params({
        'env_params': {'dense_reward': True},
        'num_hidden': 16,
    }),

    # return at most 200
    # about 175 to 200 for the oracle
    'PlanarQuadrotor-misc': make_params({
        'num_hidden': 16,
        'horizon': 40,
    }),

    # return at most 100
    # about 10 to 100 for the oracle, mostly set by how far the goal starts
    'Reacher-misc': make_params({'num_hidden': 4}),
}

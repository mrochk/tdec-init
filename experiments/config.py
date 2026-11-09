import csv, os

DEFAULT_HORIZON = 20
DEFAULT_EPISODES = 10
DEFAULT_CANDIDATES = 1000
DEFAULT_NUM_ENVS_COLLECT = 5
DEFAULT_NUM_STEPS_COLLECT = 10

SEED = 42

def make_params(dictionary):
    if (k := 'horizon') not in dictionary: dictionary[k] = DEFAULT_HORIZON
    if (k := 'num_episodes') not in dictionary: dictionary[k] = DEFAULT_EPISODES
    if (k := 'num_candidates') not in dictionary: dictionary[k] = DEFAULT_CANDIDATES
    if (k := 'num_envs_collect') not in dictionary: dictionary[k] = DEFAULT_NUM_ENVS_COLLECT
    if (k := 'num_steps_collect') not in dictionary: dictionary[k] = DEFAULT_NUM_STEPS_COLLECT
    return dictionary

ENVS = {
    'reacher': 'Reacher-misc',
    'pendulum': 'Pendulum-v1',
    'swingup': 'CartPoleSwingUp-v0',
    'acrobot': 'AcrobotContinuous-v0',
    'quadrotor': 'PlanarQuadrotor-misc',
}

def clear_rows(path: str, columns: list, env: str, seed: int, inits: list | None = None):
    rows = []
    if os.path.exists(path):
        with open(path, newline='') as f: rows = list(csv.DictReader(f))
    rerun = lambda r: r['env'] == env and r['seed'] == str(seed) and (inits is None or r['init'] in inits)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, columns, restval='', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(r for r in rows if not rerun(r))

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
        'num_hidden': 12,
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

ENV_LABELS = {
    'Reacher-misc': 'Reacher',
    'Pendulum-v1': 'Pendulum',
    'PlanarQuadrotor-misc': 'Quadrotor',
    'CartPoleSwingUp-v0': 'CartPole',
    'AcrobotContinuous-v0': 'Acrobot',
}
PAPER_ENVS = list(ENV_LABELS)

METRICS = {
    'onestep': 'one-step error',
    'jacobian': 'Jacobian error',
}

COLORS = {'random': '#8C8C8C', 'decoupling': '#4B7BBE', 'values-heldout': '#CC5555', 'sobolev-heldout': '#3A8A84'}
ORACLE_COLOR = '#1D1D1D'
GRID_COLOR = '#E8E8E8'

PAPER_INITS = {'random': 'Random', 'values-heldout': 'Values', 'sobolev-heldout': 'Sobolev', 'decoupling': 'TDec-Init'}
SANS = {'TDec-Init'}
SANS_FONTS = ['Helvetica', 'Nimbus Sans', 'TeX Gyre Heros', 'Arial', 'DejaVu Sans']
TEXT_WIDTH = 7.0
PAPER_STYLE = {'random': ('', 'o'), 'values-heldout': ((4, 2), 's'), 'sobolev-heldout': ((1, 1.5), '^'), 'decoupling': ('', 'D')}

PAPER_RC = {
    'font.family': 'sans-serif', 'mathtext.fontset': 'stixsans',
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 8,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42,
}

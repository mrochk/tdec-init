import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import *
from plot_identification import best_of, legend_entries, paper_line, paper_legend, paper_save, report_missing

RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'results', 'mbrl.csv')
OUTPUT = os.path.join(os.path.dirname(RESULTS), 'plots')

def main():
    df = pd.read_csv(RESULTS)
    df = df[df.env.isin(PAPER_ENVS)].copy()
    df['episode'] += 1

    report_missing(df, [])

    oracle, df = df[df.init == 'oracle'], df[df.init != 'oracle'].copy()

    # the oracle return of the same env, seed and episode, nan without an oracle run
    ref = oracle.set_index(['env', 'seed', 'episode'])['return'].reindex(pd.MultiIndex.from_frame(df[['env', 'seed', 'episode']])).to_numpy()
    # capped at the oracle, the planner is not optimal and a model sometimes beats it by luck
    df['ratio'] = np.minimum(np.where(ref > 0, df['return'] / ref, ref / df['return']), 1)

    paper(df)

def paper(df):
    '''
    the paper figure, envs as columns, at the text width: the ratio to the oracle against the episode, and below
    the gradient steps of the training after each episode, until the held-out loss stopped improving
    '''
    plt.rcParams.update(PAPER_RC)
    episodes = sorted(df.episode.unique())
    # one variant per env, by the mean ratio over the episodes, per episode would select the noise
    df = best_of(df, ['env'], 'ratio', agg='mean', higher=True)
    # the rows of the runs before the column existed have no steps
    steps = df.dropna(subset=['train_steps']) if 'train_steps' in df else df.iloc[:0].assign(train_steps=[])

    fig, axes = plt.subplots(2, len(PAPER_ENVS), figsize=(TEXT_WIDTH, 3.0), sharex=True, sharey='row', squeeze=False)
    for col, env in enumerate(PAPER_ENVS):
        top, bottom = axes[0, col], axes[1, col]

        top.axhline(1, color='#0b0b0b', dashes=(4, 2), linewidth=0.9, label='oracle')
        for init, label in PAPER_INITS.items():
            paper_line(top, df[(df.env == env) & (df.init == init)].groupby('episode')['ratio'], init, label)
            paper_line(bottom, steps[(steps.env == env) & (steps.init == init)].groupby('episode')['train_steps'], init, label)
        top.set(ylim=(0, 1.05))
        top.set_title(ENV_LABELS[env])
        bottom.set_xlabel('episode', labelpad=1)

        for ax in (top, bottom):
            ax.set_xlim(episodes[0] - 0.5, episodes[-1] + 0.5)
            ax.set_xticks([1, *episodes[4::5]])
            ax.grid(True, color='#e6e5e0', linewidth=0.5)
            ax.tick_params(length=2, pad=1.5)

    # once all the envs are drawn, the shared limits cover all of them
    axes[1, 0].set_ylim(bottom=0)
    axes[0, 0].set_ylabel('return relative to oracle')
    axes[1, 0].set_ylabel('gradient steps\nafter the episode')
    # thousands of steps as 2k, 4k, ...
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v / 1000:g}k' if v else '0'))

    # the oracle last in the legend
    handles, labels = legend_entries(fig)
    paper_legend(fig, handles[1:] + handles[:1], labels[1:] + labels[:1])
    sns.despine(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.6, w_pad=0.6)
    paper_save(fig, 'paper_mbrl_ratio', OUTPUT)

if __name__ == '__main__': main()

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
    '''the paper figure: the ratio to the oracle against the episode, envs as columns, at the text width'''
    plt.rcParams.update(PAPER_RC)
    episodes = sorted(df.episode.unique())
    # one variant per env, by the mean ratio over the episodes, per episode would select the noise
    df = best_of(df, ['env'], 'ratio', agg='mean', higher=True)

    fig, axes = plt.subplots(1, len(PAPER_ENVS), figsize=(TEXT_WIDTH, 1.75), sharey=True, squeeze=False)
    for col, (ax, env) in enumerate(zip(axes[0], PAPER_ENVS)):
        ax.axhline(1, color='#0b0b0b', dashes=(4, 2), linewidth=0.9, label='oracle')
        for init, label in PAPER_INITS.items():
            paper_line(ax, df[(df.env == env) & (df.init == init)].groupby('episode')['ratio'], init, label)
        ax.set(ylim=(0, 1.05), xlim=(episodes[0] - 0.5, episodes[-1] + 0.5))
        ax.set_xticks([1, *episodes[4::5]])
        ax.grid(True, color='#e6e5e0', linewidth=0.5)
        ax.tick_params(length=2, pad=1.5)
        ax.set_title(ENV_LABELS[env])
        ax.set_xlabel('episode', labelpad=1)
        if col == 0: ax.set_ylabel('return relative to oracle')

    # the oracle last in the legend
    handles, labels = legend_entries(fig)
    paper_legend(fig, handles[1:] + handles[:1], labels[1:] + labels[:1])
    sns.despine(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.6)
    paper_save(fig, 'paper_mbrl_ratio', OUTPUT)

if __name__ == '__main__': main()

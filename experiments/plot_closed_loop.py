import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import *
from plot_identification import paper_rc, legend_entries, paper_line, paper_legend, paper_save, report_missing

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

def kilo(v):
    '''thousands of steps as 2k, 2.5k, ...'''
    return f'{v / 1000:.3g}k' if v >= 1000 else f'{v:.0f}'

def paper_bars(ax, totals: pd.DataFrame):
    '''one bar per initialization of PAPER_INITS with runs: the median of the totals over the seeds, the interquartile range as whiskers'''
    inits = [init for init in PAPER_INITS if init in set(totals.init)]
    for x, init in enumerate(inits):
        q25, median, q75 = totals[totals.init == init].train_steps.quantile([0.25, 0.5, 0.75])
        ax.bar(x, median, width=0.7, color=COLORS[init], zorder=2)
        ax.errorbar(x, median, yerr=[[median - q25], [q75 - median]], color=ORACLE_COLOR, linewidth=0.7, capsize=1.5, zorder=3)
    ax.set_xticks([])
    ax.set_xlim(-0.6, max(len(inits), 1) - 0.4)

def paper(df):
    '''
    the paper figure, envs as columns, at the text width: the ratio to the oracle against the episode, and below the
    total gradient steps of the trainings between the episodes, each until the held-out loss stopped improving
    '''
    paper_rc()
    episodes = sorted(df.episode.unique())
    # the rows of the runs before the column existed have no steps
    steps = df.dropna(subset=['train_steps']) if 'train_steps' in df else df.iloc[:0].assign(train_steps=[])
    # only the trainings between the episodes, the pretraining of the values and sobolev inits is left out on purpose
    totals = steps.groupby(['env', 'seed', 'init'], as_index=False).train_steps.sum()

    fig, axes = plt.subplots(2, len(PAPER_ENVS), figsize=(TEXT_WIDTH, 3.0), squeeze=False, gridspec_kw=dict(height_ratios=[1.4, 1]))
    for col, env in enumerate(PAPER_ENVS):
        top, bottom = axes[0, col], axes[1, col]

        top.axhline(1, color=ORACLE_COLOR, dashes=(4, 2), linewidth=0.9, label='Oracle')
        for init, label in PAPER_INITS.items():
            paper_line(top, df[(df.env == env) & (df.init == init)].groupby('episode')['ratio'], init, label)
        top.set(ylim=(0, 1.05), xlim=(episodes[0] - 0.5, episodes[-1] + 0.5))
        top.set_xticks([episodes[0], episodes[-1]]) # the first and last episodes only
        top.set_xlabel('episode', labelpad=1)
        top.set_title(ENV_LABELS[env])
        if col > 0: top.tick_params(labelleft=False)

        # the totals differ a lot between the envs, each has its own scale
        paper_bars(bottom, totals[totals.env == env])
        bottom.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: kilo(v)))
        bottom.yaxis.set_major_locator(plt.MaxNLocator(3))

        for ax in (top, bottom):
            ax.grid(True, axis='y', color=GRID_COLOR, linewidth=0.5)
            ax.tick_params(length=2, pad=1.5)
        top.grid(True, axis='x', color=GRID_COLOR, linewidth=0.5)

    axes[0, 0].set_ylabel('return relative to oracle')
    axes[1, 0].set_ylabel('gradient steps')

    # the oracle last in the legend
    handles, labels = legend_entries(fig)
    paper_legend(fig, handles[1:] + handles[:1], labels[1:] + labels[:1])
    sns.despine(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.6, w_pad=0.6)
    paper_save(fig, 'paper_mbrl_ratio', OUTPUT)

if __name__ == '__main__': main()

import os, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.offsetbox import HPacker, TextArea

from config import *

RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'results', 'init.csv')
OUTPUT = os.path.join(os.path.dirname(RESULTS), 'plots')

def main():
    df = pd.read_csv(RESULTS)
    df = df[df.env.isin(PAPER_ENVS)]

    report_missing(df, ['N'])
    paper(df)

def report_missing(df, group: list):
    '''print the runs with fewer seeds than their env, the failed decouplings are skipped and have no rows'''
    seeds = df.groupby('env').seed.nunique()
    runs = df.groupby(['env', *group, 'init']).seed.nunique()
    missing = runs[runs < seeds.reindex(runs.index.get_level_values('env')).to_numpy()]
    if len(missing): print('runs with missing seeds (number of seeds left):', missing.to_string(), sep='\n')

def best_of(df, group: list, metric: str, agg: str = 'median', higher: bool = False):
    '''the rows of the best variant of each method of BEST_OF per group (by the agg of metric), renamed to the method'''
    out = [df[~df.init.isin(sum(BEST_OF.values(), []))]]
    for method, variants in BEST_OF.items():
        rows = df[df.init.isin(variants)]
        if rows.empty: continue
        # a group without any value, e.g. a ratio without an oracle run, has no best variant and is left out
        score = rows.groupby(group + ['init'])[metric].agg(agg).dropna()
        if score.empty: continue
        best = pd.DataFrame((score.groupby(group).idxmax() if higher else score.groupby(group).idxmin()).tolist(), columns=group + ['init'])
        print(f'best {method} for {metric}:', best.groupby(group[:-1] or group).init.agg(' '.join).to_string(), sep='\n')
        out.append(rows.merge(best).assign(init=method))
    return pd.concat(out)

def paper_line(ax, groups, init: str, label: str):
    '''median line and interquartile band of the grouped values of an initialization, nothing if it has no runs'''
    if groups.ngroups == 0: return
    stats = groups.quantile([0.25, 0.5, 0.75]).unstack()
    color = COLORS[init]
    ax.fill_between(stats.index, stats[0.25], stats[0.75], color=color, alpha=0.15, linewidth=0)
    ax.plot(stats.index, stats[0.5], color=color, dashes=PAPER_STYLE[init][0], marker=PAPER_STYLE[init][1], markersize=3.5,
            linewidth=1.6 if init == 'decoupling' else 1.2, label=label, zorder=3 if init == 'decoupling' else 2)

def legend_entries(fig):
    '''the handles and labels of all the axes without repeats, an env without results has none'''
    entries = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()): entries.setdefault(label, handle)
    return list(entries.values()), list(entries)

def sans_fonts():
    '''the fonts of SANS_FONTS that are installed, matplotlib warns for each missing one'''
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return [font for font in SANS_FONTS if font in installed] or ['sans-serif']

def paper_legend(fig, handles, labels):
    '''one legend above the plots, the method names of SANS in sans serif and of SMALLCAPS in small caps'''
    legend = fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.0))
    for text in legend.get_texts():
        if text.get_text() in SANS: text.set_fontfamily(sans_fonts())

    # matplotlib has no small caps: the first letter at full size, the others in capitals at 80%, on the same baseline
    size = plt.rcParams['legend.fontsize']
    for column in legend._legend_handle_box.get_children():
        for item in column.get_children():
            handle, label = item.get_children()
            if label.get_text() not in SMALLCAPS: continue
            word = label.get_text()
            item._children[1] = HPacker(pad=0, sep=0, align='baseline', children=[
                TextArea(word[0], textprops=dict(fontsize=size)), TextArea(word[1:].upper(), textprops=dict(fontsize=0.8 * size))])

def paper_save(fig, name: str, output: str = OUTPUT):
    '''save a paper figure as pdf and png'''
    os.makedirs(output, exist_ok=True)
    path = os.path.join(output, f'{name}.pdf')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.01)
    #fig.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    plt.rcdefaults()
    print('saved', path)

def paper(df):
    '''the error of the models at initialization against N, envs as columns, the metrics of METRICS as rows'''
    plt.rcParams.update(PAPER_RC)
    Ns = sorted(df.N.unique())

    fig, axes = plt.subplots(len(METRICS), len(PAPER_ENVS), figsize=(TEXT_WIDTH, 3.6), sharex=True, squeeze=False)
    for row, (metric, name) in enumerate(METRICS.items()):
        # the best variant of each method for this metric
        best = best_of(df, ['env', 'N'], metric)
        for col, env in enumerate(PAPER_ENVS):
            ax = axes[row, col]
            for init, label in PAPER_INITS.items():
                paper_line(ax, best[(best.env == env) & (best.init == init)].groupby('N')[metric], init, label)
            ax.set(xscale='log', yscale='log')
            ax.set_xticks(Ns, ['' if n == 200 and 100 in Ns else str(n) for n in Ns]) # no room for 200 next to 100
            ax.minorticks_off()
            ax.yaxis.set_major_locator(plt.LogLocator(numticks=5))
            ax.grid(True, color='#e6e5e0', linewidth=0.5)
            ax.tick_params(length=2, pad=1.5)
            if row == 0: ax.set_title(ENV_LABELS[env])
            if row == len(METRICS) - 1: ax.set_xlabel('$N$', labelpad=1)
            if col == 0: ax.set_ylabel(name)

    paper_legend(fig, *legend_entries(fig))
    sns.despine(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.4, w_pad=0.4)

    paper_save(fig, 'paper_init')

if __name__ == '__main__': main()

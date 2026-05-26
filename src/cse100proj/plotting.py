import math
import mplcursors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os
import pickle
from cse100proj.utils import load_config

config = load_config()
PICKLES = config['data']['pickles_dir']


def find_nrows(n_models, n_cols=4):
    """ Given the number of models and the number of columns in the plot,
        return the number of rows needed to plot all models."""
    return math.ceil(n_models / n_cols)


def get_subplot_inds(n_cols, metric_ind):
    """ Given the number of columns and the index of the metric, 
        return the row and column indices for the subplot."""
    r = metric_ind // n_cols
    c = metric_ind % n_cols
    return r, c


def plot_errors(results1, results2, errors, nrows, ncols, scale, 
                ylim=None, threshold=None, 
                thresh_direction=None,
                limit_models_to=None,
                fontsize=5):
    """ Enhanced error plotting function that can take in 
        two results dictionaries and plot them on the same axes 
        for comparison. It also allows for optional y-axis limits 
        and threshold-based filtering of models.
    """
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(scale*ncols, scale*nrows))
    cmap = plt.get_cmap('viridis')
    
    if limit_models_to:
        results1 = {k: v for k, v in results1.items() if k in limit_models_to}
        results2 = {k: v for k, v in results2.items() if k in limit_models_to}
        
    models = set(results1.keys()) | set(results2.keys())
        
    color_array = cmap(np.linspace(0, 1, len(models)))
    colors = {model: color for model, color in zip(models, color_array)}
    plotted_lines = []  # for adding cursor interactivity later

    def add_errors_to_plot(results, line_type, marker, axes=axes, colors=colors):
        
        for model, info in results.items():
            
            for i, metric in enumerate(errors):
                
                r, c = get_subplot_inds(ncols, i)

                x = info['x']
                y = info[metric]
                
                if threshold and thresh_direction == 'higher':
                    try:
                        if max(y) < threshold[metric]:
                            continue
                    except:
                        pass
                    
                if threshold and thresh_direction == 'lower':
                    try:
                        if min(y) > threshold[metric]:
                            continue
                    except:
                        pass
                if len(y) == 0:
                    continue
                if len(x) != len(y):
                    print(f"Warning: length mismatch for {model} on {metric} (x: {len(x)}, y: {len(y)})")
                    x = x[:len(y)]
                    
                line, = axes[r][c].plot(x, y, 
                                marker=marker,
                                label=model, 
                                linestyle=line_type, 
                                color=colors[model])
                axes[r][c].set_xlabel('# of RQs Used')
                axes[r][c].set_ylabel(metric)
                
                if ylim and metric in ylim:
                    axes[r][c].set_ylim(ylim[metric])
                    
                # store metadata on the line for hover display
                line._hover_model = model
                line._hover_metric = metric
                line._hover_dataset = marker
                
                plotted_lines.append(line)
                    
        return axes
    axes = add_errors_to_plot(results1, line_type='-', marker='o')
    axes = add_errors_to_plot(results2, line_type='--', marker='x')
    
    def add_legend(axes, fontsize=fontsize):
        # put legend only on bottom-right subplot
        legend_ax = axes[-1, -1]
        handles, labels = axes[0][0].get_legend_handles_labels()
        print(handles, labels)

        # split 
        k = len(handles)//2
        spring_handles = handles[:k]
        spring_labels = labels[:k]

        fall_handles = handles[k:]
        fall_labels = labels[k:]

        header_spring = Line2D([], [], linestyle='none', label='Spring 2025')
        header_fall = Line2D([], [], linestyle='none', label='Fall 2025')

        handles = [header_spring] + spring_handles + [header_fall] + fall_handles
        labels = [h.get_label() for h in handles]

        legend = legend_ax.legend(
            handles,
            labels,
            loc='center',
            frameon=True,
            fontsize=fontsize,
        )    
        
        # bold section headers
        for text, label in zip(legend.get_texts(), labels):
            if label in ['Spring 2025', 'Fall 2025']:
                text.set_weight('bold')
                
        return axes
    axes = add_legend(axes)
    axes[-1][-1].grid(False)  # remove grid from legend subplot
    axes[-1][-1].axis('off')
            
    fig.suptitle('Model Performance Comparison')
    fig.set_tight_layout(True)
    
    # attach hover to all plotted lines
    cursor = mplcursors.cursor(plotted_lines, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        line = sel.artist
        x, y = sel.target

        sel.annotation.set_text(
            f"Dataset: {line._hover_dataset}\n"
            f"Model: {line._hover_model}\n"
            f"Metric: {line._hover_metric}\n"
            f"# RQs: {x:.0f}\n"
            f"Value: {y:.3f}"
        )
    
    return fig



## plotting grade distributions ##

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu, gaussian_kde
import numpy as np

kind = 'quartiles' # 'bar'

def plot_category_grades_by_quarter(
    df_sub,
    kind,
):

    assert kind in ['quartiles', 'bars'], "kind must be 'quartiles' or 'bars'"

    plt.clf()

    cols = ["Preparation", "Application", "Examination", "Overall"]

    if "Total" in df_sub.columns:
        df_sub = df_sub.rename(columns={"Total": "Overall"})

    df_long = df_sub.melt(
        id_vars="course",
        value_vars=cols,
        var_name="Category",
        value_name="Grade"
    )

    # Cleaner labels
    w = "Q1 (Remote exams)"
    f = "Q2 (In-person exams)"

    df_long["Quarter"] = df_long["course"].map({
        "winter2025": w,
        "fall2025": f,
    })

    # Muted, colorblind-friendlier colors
    palette = {
        w: "blue",
        f: "red",
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    ax = sns.violinplot(
        data=df_long,
        x="Category",
        y="Grade",
        hue="Quarter",
        split=True,
        palette=palette,
        inner=None,
        linewidth=1.0,
        cut=0,
        alpha=0.65,
        linecolor="black",
        ax=ax
    )

    # Cleaner y-axis
    ax.set_ylim(25, 104)
    ax.set_ylabel("Grade (%)", fontsize=11)
    ax.set_xlabel("")

    # Light horizontal grid only
    ax.grid(True, axis="y", linestyle="-", alpha=0.18)
    ax.grid(False, axis="x")

    # Move category labels to top
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelsize=11, length=0, pad=6)
    ax.tick_params(axis="y", labelsize=10)

    # Cleaner title
    ax.set_title(
        "Category Grade Distributions by Quarter",
        fontsize=14,
        pad=28
    )

    # Legend outside / top-left-ish
    ax.legend(
        title="",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.04),
        fontsize=9
    )

    # ----- add black lines for KDE mode -----

    offset = 0.001

    widths = {
        "Preparation": {"left": .26, "right": .20},
        "Application": {"left": .395, "right": .39},
        "Examination": {"left": .355, "right": .16},
        "Overall":     {"left": .24, "right": .15},
    }

    for i, cat in enumerate(cols):
        
        if kind == 'quartiles': continue

        for side, quarter in zip(["left", "right"], ["winter2025", "fall2025"]):

            data = df_sub[df_sub["course"] == quarter][cat].dropna().values

            if len(data) < 2:
                continue

            kde = gaussian_kde(data)
            grid = np.linspace(0, 100, 2000)
            density = kde(grid)

            peak_y = grid[np.argmax(density)]

            if side == "left":
                x_start = i - widths[cat][side]
                x_end = i - offset
            else:
                x_start = i + offset
                x_end = i + widths[cat][side]

            ax.plot(
                [x_start, x_end],
                [peak_y, peak_y],
                color="black",
                lw=1.5,
                solid_capstyle="butt",
                zorder=5
            )

    # ----- add vertical quartile bars -----

    quartile_offset = 0.03

    for i, cat in enumerate(cols):
        
        if kind == 'bars': continue

        for side, quarter in zip(
            ["left", "right"],
            ["winter2025", "fall2025"]
        ):

            data = df_sub[
                df_sub["course"] == quarter
            ][cat].dropna()

            q1, med, q3 = np.percentile(data, [25, 50, 75])

            if side == "left":
                x = i - quartile_offset
            else:
                x = i + quartile_offset

            # thin vertical line spanning Q1 to Q3
            ax.plot(
                [x, x],
                [q1, q3],
                color="black",
                lw=1.2,
                zorder=6
            )

            # median tick
            ax.plot(
                [x - 0.015, x + 0.015],
                [med, med],
                color="black",
                lw=1.8,
                zorder=7
            )
            
    # ---- compute and annotate statistics ----

    annotations = {
        "Preparation": ("Q1 < Q2", r"$*p < 10^{-3}$"),
        "Application": ("Q1 < Q2", r"$**p < 10^{-9}$"),
        "Examination": ("Q1 > Q2", r"$**p < 10^{-10}$"),
        "Overall": ("Q1 ≠ Q2", r"$p = 0.13$"),
    }

    null_hypotheses = {
        "Preparation": "Q1 ≤ Q2 (one-sided)",
        "Application": "Q1 ≤ Q2 (one-sided)",
        "Examination": "Q1 ≥ Q2 (one-sided)",
        "Overall": "Q1 = Q2 (two-sided)",
    }

    h1 = 21
    h2 = 17
    heights = {
        "Preparation": {"upper": 31, "lower": 27},
        "Application": {"upper": h1, "lower": h2},
        "Examination": {"upper": h1, "lower": h2},
        "Overall": {"upper": h1, "lower": h2},
    }

    stats_table = {}
    for i, cat in enumerate(cols):

        winter = df_sub[df_sub["course"] == "winter2025"][cat].dropna()
        fall = df_sub[df_sub["course"] == "fall2025"][cat].dropna()

        ks = ks_2samp(winter, fall)
        mw = mannwhitneyu(winter, fall)

        print(f"{cat} KS test: D={ks.statistic:.3g}, p={ks.pvalue:.3g}")
        print(f"{cat} Mann-Whitney U test: U={mw.statistic:.3g}, p={mw.pvalue:.3g}")
        print(f"{cat} Winter 2025: mean={winter.mean():.2f}, median={winter.median():.2f}")
        print(f"{cat} Fall 2025: mean={fall.mean():.2f}, median={fall.median():.2f}")
        print("---\n---\n")
        
        stats_table[cat] = [
            f"{mw.statistic:.2E}",
            f"{mw.pvalue:.2E}",
            null_hypotheses[cat]
        ]

        label, pval = annotations[cat]

        # Place annotation above each violin
        ax.text(
            i,
            # 102.2,
            heights[cat]['upper'],
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="0.2"
        )

        ax.text(
            i,
            # 99.9,
            heights[cat]['lower'],
            pval,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="0.2"
        )

    # Stats Table in latex form
    stats_table_df = pd.DataFrame(
        stats_table,
        index=["MWU U", "MWU p-val", "Null hypothesis"],
    )
    latex_table = stats_table_df.to_latex(
        escape=False,
        column_format="|l|c|c|c|c|",
        caption="Mann--Whitney U test results comparing category grades between quarters.",
        label="tab:mwu_results",
        bold_rows=True,

    )

    print(latex_table)

    # Clean spines
    sns.despine(ax=ax, left=False, bottom=True)

    plt.tight_layout()

    fout = f"out/category_grades_quarter_clean_{kind}.png"
    plt.savefig(
        fout,
        dpi=300,
        bbox_inches="tight"
    )
    print(f"\nSaved figure to {fout}\n")

    return plt
plt.show()
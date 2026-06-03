import math
from matplotlib import gridspec
import mplcursors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os
import pickle
from cse100proj.utils import load_config
from scipy.stats import ks_2samp, mannwhitneyu, gaussian_kde


import seaborn as sns


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

def plot_category_grades_by_quarter(
    df_sub,
    kind="quartiles",
):
    """ Takes in df_sub and returns a plt object. 
        kind can be 'quartiles' to show vertical bars for quartiles and median,
        or 'bars' to show KDE-based horizontal bars at the distribution peaks."""
    
    assert kind in ['quartiles', 'bars'], "kind must be 'quartiles' or 'bars'"

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


def plot_min_grade_by_quarter(
    df_sub,
    fout,
    title,
    labels,
    chisq_msg='',
    sep_category="course",
):
    """ Plot the minimum grade by quarter to check for 
        any differences in min grade cat distributions. """
    
    
    # Set desired course order
    course_order = list(labels.keys())

    # Set category order explicitly
    cat_order = ["Application", "Examination", "Preparation"]

    # Counts
    counts = (
        df_sub
        .groupby([sep_category, "min_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(course_order)
        .reindex(columns=cat_order)
    )

    # Proportions
    props = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    bottom = np.zeros(len(props))

    # Muted, publication-friendly colors
    color_map = {
        "Application": "#4C72B0",
        "Examination": "#6E6E6E",
        "Preparation": "#55A868"
    }

    x = np.arange(len(props.index))

    for cat in cat_order:
        ax.bar(
            x,
            props[cat],
            bottom=bottom,
            label=cat,
            color=color_map[cat],
            edgecolor="white",
            linewidth=1.2,
            width=0.82
        )
        bottom += props[cat].values

    # Add n labels above bars
    ns = counts.sum(axis=1)

    for i, course in enumerate(props.index):
        ax.text(
            i,
            1.025,
            f"n = {ns.loc[course]}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="0.25"
        )

    ax.set_title(title, fontsize=16, pad=12)
    ax.set_ylabel("Proportion of students", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.08)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in props.index], fontsize=12)

    # Clean axes
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # Better p-value formatting
    ax.text(
        0.5, -0.152,
        chisq_msg, # what the pvalue is
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11
    )

    # Legend outside
    ax.legend(
        title="Min category",
        frameon=False,
        bbox_to_anchor=(.95, 1),
        loc="upper left",
        fontsize=11,
        title_fontsize=12,
    )
    ax.grid(False)

    plt.tight_layout()
    
    fout = "out/min_category_comparison"
    plt.savefig(f"{fout}.png", bbox_inches="tight", dpi=600)
    plt.savefig(f"{fout}.svg", bbox_inches="tight")
    
    print(f"\nSaved figure to {fout}.png and {fout}.svg\n")
    
    
def plot_errors_gs(results1, results2, 
                errors=["pr_auc",   "f1", "recall", "precision", "accuracy"],
                ylabs=["PR-AUC", "F1 Score", "Precision", "Recall", "Accuracy"],
                special_model='CategoricalNB',
                ylim=None, threshold=None, 
                thresh_direction=None,
                limit_models_to=None,
                fontsize=5,
                figsize=(17, 17),
                width_ratios=[5, 2, 2],
                vline_at=13):
    """
    Plot model errors with a dedicated GridSpec legend column.
    """

    fig = plt.figure(figsize=figsize)

    # Main layout: plots on left, legend on right
    gs = gridspec.GridSpec(
        nrows=len(errors),
        ncols=3,
        width_ratios=width_ratios,
        wspace=0.35,
        hspace=0.25,
        figure=fig,
    )

    axes = [fig.add_subplot(gs[i, 0]) for i in range(len(errors))]
    
    legend_ax1 = fig.add_subplot(gs[:, 1])
    legend_ax2 = fig.add_subplot(gs[:, 2])
    
    legend_ax1.axis("off")
    legend_ax2.axis("off")

    cmap = plt.get_cmap("viridis")
    
    if limit_models_to:
        results1 = {k: v for k, v in results1.items() if k in limit_models_to}
        results2 = {k: v for k, v in results2.items() if k in limit_models_to}
        
    models = sorted(set(results1.keys()) | set(results2.keys()))
        
    color_array = cmap(np.linspace(0, 1, len(models)))
    colors = {model: color for model, color in zip(models, color_array)}

    plotted_lines = []

    def add_errors_to_plot(results, line_type, marker, dataset_label):
        for model, info in results.items():
            for i, metric in enumerate(errors):

                ax = axes[i]

                x = info["x"]
                y = info[metric]
                
                if threshold and thresh_direction == "higher":
                    try:
                        if max(y) < threshold[metric]:
                            continue
                    except Exception:
                        pass
                    
                if threshold and thresh_direction == "lower":
                    try:
                        if min(y) > threshold[metric]:
                            continue
                    except Exception:
                        pass

                if len(y) == 0:
                    continue

                if len(x) != len(y):
                    print(
                        f"Warning: length mismatch for {model} on {metric} "
                        f"(x: {len(x)}, y: {len(y)})"
                    )
                    x = x[:len(y)]
                    
                is_special = model == special_model
                line, = ax.plot(
                    x, y,
                    marker=marker,
                    label=f"{model} {dataset_label}",
                    linestyle=line_type,
                    color=colors[model],
                    linewidth=4 if is_special else 1.5,
                    zorder=10 if is_special else 1,
                    alpha=1.0 if is_special else 0.6
                )

                # x and y labels only on leftmost plots
                ax.set_ylabel(ylabs[i], fontsize=fontsize+3)
                if i == len(errors) - 1:
                    ax.set_xlabel("Number of RQs Used", fontsize=fontsize+3)
                else:
                    ax.set_xlabel("")

                if ylim and metric in ylim:
                    ax.set_ylim(ylim[metric])

                # # old mplcursors hover metadata
                # line._hover_model = model
                # line._hover_metric = metric
                # line._hover_dataset = dataset_label

                # plotted_lines.append(line)

    add_errors_to_plot(results1, line_type="-", marker="o", dataset_label="Q1")
    add_errors_to_plot(results2, line_type="--", marker="x", dataset_label="Q2")

    # Build legend manually from model colors
    # Get one handle per model
    handles1 = []
    handles2 = []
    
    labels1 = []
    labels2 = []
    
    seen = set()

    def process_label(label):
        """ Text processing to make model names more readable in the legend."""
        label = label.replace(" Q1", "").replace(" Q2", "")
        label = label.replace("Classifier", "")
        label = label.replace("DiscriminantAnalysis", "\nDiscriminantAnalysis")
        
        return label.strip()

    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        print(f"Handles: {len(h)}, Labels: {len(l)}")
        for handle, label in zip(h, l):
            if label not in seen:
                if "Q1" in label:
                    handles1.append(handle)
                    labels1.append(process_label(label))
                elif "Q2" in label:
                    handles2.append(handle)
                    labels2.append(process_label(label))
                seen.add(label)

        if vline_at:
            ax.axvline(vline_at, color='gray', linestyle='--', alpha=0.5)

        
    print(f"Length of handles1: {len(handles1)}, Length of labels1: {len(labels1)}")
    print(f"Length of handles2: {len(handles2)}, Length of labels2: {len(labels2)}")
    # Split legend into two columns manually

    legend_ax1.legend(
        handles1,
        labels1,
        loc="center left",
        frameon=False,
        fontsize=fontsize,
        title_fontsize=fontsize+3,
        title="Q1 Models"
    )

    legend_ax2.legend(
        handles2,
        labels2,
        loc="center left",
        frameon=False,
        fontsize=fontsize,
        title_fontsize=fontsize+3,
        title="Q2 Models"
    )
    
    fig.suptitle(
        "Model Performance Comparison",
        fontsize=fontsize+6,
        y=0.93
    )

    return fig
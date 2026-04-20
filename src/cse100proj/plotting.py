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


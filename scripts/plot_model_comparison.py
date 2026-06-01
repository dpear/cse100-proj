import mplcursors
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os
import pickle
from cse100proj.utils import load_config

from cse100proj.plotting import (
    plot_errors,
    plot_errors_gs,
)

from cse100proj.modeling import (
    filter_models_by_threshold,
    rank_df_based_on_metric,
    rank_models,
)

config = load_config()
PICKLES = config['data']['pickles_dir']
errors_bin = config['model_comparison']['binary_metrics']
errors_reg = config['model_comparison']['regression_metrics']

# LOAD
with open(PICKLES + '/results1_bin.pkl', 'rb') as f:
    results1_bin = pickle.load(f)

with open(PICKLES + '/results1_reg.pkl', 'rb') as f:
    results1_reg = pickle.load(f)

with open(PICKLES + '/results2_bin.pkl', 'rb') as f:
    results2_bin = pickle.load(f)

with open(PICKLES + '/results2_reg.pkl', 'rb') as f:
    results2_reg = pickle.load(f)
    
print("Loaded results from pickles:")
print(f"  - Binary results 1: {len(results1_bin)} models")
print(f"  - Binary results 2: {len(results2_bin)} models")
print(f"  - Regression results 1: {len(results1_reg)} models")
print(f"  - Regression results 2: {len(results2_reg)} models")


# FULL BINARY
nrows = 2
ncols = 4
scale = 6
f = plot_errors(results1_bin, results2_bin, 
                errors_bin, nrows, ncols, scale,
                ylim=None,
)
f.savefig('out/model_comparison_binary_full.svg')


# FULL REGRESSION
plt.clf()
nrows = 2
ncols = 3
scale = 6
f = plot_errors(results1_reg, results2_reg, 
                errors_reg, nrows, ncols, scale,
                ylim={'rmse': (0, 17), 'mae': (0, 20), 'r2': (-.05, 1)},
)
f.savefig('out/model_comparison_regression_full.svg')


# LIMIT TO TOP MODELS - BINARY
models_to_plot = []
for d in results1_bin, results2_bin:
    for error in errors_bin:
        ranked = rank_models(d, error, top_k=1, higher_is_better=True)
        models_to_plot.append(ranked)
        
models_to_plot = set([m for sublist in models_to_plot for m, s in sublist])
f = plot_errors(results1_bin, results2_bin, 
                errors_bin, 
                nrows=2, 
                ncols=3, 
                scale=6,
                # threshold={'recall': 0.4},
                # thresh_direction='higher'
                limit_models_to=models_to_plot
)
f.savefig('out/model_comparison/binary_top.png', dpi=300)


# LIMIT TO TOP MODELS - REGRESSION

models_to_plot = []
for d in results1_reg, results2_reg:
    for error in errors_reg:
        ranked = rank_models(d, error, top_k=2, higher_is_better=True)
        models_to_plot.append(ranked)
        
models_to_plot = set([m for sublist in models_to_plot for m, s in sublist])
f = plot_errors(results1_reg, results2_reg, 
                errors_reg, 
                nrows=2, 
                ncols=2, 
                scale=6,
                ylim={'rmse': (0, 17), 'mae': (0, 20), 'r2': (-.05, 1)},
                limit_models_to=models_to_plot,
                fontsize=10
)
f.savefig('out/model_comparison/regression_top2.png', dpi=300)


############

LIMITS = {
    'accuracy': .5,
    'precision': .5,
    'recall': .5,
    'f1': .5,
    'pr_auc': .6
}

metrics = ['accuracy', 'precision', 'recall', 'f1', 'pr_auc']
models = list(results1_bin.keys())

dfs = {}
for metric in metrics:
    dfs[metric] = rank_df_based_on_metric(metric=metric)
full_df = dfs[metrics[0]]
for i in range(1, len(metrics)):
    full_df = full_df.merge(dfs[metrics[i]], on='model')
    
filtered_df = filter_models_by_threshold(full_df, LIMITS)
top_models = filtered_df['model'].tolist()

f = plot_errors_gs(
    results1_bin,
    results2_bin,
    figsize=(17, 17),
    fontsize=15,
    width_ratios=[5, 1, 1],
    limit_models_to=top_models,
)
fname = 'binary_top_multi_metric_13_rqs'
f.savefig(f'out/model_comparison/{fname}.png', dpi=300)
f.savefig(f'out/model_comparison/{fname}.svg', dpi=300)
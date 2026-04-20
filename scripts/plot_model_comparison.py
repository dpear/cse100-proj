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
    
)

from cse100proj.modeling import (
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
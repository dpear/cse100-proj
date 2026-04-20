from cse100proj.utils import load_config

import pandas as pd
import numpy as np
import os

import logging
import pickle

from cse100proj.modeling import (
    report_error_bin,
    report_error_reg,
    build_model_registry,
    get_results_dict,
    make_metric_dfs,
    save_df_dict_to_csv,
)

config = load_config()

DIR = config['data']['processed_dir']
PICKLES = config['data']['pickles_dir']
files = os.listdir(DIR)
files = sorted(files)

df1_sel = pd.read_csv(DIR + '/winter2025.csv')  
df2_sel = pd.read_csv(DIR + '/fall2025.csv')

####### 

# if binary, use classification metrics; else regression metrics
error_fns = { 
    True: report_error_bin,
    False: report_error_reg,
}

# # --- set up logging ---
# logging.basicConfig(
#     filename='model_comparison.log',
#     filemode='w',  # overwrite each run ('a' to append)
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     force=True
# )

models, failed_models = build_model_registry()

# # --- logging instead of print ---
# logging.info(f'{len(models)} models loaded')
# for x in models.keys():
#     logging.info(f'LOADED: {x}')

# logging.info('---')
# logging.info(f'---{len(failed_models)} models failed to load')

# for x in failed_models.keys():
#     logging.info(f'FAILED: {x}')

models_bin = {k: v for k, v in models.items() if v['binary']}
models_reg = {k: v for k, v in models.items() if not v['binary']}

errors_reg = ['rmse', 'mae', 'r2']
errors_bin = ['accuracy', 'precision', 'recall', 'f1', 'pr_auc']

# logging.info(f"Number of models: {len(models)}")
# logging.info(f"Number of binary models: {len(models_bin)}")
# logging.info(f"Number of regression models: {len(models_reg)}")
# logging.info(f"---")
# logging.info(f"---")
# logging.info(f"---")


import warnings
warnings.filterwarnings("ignore")


# DATASET1: Winter 2025
results1_bin = get_results_dict(
    models=models_bin,
    error_fn=report_error_bin,
    errors=errors_bin,
    df=df1_sel,
    response_col='atrisk'
)
# logging.info(f"Finished binary models on dataset 1")

results1_reg = get_results_dict(
    models=models_reg,
    error_fn=report_error_reg,
    errors=errors_reg,
    df=df1_sel,
    response_col='Total'
)
# logging.info(f"Finished regression models on dataset 1")

# DATASET2: Fall 2025
results2_bin = get_results_dict(
    models=models_bin,
    error_fn=report_error_bin,
    errors=errors_bin,
    df=df2_sel,
    response_col='atrisk'
)
# logging.info(f"Finished binary models on dataset 2")

results2_reg = get_results_dict(
    models=models_reg,
    error_fn=report_error_reg,
    errors=errors_reg,
    df=df2_sel,
    response_col='Total'
)
# logging.info(f"Finished regression models on dataset 2")


# SAVE RESULTS AS PICKLES
with open(PICKLES+'results1_bin.pkl', 'wb') as f:
    pickle.dump(results1_bin, f)
    
with open(PICKLES+'results1_reg.pkl', 'wb') as f:
    pickle.dump(results1_reg, f)
    
with open(PICKLES+'results2_bin.pkl', 'wb') as f:
    pickle.dump(results2_bin, f)
    
with open(PICKLES+'results2_reg.pkl', 'wb') as f:
    pickle.dump(results2_reg, f)


# SAVE RESULTS

metric_dfs1 = make_metric_dfs(results1_bin, None)
metric_dfs1 = make_metric_dfs(results1_reg, metric_dfs1)

metric_dfs2 = make_metric_dfs(results2_reg, None)
metric_dfs2 = make_metric_dfs(results2_bin, metric_dfs2)


# save_df_dict_to_csv(
#     metric_dfs1, 'out/model_comparison/tables/winter2025', index=False)
# save_df_dict_to_csv(
#     metric_dfs2, 'out/model_comparison/tables/fall2025',   index=False)
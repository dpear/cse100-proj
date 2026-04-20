from cse100proj.utils import load_config

import pandas as pd
import numpy as np
import os
import re

import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, auc, precision_recall_curve
import inspect
import logging
from sklearn.utils import all_estimators

from cse100proj.preprocessing import (
    get_reg_cols,
)

config = load_config()

DIR = config['data']['processed_dir']
LOG = config['logging']['model_comparison_log']
files = os.listdir(DIR)
files = sorted(files)



def report_error_bin(clf, X, y, cv=5):
    """
    Evaluates a binary classification model using cross-validation.
    
    Parameters:
    - clf: The classifier model (must have fit and predict_proba methods).
    - X: Feature matrix (pandas DataFrame or numpy array).
    - y: Target vector (binary labels).
    - cv: Number of cross-validation folds (default=5).
    
    Returns:
    - dict: Dictionary with mean scores for accuracy, 
    precision, recall, f1, and pr_auc.
    """
    # Standard metrics
    scores = cross_validate(
        clf, X, y, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1'])
    
    #### PR-AUC requires probabilities, so compute manually
    pr_aucs = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    try:
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            clf.fit(X_train, y_train)
            y_proba = clf.predict_proba(X_test)[:, 1]  # Probability for positive class
            
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_aucs.append(auc(recall, precision))
    except Exception as e:
        # logging.warning(f"Failed to compute PR-AUC: {e}")
        pr_aucs = None
    result = {
        'accuracy': np.mean(scores['test_accuracy']),
        'precision': np.mean(scores['test_precision']),
        'recall': np.mean(scores['test_recall']),
        'f1': np.mean(scores['test_f1']),
    }
    
    if pr_aucs:
        result['pr_auc'] = np.mean(pr_aucs)
    
    return result


def report_error_reg(clf, X, y, cv=5):
    """ Evaluate a regression model using cross-validation.
        Reports rmse, mae, and r2 in a dictionary."""
    
    # RMSE
    mse_scores = cross_val_score(clf, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-mse_scores.mean())
    
    # MAE
    mae_scores = cross_val_score(clf, X, y, cv=cv, scoring='neg_mean_absolute_error')
    mae = -mae_scores.mean()
    
    # R²
    r2_scores = cross_val_score(clf, X, y, cv=cv, scoring='r2')
    r2 = r2_scores.mean()
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def can_instantiate(cls):
    """Return True if cls can be instantiated with no required args."""
    try:
        sig = inspect.signature(cls)
        for param in sig.parameters.values():
            if (
                param.default is inspect._empty
                and param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ):
                return False
        return True
    except Exception as e:
        # logging.warning(f'Failed to inspect {cls}: {e}')
        return False
    
    
def build_model_registry():
    
    # logger = logging.getLogger(LOG)
    models = {}
    failed_models = {}

    for name, cls in all_estimators(type_filter='classifier'):
        if can_instantiate(cls):
            try:
                models[name] = {
                    'model': cls(),
                    'binary': True,
                }
            except Exception as e:
                # logging.error(f'Failed to instantiate classifier {name}: {e}')
                pass
        else:
            failed_models[name] = cls
            # logging.warning(f'Failed to instantiate classifier {name}')

    for name, cls in all_estimators(type_filter='regressor'):
        if can_instantiate(cls):
            try:
                models[name] = {
                    'model': cls(),
                    'binary': False,
                }
            except Exception as e:
                # logging.error(f'Failed to instantiate regressor {name}: {e}')
                pass
        else:
            failed_models[name] = cls

    return models, failed_models


def get_results_dict(models, error_fn, errors, df, response_col):
    results = {}

    rq_cols = get_reg_cols(df, 'Reading')
    # logging.info(f"Found {len(rq_cols)} RQ columns")

    for model_name, info in models.items():
        # logging.info(f"Evaluating {model_name}...")
        
        results[model_name] = {k: [] for k in errors}
        results[model_name]['x'] = []


        # Train the model on increasing # of RSQs and record errors
        for k in range(1, len(rq_cols) + 1):
            X = df[rq_cols[:k]]
            y = df[response_col]

            try:
                per_k_results = error_fn(
                    clf=info['model'],
                    X=X,
                    y=y
                )

            except Exception as e:
                # logging.exception(
                #     f"Error evaluating {model_name} with {k} RQs "
                #     f"(X shape={X.shape}, y shape={y.shape})\n"
                #     f"Exception: {e}"
                # )
                continue

            for key, value in per_k_results.items():
                results[model_name][key].append(value)

            results[model_name]['x'].append(k)

    # logging.info("Finished evaluating all models")
    return results


def make_metric_dfs(d, add=None):

    """ Transforms the nested results dictionary into 
        a dictionary of DataFrames, where each DataFrame corresponds 
        to a metric and has columns for 'x' and the metric values 
        for each classifier.
        If 'add' is provided, it will append to the existing DataFrames 
        instead of creating new ones.
    """

    classifiers = list(d.keys())
    if add is None:
        metric_dfs = {}
    else:
        metric_dfs = add
        
    for metric in d[classifiers[0]].keys():
        metric_dfs[metric] = pd.DataFrame({'x': d[classifiers[0]]['x']})
        for classifier in classifiers:

            if metric == 'x':
                continue
            
            data = d[classifier][metric]
            x = d[classifier]['x']
            data = data + [None] * (len(x) - len(data))  # Pad with None if lengths differ
            df = pd.DataFrame({'x': x, classifier: data})
            metric_dfs[metric] = pd.merge(
                metric_dfs[metric], 
                df,
                on='x', 
                how='outer')
            
    return metric_dfs


def save_df_dict_to_csv(df_dict, out_dir, index=False, prefix=None):
    os.makedirs(out_dir, exist_ok=True)

    for name, df in df_dict.items():
        if isinstance(df, pd.DataFrame):
            filename = f"{name}.csv"
            path = os.path.join(out_dir, filename)
            df.to_csv(path, index=index)
        else:
            print(f"Skipping {name!r}: not a pandas DataFrame.")
            
            
def rank_models(results, metric, top_k=5, higher_is_better=True):
    """ Ranks models based on their performance on a specific metric.
        Returns the top_k models with their average metric value across all k RQs.
    """
    model_scores = {}
    
    for model, info in results.items():
        if metric in info:
            x = [i for i in info[metric] if i is not None]
            avg_score = np.mean(x)
            model_scores[model] = avg_score
    
    # Sort models based on score
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=higher_is_better)
    
    return sorted_models[:top_k]
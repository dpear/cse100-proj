from cse100proj.utils import load_config

from cse100proj.preprocessing import (
    get_df_sub,
)

from cse100proj.plotting import (
    plot_category_grades_by_quarter,
    plot_min_grade_by_quarter,
)

import os
import pandas as pd

config = load_config()
DIR = config['data']['input_dir']

files = os.listdir(DIR)
files = sorted(files)

df_sub = get_df_sub(files)


# Violin plot of grade category distributions by quarter
plot_category_grades_by_quarter(df_sub, kind='quartiles')


# By quarter (remote vs in-person exams)
plot_min_grade_by_quarter(
    df_sub, 
    fout=config['cat_comparison_by_q']['fout'],
    title=config['cat_comparison_by_q']['title'],
    labels=config['cat_comparison_by_q']['labels'], 
    sep_category="course"
)

## By at-risk status (failing vs passing students)
plot_min_grade_by_quarter(
    df_sub, 
    fout=config['cat_comparison_by_atrisk']['fout'],
    title=config['cat_comparison_by_atrisk']['title'],
    labels=config['cat_comparison_by_atrisk']['labels'], 
    sep_category="atrisk"
)
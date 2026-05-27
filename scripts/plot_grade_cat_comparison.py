from cse100proj.utils import load_config

from cse100proj.preprocessing import (
    get_df_sub,
)

from cse100proj.plotting import (
    plot_category_grades_by_quarter,
)

import os
import pandas as pd

config = load_config()
DIR = config['data']['input_dir']

files = os.listdir(DIR)
files = sorted(files)

df_sub = get_df_sub(files)

plot_category_grades_by_quarter(df_sub, kind='quartiles')

plot_min_grade_by_quarter(df_sub)
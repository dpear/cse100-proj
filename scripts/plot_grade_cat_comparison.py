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
fout = "out/min_category_comparison"
course_order = ["winter2025", "fall2025"]
labels = {
    "winter2025": "Q1 (Remote exams)",
    "fall2025": "Q2 (In-person exams)"
}
title = "Minimum Grade Category by Quarter"
plot_min_grade_by_quarter(
    df_sub, 
    fout,
    title,
    labels=labels, 
    sep_category="course"
)

## By at-risk status (failing vs passing students)
fout = "out/min_category_by_atrisk"
course_order = [True, False]
labels = {
    True: "Failing students",
    False: "Passing students"
}
title = "Minimum Grade Category by Passing Status"
plot_min_grade_by_quarter(
    df_sub, 
    fout,
    title,
    labels=labels, 
    sep_category="course"
)
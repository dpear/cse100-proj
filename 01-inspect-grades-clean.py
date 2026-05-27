import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

import os
import re

from scipy.stats import gaussian_kde, ks_2samp, mannwhitneyu
DIR = "./data/raw/"
files = os.listdir(DIR)
files = sorted(files)

def get_df(f):
    """ Read in the grades excel file, skip first row """
    df = pd.read_excel(DIR + f)  # Skip the first two rows of metadata
    df = df.iloc[1:] # Remove the first row ("points out of")
    return df

def find_title(f):
    """ If file is in the format XX-Grades_<course>_X_<quarter>.xlsx, return <course>_<quarter> """
    match = re.search(r'(\d{4})\-\d+_(Fall|Winter|Spring|Summer)', f)
    year_term = f"{match.group(1)}_{match.group(2)}"
    return year_term

def get_score_col_name(df, score_type='exam'):
    """ Return the name of the final grade column that is a percentage. 
        If exam=False, return the final score column instead.
    """
    
    if score_type == 'total':
        return 'Total'
    if score_type == 'exam':
        cols = [x for x in df.columns if 'Final' in x]
        name = df[cols].sum().idxmin()
        return name
    if score_type == 'midterm':
        cols = [x for x in df.columns if 'Midterm' in x]
        name = df[cols].sum().idxmin()
        return name

def get_inperson_section_name(df):
    """ Return the name of the in-person value for the 'Section' column"""
    names = df['Section'].unique()
    for name in names:
        if not 'R' in name:
            return name
    return None

def get_remote_section_name(df):
    """ Return the name of the remote value for the 'Section' column"""
    names = df['Section'].unique()
    for name in names:
        if 'R' in name:
            return name
    return None
df = get_df(files[0])
midterm_columns = [x for x in df.columns if 'Midterm' in x]
midterm_columns
fig, axes = plt.subplots(1, 8, figsize=(28, 4))

def generate_plots(score_type='exam'):
    
    for i in range(len(files)):
        print('Processing', files[i])
        ax = axes[i]
        file_name = files[i]
        title = find_title(file_name)
        df = get_df(file_name)

        inperson = get_inperson_section_name(df)
        remote = get_remote_section_name(df)
        score = get_score_col_name(df, score_type=score_type)

        left = df.loc[df["Section"] == inperson, score].dropna().to_numpy()
        right = df.loc[df["Section"] == remote, score].dropna().to_numpy()

        try:
            y = np.linspace(min(left.min(), right.min()), max(left.max(), right.max()), 2000)
            skip_left = False
            skip_right = False
        except ValueError:
            if len(left) == 0:
                print(f"{title} in person side is empty")
                skip_left = True
            if len(right) == 0:
                print(f"{title} remote side is empty")
                skip_right = True

        def kde_width_at(kde, y_grid, widths, y0):
            return float(np.interp(y0, y_grid, widths))

        ax.axvline(0, color="black", linewidth=1)

        if not skip_left:
            kde_left = gaussian_kde(left)
            w_left = kde_left(y)
            ax.fill_betweenx(y, -w_left, 0, alpha=0.6, label='in person', color='green')
            
            grid_l = np.linspace(left.min(), left.max(), 2000)  # dense grid helps
            dens_l = kde_left(grid_l)
            mode_est_l = grid_l[np.argmax(dens_l)]  # y-value where KDE is maximal
            wl = kde_width_at(kde_left, grid_l, dens_l, mode_est_l)
            ax.hlines(mode_est_l, xmin=-wl, xmax=0, linewidth=1.8, color="black")

        if not skip_right:
            kde_right = gaussian_kde(right)
            w_right = kde_right(y)
            ax.fill_betweenx(y, 0, w_right, alpha=0.6, label='remote', color='grey')

            grid_r = np.linspace(right.min(), right.max(), 2000)  # dense grid helps
            dens_r = kde_right(grid_r)
            mode_est_r = grid_r[np.argmax(dens_r)]  # y-value where KDE is maximal
            wr = kde_width_at(kde_right, grid_r, dens_r, mode_est_r)
            ax.hlines(mode_est_r, xmin=0, xmax=wr, linewidth=1.8, color="black")

        
        if not skip_left and not skip_right:
            ks = ks_2samp(left, right, alternative="two-sided", mode="auto")
            ax.text(
                0.5, 0.98,
                f"KS test: D={ks.statistic:.3g}, p={ks.pvalue:.3g}",
                transform=ax.transAxes,
                ha="center", va="top"
            )
            
            u, p = mannwhitneyu(left, right, alternative="two-sided") 
            ax.text(
                0.5, 0.0,
                f"Mann-Whitney U: U={u:.3g}, p={p:.3g}",
                transform=ax.transAxes,
                ha="center", va="top"
            )

        ax.set_xticks([])
        ax.set_ylabel("value")
        if score_type == 'total':
            ax.set_ylim(40, 100)
        ax.set_title(title)
        ax.set_xlim(-max(w_left.max() if not skip_left else 0, w_right.max() if not skip_right else 0)*1.1,
                         max(w_left.max() if not skip_left else 0, w_right.max() if not skip_right else 0)*1.1)
        print(f'\tMODE inperson = {mode_est_l:.3g}, remote = {mode_est_r:.3g}')
        print(f'\tMEAN inperson = {np.mean(left):.3g}, remote = {np.mean(right):.3g}')
        print(f'\tMED inperson median = {np.median(left):.3g}, remote median = {np.median(right):.3g}')
        print(f'\tKS test: D={ks.statistic:.3g}, p={ks.pvalue:.3g}')
        print(f'\tMann-Whitney U: U={u:.3g}, p={p:.3g}')
    plt.tight_layout()
    axes[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.02))
    return fig

fig = generate_plots(score_type='exam')
fig.suptitle("Final Exam Distributions by Section", fontsize=16, y=1.03)
fig.savefig("out/final_exam_distributions.png", bbox_inches="tight", dpi=300)
plt.show()
plt.clf()

fig, axes = plt.subplots(1, 8, figsize=(28, 4))
fig = generate_plots(score_type='total')
fig.suptitle("Final Grade Distributions by Section", fontsize=16, y=1.03)
fig.savefig("out/final_grade_distributions.png", bbox_inches="tight", dpi=300)
plt.show()
plt.clf()
fig, axes = plt.subplots(1, 8, figsize=(28, 4))
fig = generate_plots(score_type='midterm')
fig.suptitle("Midterm Grade Distributions by Section", fontsize=16, y=1.03)
fig.savefig("out/midterm_grade_distributions.png", bbox_inches="tight", dpi=300)
plt.show()
d = {
    
 '01-Grades_CSE100_2022-4_Fall.xlsx':   {'year': 2022, 'quarter': 'fall'},
 '02-Grades_CSE100_2023-1_Winter.xlsx': {'year': 2023, 'quarter': 'winter'},
 '03-Grades_CSE100_2023-4_Fall.xlsx':   {'year': 2023, 'quarter': 'fall'},
 '04-Grades_CSE100_2024-1_Winter.xlsx': {'year': 2024, 'quarter': 'winter'},
 '05-Grades_CSE100_2024-2_Spring.xlsx': {'year': 2024, 'quarter': 'spring'},
 '06-Grades_CSE100_2024-4_Fall.xlsx':   {'year': 2024, 'quarter': 'fall'},
 '07-Grades_CSE100_2025-1_Winter.xlsx': {'year': 2025, 'quarter': 'winter'},
 '08-Grades_CSE100_2025-4_Fall.xlsx':   {'year': 2025, 'quarter': 'fall'},
 
}
dfs = pd.DataFrame()
for i,f in enumerate(files):
    df = get_df(f)
    
    midterm_col = get_score_col_name(df, score_type='midterm')
    exam_col    = get_score_col_name(df, score_type='exam')
    total_col   = get_score_col_name(df, score_type='total')

    df = df[[midterm_col, exam_col, total_col, 'Section']]
    df.rename(columns={midterm_col: 'Midterm', exam_col: 'Final Exam', total_col: 'Total'}, inplace=True)
    df['year'] = d[f]['year']
    df['quarter'] = d[f]['quarter']
    df['Section'] = df['Section'].apply(lambda x: 'in person' if x == get_inperson_section_name(df) else 'remote')
    df['course'] = df['quarter'].astype(str) + df['year'].astype(str)
    dfs = pd.concat([dfs, df], ignore_index=True)
    df['exam_type'] = 'inperson'
    
dfs.head()
metric1 = 'quarter'
counts = (
    dfs
    .groupby([metric1, 'Section'])
    .size()
    .reset_index(name="count")
)

sns.barplot(
    data=counts,
    x=metric1,
    y="count",
    hue="Section",
    palette=["green", "grey"]
)

plt.ylabel("# Students")
plt.title("Number of Students by quarter, separated by section")
plt.show()

metric1 = 'year'
counts = (
    dfs
    .groupby([metric1, 'Section'])
    .size()
    .reset_index(name="count")
)

sns.barplot(
    data=counts,
    x=metric1,
    y="count",
    hue="Section",
    palette=["green", "grey"]
)

plt.ylabel("# Students")
plt.title("Number of Students by year, separated by section")
plt.show()

metric1 = 'year'
counts = (
    dfs
    .groupby(['course', 'Section'])
    .size()
    .reset_index(name="count")
)
order = ['fall2022', 'winter2023', 'fall2023', 'winter2024', 'spring2024', 'fall2024', 'winter2025', 'fall2025']

g = sns.barplot(
    data=counts,
    x='course',
    y="count",
    hue="Section",
    palette=["green", "grey"],
    order=order,
)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.ylabel("# Students")
plt.title("Number of Students by year and quarter, separated by section")
plt.show()

metrics = ["Midterm", "Final Exam", "Total"]

df_long = dfs.melt(
    id_vars=["year", "quarter"],
    value_vars=metrics,
    var_name="assessment",
    value_name="score"
)

df_long["quarter"] = pd.Categorical(df_long["quarter"], ["winter", "spring", "fall"], ordered=True)
df_long["assessment"] = pd.Categorical(df_long["assessment"], metrics, ordered=True)

sns.set_theme(style="whitegrid")
plt.show()
g = sns.catplot(
    data=df_long,
    x="year", y="score",
    col="assessment",
    kind="box",
    sharey=False,
    height=4, aspect=1.1
)

g.set_axis_labels("Year", "Score")
for ax in g.axes.flat:
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

q_means = (
    df_long.groupby(["quarter", "assessment"], observed=True)["score"]
    .mean()
    .reset_index()
)

g = sns.catplot(
    data=q_means,
    x="quarter", y="score",
    col="assessment",
    kind="bar",
    sharey=False,
    height=4, aspect=1.1,
    errorbar='ci'
)

g.set_axis_labels("Quarter", "Average score")
g.set_titles("{col_name}")
plt.tight_layout()
plt.show()

x = 'Final Exam'
y = 'Total'
hue = 'course'

sns.scatterplot(
    data=dfs,
    x=x,
    y=y,
    hue=hue,
    alpha=.7,
    markers='o'
)
import seaborn as sns
import matplotlib.pyplot as plt

sections = ['remote', 'in person']   # <-- replace with your actual section names
df_plot = dfs.dropna()
df_plot = dfs[dfs['Final Exam'] > 0.3]

hue = 'course'

sns.lmplot(
    data=df_plot,
    x='Final Exam',
    y='Total',
    hue=hue,
    height=5,
    aspect=1.2,
    scatter_kws={'alpha': 0.6},
    palette='viridis',
)

plt.xlabel('Final Exam Score')
plt.ylabel('Total Score')
plt.title(f'Final Exam vs Total by {hue}')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

predictors = ['Final Exam', 'Midterm']   # change as needed

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, x in zip(axes, predictors):
    g = sns.lmplot(
        data=dfs,
        x=x,
        y='Total',
        hue='course',
        palette='viridis',
    )
    
    g.ax.set_xlabel(x)
    g.ax.set_title(f'{x} vs Total')

plt.tight_layout()
plt.show()

sections = ['remote', 'in person']   # <-- replace with your actual section names
df_plot = dfs.dropna()
df_plot = dfs[dfs['Midterm'] > 0.2]

hue = 'course'

sns.lmplot(
    data=df_plot,
    x='Midterm',
    y='Total',
    hue=hue,
    height=5,
    aspect=1.2,
    scatter_kws={'alpha': 0.6},
    palette='viridis',
)

plt.xlabel('Midterm Score')
plt.ylabel('Total Score')
plt.title(f'Midterm vs Total by {hue}')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

hue_vars = ["course", "year", "quarter"]
x = 'Midterm'
y = 'Total'

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, hue in zip(axes, hue_vars):
    sns.scatterplot(
        data=dfs,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        palette='viridis',
    )
    sns.regplot(
        data=dfs,
        x=x,
        y=y,
        scatter=False,
        ax=ax,
        color="black",
    )
    ax.set_title(f"{y} by {x} separated by {hue}")

plt.tight_layout()

from matplotlib import cm
from matplotlib.lines import Line2D

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import linregress
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

def make_three_panel_regression_figure(
    df=dfs,
    x='Midterm',
    y='Total',
    hue_vars=("course", "year", "quarter"),
    ci=False,
    annotate_max_lines=8,
    height=4.2,
    aspect=1.15,
):
    """
    Creates a 1x3 panel plot of x vs y.
    Each panel uses a different hue variable; within panel, plots per-group regression lines,
    annotates slope and R^2 per group, and reports ANCOVA interaction p-value for slope differences.

    Returns:
        g: seaborn FacetGrid
        results: dict of per-panel summary tables (group, n, slope, r2)
    """
    
    df_long = df.melt(
        id_vars=[x, y],
        value_vars=list(hue_vars),
        var_name="hue_var",
        value_name="group",
    ).dropna(subset=[x, y, "group"])

    df_long["group"] = df_long["group"].astype(str)

    g = sns.FacetGrid(
        df_long,
        col="hue_var",
        col_order=list(hue_vars),
        sharex=True,
        sharey=True,
        height=height,
        aspect=aspect,
        despine=False,
    )

    results = {}

    def _panel_plot(data, **kws):
        ax = plt.gca()

        levels = list(pd.unique(data["group"]))
        if len(levels) > annotate_max_lines: # huge annotation box
            print(
                f"Warning: panel '{data['hue_var'].iloc[0]}' has",
                f"{len(levels)} groups; annotation box may be large."
            )
            pass

        cmap = cm.get_cmap("viridis")
        palette = [cmap(v) for v in np.linspace(0, 1, len(levels))]
        color_map = dict(zip(levels, palette))

        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue="group",
            hue_order=levels,
            palette=color_map,
            alpha=0.45,
            s=35,
            ax=ax,
            legend=False,  # we'll add one global legend later
        )
        ax.set_ylim(0)  # Assuming scores can't be negative; adjust as needed
        ax.set_xlim(0)  

        rows = []
        legend_handles = []
        for lvl in levels:
            d = data[data["group"] == lvl]
            n = len(d)
            if n < 2:
                continue

            lr = linregress(d[x], d[y])
            slope = lr.slope
            r2 = lr.rvalue**2

            sns.regplot(
                data=d,
                x=x,
                y=y,
                scatter=False,
                ci=ci,
                color=color_map[lvl],
                ax=ax,
                truncate=False,
            )
            
            legend_handles.append(
                Line2D(
                    [0], [0],
                    color=color_map[lvl],
                    lw=2,
                    label=f"{lvl}: R² = {r2:.2f}"
                )
            )

            rows.append({"group": lvl, "n": n, "slope": slope, "r2": r2})
        print(rows)
        panel_name = data["hue_var"].iloc[0]
        panel_table = pd.DataFrame(rows).sort_values("group")
        results[panel_name] = panel_table

        try:
            model = smf.ols(f"{y} ~ {x} * C(group)", data=data).fit()
            aov = anova_lm(model, typ=2)  # Type II ANOVA
            interaction_term = f"{x}:C(group)"
            p_slope_diff = float(aov.loc[interaction_term, "PR(>F)"])
        except Exception:
            p_slope_diff = np.nan

        lines = []
        for _, r in panel_table.iterrows(): # e.g., "A: m=0.31, R²=0.42"
            lines.append(f"{r['group']}: R²={r['r2']:.3g}")
        if len(lines) > annotate_max_lines: # If too many, truncate (opt))
            lines = lines[:annotate_max_lines] + ["…"]

        leg = ax.legend(
            handles=legend_handles,
            title="R² by group",
            loc="lower right",
            frameon=True,
            fontsize=9,
            title_fontsize=10
        )
        ax.add_artist(leg)

        
        p_txt = "p(slope diff) = NA" if np.isnan(p_slope_diff) else f"p(slope diff) = {p_slope_diff:.2e}"
        ax.text(
            0.02,
            -0.18,
            p_txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
        ax.set_title(f"{x} by {y} separated by {panel_name}")

    g.map_dataframe(_panel_plot)

    g.set_axis_labels(x, y)
    plt.tight_layout()

    return g, results

g, results = make_three_panel_regression_figure()
plt.show()
g, results = make_three_panel_regression_figure(dfs,
                                                x='Final Exam',
                                                y='Total',
                                                hue_vars=('Section', 'year')
                                                )
plt.show()

dfs_sub = dfs[dfs['course'].isin(['winter2025', 'fall2025'])]
g, results = make_three_panel_regression_figure(dfs_sub,
                                                x='Midterm',
                                                y='Total',
                                                hue_vars=('Section', 'course')
                                                )

fig, axes = plt.subplots(1, 3, figsize=(12, 5))

y = 'Final Exam'

for y, ax in zip(['Midterm', 'Final Exam', 'Total'], axes):

    sns.boxplot(
        data=dfs_sub,
        x='course',
        y=y,
        palette='viridis',
        ax=ax
    )

    left = dfs_sub.loc[dfs_sub["course"] == 'fall2025', y].dropna().to_numpy()
    right = dfs_sub.loc[dfs_sub["course"] == 'winter2025', y].dropna().to_numpy()
    ks = ks_2samp(left, right, alternative="two-sided", mode="auto")
    u, p = mannwhitneyu(left, right, alternative="two-sided") 
    
    ax.text(
        0.5, 0.2,
        f"KS test: D={ks.statistic:.3g}, p={ks.pvalue:.3g}\nMann-Whitney U: U={u:.3g}, p={p:.3g}",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=10
    )
    ax.set_title(f"{y} by course")
    ax.set_ylabel("")

    print(f'KS test {y}: D={ks.statistic:.3g}, p={ks.pvalue:.3g}')
    print(f'Mann-Whitney U test {y}: U={u:.3g}, p={p:.3g}')

sns.boxplot(
    data=dfs_sub,
    x='course',
    y='Total',
    palette=["green", "grey"]
)
plt.show()

y = 'Final Exam'
left = dfs_sub.loc[dfs_sub["course"] == 'fall2025', y].dropna().to_numpy()
right = dfs_sub.loc[dfs_sub["course"] == 'winter2025', y].dropna().to_numpy()
ks = ks_2samp(left, right, alternative="two-sided", mode="auto")
print(f'KS test {y}: D={ks.statistic:.3g}, p={ks.pvalue:.3g}')
df1 = get_df(files[6])
df2 = get_df(files[7])

df1['quarter'] = 'winter'
df1['year'] = 2025
df1['course'] = df1['quarter'].astype(str) + df1['year'].astype(str)
df1['exam_type'] = 'inperson'         

df2['quarter'] = 'fall'
df2['year'] = 2025
df2['course'] = df2['quarter'].astype(str) + df2['year'].astype(str)
df2['exam_type'] = 'inperson'    

print(df.columns)

df = pd.concat([df1, df2], ignore_index=True)
df_sub = df[['Preparation','Application', 'Examination', 'Total', 'course']]

df_sub['min_category'] = df_sub[['Preparation','Application', 'Examination']].idxmin(axis=1)

import pandas as pd
from scipy.stats import chi2_contingency

ct = pd.crosstab(df_sub["course"], df_sub["min_category"])
chi2, p, dof, expected = chi2_contingency(ct)
print(f"Chi-square statistic: {chi2:.3f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p:.4e}")

df_sub

cats = df_sub['min_category'].unique()
n_cat = len(cats)

counts = (
    df_sub
    .groupby(["course", "min_category"])
    .size()
    .unstack(fill_value=0)
)

props = counts.div(counts.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(6, 4))

bottom = np.zeros(len(props))

cmap = cm.get_cmap("Accent")
palette = [cmap(v) for v in np.linspace(0, 1, n_cat)]
color_map = dict(zip(cats, palette))

for cat in props.columns:
    ax.bar(
        props.index,
        props[cat],
        bottom=bottom,
        label=cat,
        color=color_map.get(cat),
        edgecolor="white"
    )
    bottom += props[cat]

ax.set_ylabel("Proportion of students")
ax.set_xlabel("")
ax.set_ylim(0, 1)
ax.legend(title="Min category", frameon=False)

ax.text(
    0.3, -0.15,
    f"Chi-square p-value: {p:.4e}", transform=ax.transAxes)

plt.tight_layout()
plt.show()

df_long
course_order = ["winter2025", "fall2025"]
plt.clf()
cats = df_sub["min_category"].unique()
n_cat = len(cats)
labels = {'winter2025': 'Winter 2025\n (remote, unproctored)', 'fall2025': 'Fall 2025\n (in-person, proctored)'}
counts = (
    df_sub
    .groupby(["course", "min_category"])
    .size()
    .unstack(fill_value=0)
    .reindex(course_order)
)

props = counts.div(counts.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(6, 4))

bottom = np.zeros(len(props))

cmap = cm.get_cmap("Accent")
palette = [cmap(v) for v in np.linspace(0, 1, n_cat)]
color_map = dict(zip(cats, palette))

for cat in props.columns:
    ax.bar(
        props.index,
        props[cat],
        bottom=bottom,
        label=cat,
        color=color_map.get(cat),
        edgecolor="white"
    )
    bottom += props[cat]

ns = counts.sum(axis=1)

for i, course in enumerate(props.index):
    ax.text(
        i,
        1.02,
        f"n = {ns.loc[course]}",
        ha="center",
        va="bottom"
    )

ax.set_ylabel("Proportion of students")
ax.set_xlabel("")
ax.set_ylim(0, 1.08)

ax.legend(
    title="Min category",
    frameon=True,
    loc="upper right",
    bbox_to_anchor=(1.4, 1)
)
ax.set_xticklabels(["Winter 2025", "Fall 2025"])

ax.text(
    0.3, -0.15,
    f"Chi-square p-value: {p:.4e}",
    transform=ax.transAxes
)
plt.tight_layout()
plt.title('Min Grade Category by Quarter')
plt.savefig("out/min_category_comparison.png", bbox_inches="tight", dpi=300)
plt.show()
course_order = ["winter2025", "fall2025"]
labels = {
    "winter2025": "Q1 (Remote exams)",
    "fall2025": "Q2 (In-person exams)"
}

cat_order = ["Application", "Examination", "Preparation"]

counts = (
    df_sub
    .groupby(["course", "min_category"])
    .size()
    .unstack(fill_value=0)
    .reindex(course_order)
    .reindex(columns=cat_order)
)

props = counts.div(counts.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(6.2, 4.2))

bottom = np.zeros(len(props))

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

ax.set_title("Minimum Grade Category by Quarter", fontsize=16, pad=12)
ax.set_ylabel("Proportion of students", fontsize=12)
ax.set_xlabel("")
ax.set_ylim(0, 1.08)

ax.set_xticks(x)
ax.set_xticklabels([labels[c] for c in props.index], fontsize=12)

ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

ax.text(
    0.5, -0.152,
    r"$\chi^2$ test: $p < 0.001$",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=11
)

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
plt.savefig("out/min_category_comparison.png", bbox_inches="tight", dpi=600)
plt.savefig("out/min_category_comparison.svg", bbox_inches="tight")
plt.show()
g, results = make_three_panel_regression_figure(df_sub,
                                                x='Examination',
                                                y='Preparation',
                                                hue_vars=('min_category', 'course'),
                                                )

g, results = make_three_panel_regression_figure(df_sub,
                                                x='Examination',
                                                y='Application',
                                                hue_vars=('min_category', 'course'),
                                                )

g, results = make_three_panel_regression_figure(df_sub,
                                                x='Application',
                                                y='Preparation',
                                                hue_vars=('min_category', 'course'),
                                                )

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_long = df.melt(
    id_vars="course",
    value_vars=["Preparation", "Application", "Examination", "Total"],
    var_name="Category",
    value_name="Grade"
)

plt.figure(figsize=(8,5))

sns.violinplot(
    data=df_long,
    x="course",
    y="Grade",
    hue="Category",
    inner="quartile",
    cut=0
)

plt.ylim(0, 100)
plt.ylabel("Grade (%)")
plt.xlabel("")
plt.title("Grade Distributions by Quarter and Assessment Category")

plt.legend(title="")
plt.tight_layout()
plt.show()

cat_map = {
    'Preparation': {'alternative':'greater'},
    'Application': {'alternative':'greater'},
    'Examination': {'alternative':'less'},
    'Overall':       {'alternative':'two-sided'},

}

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

def generate_plots_exams(data=df_long, axes=axes):
    
    for ax, category in zip(axes, ['Preparation', 'Application', 'Examination', 'Overall']):
        
        df = data[data['Category'] == category]
        print('Processing', category)
        split_on = 'course'
        splits = ['winter2025', 'fall2025']
        score = 'Grade'
        
        left  = df.loc[df[split_on] == splits[0], score].dropna().to_numpy()
        right = df.loc[df[split_on] == splits[1], score].dropna().to_numpy()

        try:
            y = np.linspace(min(left.min(), right.min()), max(left.max(), right.max()), 2000)
            skip_left = False
            skip_right = False
        except ValueError:
            if len(left) == 0:
                print(f"{splits[0]} side is empty")
                skip_left = True
            if len(right) == 0:
                print(f"{splits[1]}side is empty")
                skip_right = True

        def kde_width_at(kde, y_grid, widths, y0):
            return float(np.interp(y0, y_grid, widths))

        ax.axvline(0, color="black", linewidth=1)

        if not skip_left:
            kde_left = gaussian_kde(left)
            w_left = kde_left(y)
            ax.fill_betweenx(y, -w_left, 0, alpha=0.6, label=splits[0], color='lightblue')
            
            grid_l = np.linspace(left.min(), left.max(), 2000)  # dense grid helps
            dens_l = kde_left(grid_l)
            mode_est_l = grid_l[np.argmax(dens_l)]  # y-value where KDE is maximal
            wl = kde_width_at(kde_left, grid_l, dens_l, mode_est_l)
            ax.hlines(mode_est_l, xmin=-wl, xmax=0, linewidth=1.8, color="black")

        if not skip_right:
            kde_right = gaussian_kde(right)
            w_right = kde_right(y)
            ax.fill_betweenx(y, 0, w_right, alpha=0.6, label=splits[1], color='red')

            grid_r = np.linspace(right.min(), right.max(), 2000)  # dense grid helps
            dens_r = kde_right(grid_r)
            mode_est_r = grid_r[np.argmax(dens_r)]  # y-value where KDE is maximal
            wr = kde_width_at(kde_right, grid_r, dens_r, mode_est_r)
            ax.hlines(mode_est_r, xmin=0, xmax=wr, linewidth=1.8, color="black")

        
        if not skip_left and not skip_right:
            print(cat_map[category])
            ks = ks_2samp(left, right, alternative=cat_map[category]['alternative'], mode="auto")
            ax.text(
                0.5, 0.98,
                f"KS test: D={ks.statistic:.3g}, p={ks.pvalue:.3g}",
                transform=ax.transAxes,
                ha="center", va="top"
            )
            
            u, p = mannwhitneyu(left, right, alternative=cat_map[category]['alternative']) 
            ax.text(
                0.5, 0.0,
                f"Mann-Whitney U: U={u:.3g}, p={p:.3g}",
                transform=ax.transAxes,
                ha="center", va="top"
            )

        ax.set_xticks([])
        ax.set_ylabel("Grade")
        
        ax.set_title(category)
        ax.set_xlim(-max(w_left.max() if not skip_left else 0, w_right.max() if not skip_right else 0)*1.1,
                     max(w_left.max() if not skip_left else 0, w_right.max() if not skip_right else 0)*1.1)
        print(f'\tMODE {splits[0]}={mode_est_l}\t{splits[1]}={mode_est_r}')
        print(f'\tMEAN {splits[0]}={np.mean(left)}\t{splits[1]}={np.mean(right)}')
        print(f'\tMED  {splits[0]}={np.median(left)}\t{splits[1]}={np.median(right)}')
        print(f'\tKS test: D={ks.statistic:.3g}, p={ks.pvalue:.3g}')
        print(f'\tMann-Whitney U: U={u:.3g}, p={p:.3g}')
    plt.tight_layout()
    axes[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.02))
    return fig

fig = generate_plots_exams()
fig.suptitle("Final Grade Category Distributions by Quarter", fontsize=16, y=1.03)
fig.savefig("out/category_quarter_violin.png", bbox_inches="tight", dpi=300)
plt.show()
df.columns
df_long.loc[df_long["Category"] == "Overall", "Category"] = "Total"
df_sub.rename(columns={'Total':'Overall'})
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu, gaussian_kde
import numpy as np

plt.clf()

cols = ["Preparation", "Application", "Examination", "Overall"]
if 'Total' in df_sub.columns:
    df_sub = df_sub.rename(columns={'Total':'Overall'})
df_long = df_sub.melt(
    id_vars="course",
    value_vars=cols,
    var_name="Category",
    value_name="Grade"
)

w = "Winter 2025 (online unproctored)"
f = "Fall 2025 (in-person proctored)"

df_long["Quarter"] = df_long["course"].map({
    "winter2025": w,
    "fall2025": f,
})

palette = {
    w: "blue",
    f: "red"
}

plt.figure(figsize=(8,5))
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

ax = sns.violinplot(
    data=df_long,
    x="Category",
    y="Grade",
    hue="Quarter",
    split=True,
    palette=palette,
    inner=None,
    linewidth=1,
    cut=0,
    alpha=0.6,
    linecolor='black',
)

plt.ylim(0,100)
plt.ylabel("Grade (%)")
plt.xlabel("")
plt.legend(title="", frameon=False)

max_half_width = 0.4   # same as seaborn violin width
offset = 0.001         # tiny center gap

widths = {
    "Preparation": {"left": .26, "right": .20},
    "Application": {"left": .40, "right": .39},
    "Examination": {"left": .36, "right": .16},
    "Overall":     {"left": .24, "right": .15},
}

for i, cat in enumerate(cols):

    for side, quarter in zip(["left","right"], ["winter2025","fall2025"]):

        data = df_sub[df_sub["course"] == quarter][cat].values
        kde = gaussian_kde(data)
        grid = np.linspace(0,100,2000)
        density = kde(grid)
        peak_index = np.argmax(density)
        peak_y = grid[peak_index]
        peak_density = density[peak_index]
        print('\t', side, peak_density)

        if side == "left":
            x_start = i - widths[cat][side]
            x_end   = i - offset
        else:
            x_start = i + offset
            x_end = i + widths[cat][side]

        ax.plot(
            [x_start, x_end],
            [peak_y, peak_y],
            color="black", lw=2.5, solid_capstyle="butt", zorder=5
        )

for i, cat in enumerate(cols):

    winter = df_sub[(df_sub["course"]=="winter2025")][cat]
    fall   = df_sub[(df_sub["course"]=="fall2025")][cat]

    ks = ks_2samp(winter, fall)
    mw = mannwhitneyu(winter, fall)
    
    print(f"{cat} KS test: D={ks.statistic:.3g}, p={ks.pvalue:.3g}")
    print(f"{cat} Mann-Whitney U test: U={mw.statistic:.3g}, p={mw.pvalue:.3g}")
    print(f"{cat} Winter 2025: mean={winter.mean():.2f}, median={winter.median():.2f}")
    print(f"{cat} Fall 2025: mean={fall.mean():.2f}, median={fall.median():.2f}")
    print('---\n---\n')
    y = -3
    s = 8
    ofst = .03
    if cat == 'Overall':
        ax.text(i-ofst, y, "WI25", color="blue", ha="right", va="center", fontsize=s)
        ax.text(i, y, " ≠ ", color="black", ha="center", va="center", fontsize=s)
        ax.text(i+ofst, y, "FA25", color="red", ha="left", va="center", fontsize=s)  
        ax.text(i+ofst+0.135, y, " (p < 0.13)", color="black", ha="left", va="center", fontsize=s)
        
    if cat == 'Preparation':
        ax.text(i-ofst, y, "WI25", color="blue", ha="right", va="center", fontsize=s)
        ax.text(i, y, " < ", color="black", ha="center", va="center", fontsize = s)
        ax.text(i+ofst, y, "FA25", color="red", ha="left", va="center", fontsize = s)  
        ax.text(i+ofst+0.155, y, r"(p < $10^{-3}$)"+"*", color="black", ha="left", va="center", fontsize=s)
        
    if cat == 'Application':
        ax.text(i-ofst, y, "WI25", color="blue", ha="right", va="center", fontsize=s)
        ax.text(i, y, " < ", color="black", ha="center", va="center", fontsize = s)
        ax.text(i+ofst, y, "FA25", color="red", ha="left", va="center", fontsize = s)  
        ax.text(i+ofst+0.155, y, r"(p < $10^{-9}$)"+"*", color="black", ha="left", va="center", fontsize=s)

    if cat == 'Examination':
        ax.text(i-ofst, y, "WI25", color="blue", ha="right", va="center", fontsize=s, alpha=0.8)
        ax.text(i, y, " > ", color="black", ha="center", va="center", fontsize = s)
        ax.text(i+ofst, y, "FA25", color="red", ha="left", va="center", fontsize = s)  
        ax.text(i+ofst+0.155, y, r"(p < $10^{-10}$)"+"*", color="black", ha="left", va="center", fontsize=s)

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_title("Category Grades by Quarter", fontsize=16, y=1.11)

sns.despine()
plt.tight_layout()

plt.savefig('out/category_grades_quarter.png')
plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu, gaussian_kde
import numpy as np

kind = 'quartiles' # 'bar'
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

w = "Q1 (Winter 2025)"
f = "Q2 (Fall 2025)"

df_long["Quarter"] = df_long["course"].map({
    "winter2025": w,
    "fall2025": f,
})

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

ax.set_ylim(25, 104)
ax.set_ylabel("Grade (%)", fontsize=11)
ax.set_xlabel("")

ax.grid(True, axis="y", linestyle="-", alpha=0.18)
ax.grid(False, axis="x")

ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
ax.tick_params(axis="x", labelsize=11, length=0, pad=6)
ax.tick_params(axis="y", labelsize=10)

ax.set_title(
    "Category Grade Distributions by Quarter",
    fontsize=14,
    pad=28
)

ax.legend(
    title="",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.01, 0.04),
    fontsize=9
)

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

        ax.plot(
            [x, x],
            [q1, q3],
            color="black",
            lw=1.2,
            zorder=6
        )

        ax.plot(
            [x - 0.015, x + 0.015],
            [med, med],
            color="black",
            lw=1.8,
            zorder=7
        )
        

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

    ax.text(
        i,
        heights[cat]['upper'],
        label,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.2"
    )

    ax.text(
        i,
        heights[cat]['lower'],
        pval,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.2"
    )

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

sns.despine(ax=ax, left=False, bottom=True)

plt.tight_layout()

plt.savefig(
    f"out/category_grades_quarter_clean_{kind}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.clf()
sns.boxplot(
    data=df_long,
    x="Category",
    y="Grade",
    hue="Quarter",
    palette=palette,
    showfliers=True,
)
plt.show()
df1 = get_df(files[6])

df1['quarter'] = 'winter'
df1['year'] = 2025
df1['course'] = df1['quarter'].astype(str) + df1['year'].astype(str)
df1['exam_type'] = 'inperson'
df1['remote'] = df1['Section'].apply(lambda x: 'remote' if x == get_remote_section_name(df1) else 'in person')        

df2 = get_df(files[7])

df2['quarter'] = 'fall'
df2['year'] = 2025
df2['course'] = df2['quarter'].astype(str) + df2['year'].astype(str)
df2['exam_type'] = 'inperson' 
def select_and_rename(df):
    """ Selects columns from df that have '.1' in their name, 
        renames them by removing the ' (1)' suffix, 
        and returns the resulting dataframe.
    """
    renamed = {}
    df_cols = []
    new_df_cols = []
    for x in df.columns:
        print(x)
        if '.1' in x:
            renamed[x] = x.split(' (')[0]
            df_cols.append(x)
            new_df_cols.append(renamed[x])
        if x in ['quarter', 'year', 'course', 'exam_type',
                 'Preparation', 'Application', 'Examination', 
                 'Total', 'remote']:
            print(f'found {x}')
            df_cols.append(x)
            new_df_cols.append(x)

    df = df[df_cols]
    df = df.rename(columns=renamed)
    return df

df1_sel = select_and_rename(df1)
print('\n\n')
print(df2.columns.tolist())
df2_sel = select_and_rename(df2)
df1_sel['min_category'] = df1_sel[['Preparation','Application', 'Examination']].idxmin(axis=1)
df2_sel['min_category'] = df2_sel[['Preparation','Application', 'Examination']].idxmin(axis=1)

df = pd.concat([df1_sel, df2_sel], ignore_index=True)

df1_sel[['min_category']]
qs = df1_sel.columns[10:35].tolist()
l = 5
data = df1_sel
y = 'Final'
kind = 'avg_' #  'avg_' or 'sum_'

qsubsets = {
    f'subset{i//l+1}': qs[i:i+l] for i in range(0, len(qs), l)   
}

for subset, qlist in qsubsets.items():
    print(subset)
    df1_sel[f'sum_{subset}'] = df1_sel[qlist].sum(axis=1)
    df1_sel[f'avg_{subset}'] = df1_sel[qlist].mean(axis=1)
    
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, subset in zip(axes, qsubsets.keys()):
    sns.scatterplot(
        data=data,
        y=y,
        x=f'{kind}{subset}',
        palette='viridis',
        ax=ax
    )
    ax.set_title(subset)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    sns.regplot(
        data=data,
        x=f'{kind}{subset}',
        y=y,
        scatter=False,
        ax=ax,
        truncate=False,
    )
    
    lr = linregress(data[f'{kind}{subset}'], data[y])
    slope = lr.slope
    r2 = lr.rvalue**2
    
    ax.text(
        0.05, 0.95,
        f"Slope={slope:.3g}\nR²={r2:.3g}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10
    )
    
plt.tight_layout()
plt.suptitle(f'{y} vs. Quiz Subset Scores', fontsize=16, y=1.02)
plt.show()  
qs = df1_sel.columns[10:35].tolist()
l = 25
data = df1_sel
y = 'Midterm'
kind = 'avg_' # or 'avg_'

qsubsets = {
    f'subset{i//l+1}': qs[i:i+l] for i in range(0, len(qs), l)   
}

for subset, qlist in qsubsets.items():
    print(subset)
    df1_sel[f'sum_{subset}'] = df1_sel[qlist].sum(axis=1)
    df1_sel[f'avg_{subset}'] = df1_sel[qlist].mean(axis=1)
    
fig, axes = plt.subplots(1, 1, figsize=(4, 4))

sns.scatterplot(
    data=data,
    y=y,
    x=f'{kind}{subset}',
    palette='viridis',
    ax=axes
)
axes.set_title(subset)
axes.set_xlabel("")
axes.set_ylabel("")

sns.regplot(
    data=data,
    x=f'{kind}{subset}',
    y=y,
    scatter=False,
    ax=axes,
    truncate=False,
)

lr = linregress(data[f'{kind}{subset}'], data[y])
slope = lr.slope
r2 = lr.rvalue**2

axes.text(
    0.05, 0.95,
    f"Slope={slope:.3g}\nR²={r2:.3g}",
    transform=axes.transAxes,
    ha="left", va="top",
    fontsize=10
)
print(f"Slope={slope:.3g}, R²={r2:.3g}")

plt.tight_layout()
plt.suptitle(f'{y} vs. Quiz Subset Scores', fontsize=16, y=1.02)
plt.show()  
make_three_panel_regression_figure(
    df=df1_sel,
    x='avg_subset1',
    y='Midterm',
    hue_vars=('min_category', 'remote','course'),
)
df1 = get_df(files[6])
df2 = get_df(files[7])

df1['quarter'] = 'winter'
df1['year'] = 2025
df1['course'] = df1['quarter'].astype(str) + df1['year'].astype(str)
df1['exam_type'] = 'inperson'
df1['remote'] = df1['Section'].apply(lambda x: 'remote' if x == get_remote_section_name(df1) else 'in person')        

df2['quarter'] = 'fall'
df2['year'] = 2025
df2['course'] = df2['quarter'].astype(str) + df2['year'].astype(str)
df2['exam_type'] = 'inperson'    
df2['remote'] = df2['Section'].apply(lambda x: 'remote' if x == get_remote_section_name(df2) else 'in person')        

def select_and_rename(df):
    """ Selects columns from df that have '.1' in their name, 
        renames them by removing the ' (1)' suffix, 
        and returns the resulting dataframe.
    """
    renamed = {}
    df_cols = []
    new_df_cols = []
    for x in df.columns:
        if '.1' in x:
            renamed[x] = x.split(' (')[0]
            df_cols.append(x)
            new_df_cols.append(renamed[x])
        if x in ['quarter', 'year', 'course', 'exam_type',
                 'Preparation', 'Application', 'Examination', 
                 'Total', 'remote']:
            df_cols.append(x)
            new_df_cols.append(x)

    df = df[df_cols]
    df = df.rename(columns=renamed)
    return df

df1_sel = select_and_rename(df1)
df2_sel = select_and_rename(df2)
df1_sel['min_category'] = df1_sel[['Preparation','Application', 'Examination']].idxmin(axis=1)
df2_sel['min_category'] = df2_sel[['Preparation','Application', 'Examination']].idxmin(axis=1)

df = pd.concat([df1_sel, df2_sel], ignore_index=True)

data = df1_sel
y = 'Final'
y = 'Midterm'
y = 'Examination'

def get_reg_cols(df, s):
    return [col for col in df.columns if s in col]

reg_cols = get_reg_cols(df1_sel, 'Reading')
reg_cols += get_reg_cols(df1_sel, 'Programming')
reg_cols += get_reg_cols(df1_sel, 'Project')

f"{y} ~ " + " + ".join(reg_cols)
model = smf.ols(
    f'{y} ~ Q("' + '") + Q("'.join(reg_cols) + '")', 
    data=data).fit()

print(model.summary())
import pandas as pd
import matplotlib.pyplot as plt

coefs = model.params
conf = model.conf_int()
conf.columns = ["lower", "upper"]

df_plot = pd.concat([coefs, conf], axis=1).reset_index()
df_plot.columns = ["term", "coef", "lower", "upper"]

df_plot = df_plot[df_plot["term"] != "Intercept"]

fig, ax = plt.subplots(figsize=(5,10))

ax.errorbar(
    df_plot["coef"],
    df_plot["term"],
    xerr=[
        df_plot["coef"] - df_plot["lower"],
        df_plot["upper"] - df_plot["coef"]
    ],
    fmt='o'
)

ax.axvline(0, linestyle='--')
ax.set_xlabel("Effect on Examination")
ax.set_ylabel("Predictor")

plt.show()
y_pred = model.predict(df)

plt.scatter(df["Examination"], y_pred)
plt.plot([0, 100], [0, 100], linestyle='--')  # perfect fit line

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Model Fit")
plt.ylim(60, 100)
plt.xlim(60, 100)

plt.show()
resid = model.resid
fitted = model.fittedvalues

plt.scatter(fitted, resid)
plt.axhline(0, linestyle='--')

plt.xlabel("Fitted values")
plt.ylabel("Residuals")

plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_regression_summary_panels(
    model,
    df,
    outcome_col=None,
    rename_terms=None,
    figsize=(14, 4.5),
    title=None,
    color='blue'
):
    """
    Create a 3-panel regression summary figure:
      A) Coefficient plot with 95% CI
      B) Predicted vs actual
      C) Residuals vs fitted

    Parameters
    ----------
    model : statsmodels regression results object
        A fitted model from statsmodels, e.g. smf.ols(...).fit()
    df : pd.DataFrame
        DataFrame used for fitting
    outcome_col : str, optional
        Name of the outcome column. If None, tries to infer from model.model.endog_names
    rename_terms : dict, optional
        Mapping from model term names to prettier labels for plotting
    figsize : tuple
        Figure size
    title : str, optional
        Optional overall figure title

    Returns
    -------
    fig, axes
    """
    if outcome_col is None:
        outcome_col = model.model.endog_names

    if rename_terms is None:
        rename_terms = {}

    coef = model.params.copy()
    conf = model.conf_int().copy()
    pvals = model.pvalues.copy()

    coef_df = pd.DataFrame({
        "term": coef.index,
        "coef": coef.values,
        "lower": conf[0].values,
        "upper": conf[1].values,
        "pval": pvals.values,
    })

    coef_df = coef_df[coef_df["term"] != "Intercept"].copy()

    coef_df["label"] = coef_df["term"].map(
        lambda x: rename_terms.get(x, x)
    )

    coef_df = coef_df.sort_values("coef")
    coef_df = coef_df.reset_index(drop=True)

    def stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""

    coef_df["stars"] = coef_df["pval"].apply(stars)

    y_true = df[outcome_col].values
    y_pred = model.predict(df)
    resid = model.resid
    fitted = model.fittedvalues

    r2 = model.rsquared if hasattr(model, "rsquared") else None

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax0, ax1, ax2 = axes

    y_pos = np.arange(len(coef_df))

    xerr = np.vstack([
        coef_df["coef"] - coef_df["lower"],
        coef_df["upper"] - coef_df["coef"]
    ])

    ax0.errorbar(
        coef_df["coef"],
        y_pos,
        xerr=xerr,
        fmt="o",
        capsize=3,
    )
    ax0.axvline(0, linestyle="--", linewidth=1)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(coef_df["label"])
    ax0.set_xlabel("Coefficient estimate")
    ax0.set_title("Effect sizes")
    ax0.grid(True, axis="x", alpha=0.3)
    ax0.set_axisbelow(True)

    for i, row in coef_df.iterrows():
        ax0.text(
            row["upper"],
            i,
            f"  {row['stars']}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax1.scatter(y_true, y_pred, alpha=0.7)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=1,
    )

    ax1.set_xlabel("Observed")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Model fit")
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    if r2 is not None:
        ax1.text(
            0.05,
            0.95,
            f"$R^2$ = {r2:.3f}",
            transform=ax1.transAxes,
            ha="left",
            va="top",
        )

    ax2.scatter(fitted, resid, alpha=0.7)
    ax2.axhline(0, linestyle="--", linewidth=1)
    ax2.set_xlabel("Fitted values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residual biases")
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    if title is not None:
        fig.suptitle(title, y=1.05, fontsize=14)

    plt.tight_layout()
    return fig, axes

def get_reg_cols(df, s):
    return [col for col in df.columns if s in col]

data = df1_sel
y = 'Examination'

reg_cols = get_reg_cols(df1_sel, 'Reading')
reg_cols += get_reg_cols(df1_sel, 'Programming')
reg_cols += get_reg_cols(df1_sel, 'Project')

model = smf.ols(
    f'{y} ~ Q("' + '") + Q("'.join(reg_cols) + '")', 
    data=data).fit()

fig, axes = plot_regression_summary_panels(
    model=model,
    df=df1_sel,
    outcome_col=y,
    title=f"Regression summary: {y} as outcome",
    figsize=(10,7),

)

plt.show()

data = df2_sel
if data.equals(df1_sel):
    title = "Score distributions for PAs and Projects (Winter 2025)"
else:
    title = "Score distributions for PAs and Projects (Fall 2025)"

cols = [x for x in data.columns if 'Proj' in x or 'Prog' in x]
pans = len(cols)

fig, axes = plt.subplots(1, pans, figsize=(4*pans, 4))
for ax, col in zip(axes, cols):
    df1_sel[col].hist(bins=20, ax=ax)
    ax.set_title(col)   
    ax.set_ylim(0,400)
plt.suptitle(title, fontsize=16)
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

fig, ax = plt.subplots(figsize=(6, 4))
target = "Examination"
colors = ['blue', 'red']

for data, color, label in zip([df1_sel, df2_sel], colors, ['Winter 2025', 'Fall 2025']):

    rq_cols = get_reg_cols(data, 'Reading')
    rmse_values = []
    num_rqs = []

    for k in range(1, len(rq_cols) + 1):
        X = data[rq_cols[:k]]
        y = data[target]
        print(y.mean())
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        rmse_values.append(rmse)
        num_rqs.append(k)

    ax.plot(num_rqs, rmse_values, marker='o', color=color, label=label)
    
ax.set_xlabel("Number of RQs Used")
ax.legend()
ax.set_ylabel("RMSE")
ax.set_title("Error vs Number of RQs")
ax.grid(True)
plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

fig, ax = plt.subplots(figsize=(6, 4))
target = "Examination"
colors = ['blue', 'red']

for data, color, label in zip([df1_sel, df2_sel], colors, ['Winter 2025', 'Fall 2025']):

    rq_cols = get_reg_cols(data, 'Reading')
    rmse_values = []
    num_rqs = []

    for k in range(1, len(rq_cols) + 1):
        X = data[rq_cols[:k]]
        y = data[target]
        print(y.mean())
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        rmse_values.append(rmse)
        num_rqs.append(k)

    ax.plot(num_rqs, rmse_values, marker='o', color=color, label=label)
    
ax.set_xlabel("Number of RQs Used")
ax.legend()
ax.set_ylabel("RMSE")
ax.set_title("Error vs Number of RQs")
ax.grid(True)
plt.show()

from sklearn.model_selection import cross_val_score
import numpy as np

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, auc, precision_recall_curve
import numpy as np

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
    scores = cross_validate(
        clf, X, y, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1'])
    
    pr_aucs = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]  # Probability for positive class
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_aucs.append(auc(recall, precision))
    
    result = {
        'accuracy': np.mean(scores['test_accuracy']),
        'precision': np.mean(scores['test_precision']),
        'recall': np.mean(scores['test_recall']),
        'f1': np.mean(scores['test_f1']),
        'pr_auc': np.mean(pr_aucs)
    }
    
    return result

def report_error_reg(clf, X, y, cv=5):
    
    mse_scores = cross_val_score(clf, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-mse_scores.mean())
    
    mae_scores = cross_val_score(clf, X, y, cv=cv, scoring='neg_mean_absolute_error')
    mae = -mae_scores.mean()
    
    r2_scores = cross_val_score(clf, X, y, cv=cv, scoring='r2')
    r2 = r2_scores.mean()
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

import sklearn

    

error_fns = { # if binary, use classification metrics; else regression metrics
    True: report_error_bin,
    False: report_error_reg,
}
import inspect
import logging
from sklearn.utils import all_estimators

logging.basicConfig(
    filename='model_reg.log',
    filemode='w',  # overwrite each run ('a' to append)
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

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
        logging.warning(f'Failed to inspect {cls}: {e}')
        return False

def build_model_registry():
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
                logging.error(f'Failed to instantiate classifier {name}: {e}')
        else:
            failed_models[name] = cls

    for name, cls in all_estimators(type_filter='regressor'):
        if can_instantiate(cls):
            try:
                models[name] = {
                    'model': cls(),
                    'binary': False,
                }
            except Exception as e:
                logging.error(f'Failed to instantiate regressor {name}: {e}')
        else:
            failed_models[name] = cls

    return models, failed_models

models, failed_models = build_model_registry()

logging.info(f'{len(models)} models loaded')
for x in models.keys():
    logging.info(f'LOADED: {x}')

logging.info('---')
logging.info(f'---{len(failed_models)} models failed to load')
for x in failed_models.keys():
    logging.info(f'FAILED: {x}')
def failing(df, threshold=70):
    df['atrisk'] = df['Total'] < threshold
    return df

df1_sel = failing(df1_sel)
df2_sel = failing(df2_sel)
import math

def find_nrows(n_models, n_cols=4):
    return math.ceil(n_models / n_cols)

n_models = len(models)

results_bin = {}
results_reg = {}

empty_results_bin = {
    'x': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'pr_auc': [],
}

empty_results_reg = {
    'x': [],
    'rmse': [],
    'mae': [],
    'r2': [],
}

models_bin = {k: v for k, v in models.items() if v['binary']}
models_reg = {k: v for k, v in models.items() if not v['binary']}

errors_reg = ['rmse', 'mae', 'r2']
errors_bin = ['accuracy', 'precision', 'recall', 'f1', 'pr_auc']

print(f"Number of models: {n_models}")
print(f"Number of binary models: {len(models_bin)}")
print(f"Number of regression models: {len(models_reg)}")
l1 = len([x for x in df1_sel.columns.tolist() if 'Reading' in x])
l2 = len([x for x in df2_sel.columns.tolist() if 'Reading' in x])

print(f"Number of Reading quizzes in df1_sel: {l1}")
print(f"Number of Reading quizzes in df2_sel: {l2}")
import warnings
import logging

warnings.filterwarnings("ignore")

def get_results_dict(models, error_fn, empty_results, df, response_col):
    results = {}

    rq_cols = get_reg_cols(df, 'Reading')
    logging.info(f"Found {len(rq_cols)} RQ columns")

    for model_name, info in models.items():
        logging.info(f"Evaluating {model_name}...")
        
        results[model_name] = {k: [] for k in empty_results.keys()}
        results[model_name]['x'] = []

        for k in range(1, len(rq_cols) + 1):
            X = df[rq_cols[:k]]
            y = df[response_col]

            try:
                per_k_results = error_fn(
                    clf=info['model'],
                    X=X,
                    y=y
                )

            except Exception:
                logging.exception(
                    f"Error evaluating {model_name} with {k} RQs "
                    f"(X shape={X.shape}, y shape={y.shape})"
                )
                continue

            logging.debug(
                f"{model_name} | k={k} | results={per_k_results}"
            )

            for key, value in per_k_results.items():
                results[model_name][key].append(value)

            results[model_name]['x'].append(k)

    logging.info("Finished evaluating all models")
    return results
import pickle

    
    
    
    
    
with open('out/pickles/results1_bin.pkl', 'rb') as f:
    results1_bin = pickle.load(f)

with open('out/pickles/results1_reg.pkl', 'rb') as f:
    results1_reg = pickle.load(f)

with open('out/pickles/results2_bin.pkl', 'rb') as f:
    results2_bin = pickle.load(f)

with open('out/pickles/results2_reg.pkl', 'rb') as f:
    results2_reg = pickle.load(f)

def make_metric_dfs(d, add=None):

    """ Transforms the nested results dictionary into a dictionary of DataFrames,
        where each DataFrame corresponds to a metric and has columns for 'x' and the metric values for each classifier.
        If 'add' is provided, it will append to the existing DataFrames instead of creating new ones.
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
            safe_name = re.sub(r'[^\w\-. ]', '_', str(name)).strip()
            filename = f"{prefix + '_' if prefix else ''}{safe_name}.csv"
            path = os.path.join(out_dir, filename)
            df.to_csv(path, index=index)
        else:
            print(f"Skipping {name!r}: not a pandas DataFrame.")
            
def get_subplot_inds(ncols, k):
    
    i = k // ncols
    j = k % ncols
    return i, j
def plot_errors1(results, errors, nrows, ncols, scale):
    """ Simple error plotting function that takes in 
        one results dictionary and plots each metric 
        in a separate subplot.
        in: results, errors, nrows, ncols, scale
        scale is kind of broken because of the fat legend, 
        but it controls the overall size of the figure.
    """
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(scale*ncols, scale*nrows))

    for model, info in results.items():
        
        for i, metric in enumerate(errors):
            
            r, c = get_subplot_inds(ncols, i)
            
            axes[r][c].plot(info['x'], info[metric], marker='o', label=model)
            axes[r][c].set_xlabel('# of RQs Used')
            axes[r][c].set_ylabel(metric)
            
    fig.suptitle('Model Performance Comparison')
    fig.set_tight_layout(True)
    return fig

nrows, ncols = 2, 3
scale = 6

plt.clf()
errors_reg = ['rmse', 'mae', 'r2']
nrows, ncols = 2, 2
scale = 5
f = plot_errors1(results1_reg, errors_reg, nrows, ncols, scale)

import mplcursors
import matplotlib
plt.clf()

def plot_errors(results1, results2, errors, nrows, ncols, scale, 
                ylim=None, threshold=None, 
                thresh_direction=None,
                limit_models_to=None,
                fontsize=5,
                figsize=(10,30)):
    """ Enhanced error plotting function that can take in two results dictionaries
        and plot them on the same axes for comparison.
        It also allows for optional y-axis limits and threshold-based filtering of models."""
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
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
                    
                line, = axes[i].plot(x, y, 
                                marker=marker,
                                label=model, 
                                linestyle=line_type, 
                                color=colors[model])
                axes[i].set_xlabel('# of RQs Used')
                axes[i].set_ylabel(metric)
                
                if ylim and metric in ylim:
                    axes[i].set_ylim(ylim[metric])
                    
                line._hover_model = model
                line._hover_metric = metric
                line._hover_dataset = marker
                
                plotted_lines.append(line)
                    
        return axes
    axes = add_errors_to_plot(results1, line_type='-', marker='o')
    axes = add_errors_to_plot(results2, line_type='--', marker='x')
    
    def add_legend(axes, fontsize=fontsize):
        legend_ax = axes[-1]
        handles, labels = axes[0].get_legend_handles_labels()
        print(handles, labels)

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
        
        for text, label in zip(legend.get_texts(), labels):
            if label in ['Spring 2025', 'Fall 2025']:
                text.set_weight('bold')
                
        return axes
    axes = add_legend(axes)
    axes[-1].grid(False)  # remove grid from legend subplot
    axes[-1].axis('off')
            
    fig.suptitle('Model Performance Comparison')
    fig.set_tight_layout(True)
    

    
    return fig

binary_errors = ['accuracy', 'precision', 'recall', 'f1', 'pr_auc']
nrows, ncols = 6,1
scale = 6
plt.clf()

f = plot_errors(results1_bin, results2_bin, 
                binary_errors, nrows, ncols, scale,
                ylim=None,
                figsize=(6,20),
)
plt.tight_layout()
plt.show()

f.savefig('out/model_comparison_binary_full_long.svg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.lines import Line2D

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

                ax.set_ylabel(ylabs[i], fontsize=fontsize+3)
                if i == len(errors) - 1:
                    ax.set_xlabel("Number of RQs Used", fontsize=fontsize+3)
                else:
                    ax.set_xlabel("")

                if ylim and metric in ylim:
                    ax.set_ylim(ylim[metric])

    add_errors_to_plot(results1, line_type="-", marker="o", dataset_label="Q1")
    add_errors_to_plot(results2, line_type="--", marker="x", dataset_label="Q2")

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

binary_errors = ["pr_auc",  "precision", "f1", "recall", "accuracy" ]

plt.clf()

f = plot_errors_gs(
    results1_bin,
    results2_bin,
    ylim=None,
    figsize=(17, 17),
    fontsize=15,
    width_ratios=[5, 1, 1]
)

plt.tight_layout()
plt.savefig("out/model_comparison_binary_full_long_gs.svg")
plt.show()

def plot_errors(results1, results2, errors, nrows, ncols, scale, 
                ylim=None, threshold=None, 
                thresh_direction=None,
                limit_models_to=None,
                fontsize=5):
    """ Enhanced error plotting function that can take in two results dictionaries
        and plot them on the same axes for comparison.
        It also allows for optional y-axis limits and threshold-based filtering of models."""
    
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
                    
                line, = axes[i].plot(x, y, 
                                marker=marker,
                                label=model, 
                                linestyle=line_type, 
                                color=colors[model])
                axes[i].set_xlabel('# of RQs Used')
                axes[i].set_ylabel(metric)
                
                if ylim and metric in ylim:
                    axes[i].set_ylim(ylim[metric])
                    
                line._hover_model = model
                line._hover_metric = metric
                line._hover_dataset = marker
                
                plotted_lines.append(line)
                    
        return axes
    axes = add_errors_to_plot(results1, line_type='-', marker='o')
    axes = add_errors_to_plot(results2, line_type='--', marker='x')
    
    def add_legend(axes, fontsize=fontsize):
        legend_ax = axes[-1]
        handles, labels = axes[0].get_legend_handles_labels()
        print(handles, labels)

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
        
        for text, label in zip(legend.get_texts(), labels):
            if label in ['Spring 2025', 'Fall 2025']:
                text.set_weight('bold')
                
        return axes
    axes = add_legend(axes)
    axes[-1].grid(False)  # remove grid from legend subplot
    axes[-1].axis('off')
            
    fig.suptitle('Model Performance Comparison')
    fig.set_tight_layout(True)
    

    
    return fig

binary_errors = ['accuracy', 'precision', 'recall', 'f1', 'pr_auc']
nrows, ncols = 6,1
scale = 6
plt.clf()

f = plot_errors(results1_bin, results2_bin, 
                binary_errors, nrows, ncols, scale,
                ylim=None,
)

binary_errors = ['accuracy', 'precision', 'recall', 'f1', 'pr_auc']
nrows, ncols = 2,2
scale = 6
plt.clf()

f = plot_errors(results1_reg, results2_reg, 
                errors_reg, nrows, ncols, scale,
                ylim={'rmse': (0, 17), 'mae': (0, 20), 'r2': (-.05, 1)},
)
f.savefig('out/model_comparison_regression_full.svg')
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
    
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=higher_is_better)
    
    return sorted_models[:top_k]

def rank_df_based_on_metric(
    quarter="Q1",
    metric="pr_auc",
    rqs=13,
):
    results = results1_bin if quarter == "Q1" else results2_bin
    ranked_df = []

    for model, info in results.items():
        try:
            metric_at_k = info[metric][rqs - 1]  # rqs-1 because of 0-indexing
        except IndexError:
            print(f"{model}: Not enough data points to get {metric} at {rqs} RQs")
            metric_at_k = 0
        ranked_df.append({'model': model, f'{metric}_at_{rqs}_RQs': metric_at_k})
    
    ranked_df = pd.DataFrame(ranked_df).sort_values(by=f'{metric}_at_{rqs}_RQs', ascending=False)
    return ranked_df

def filter_models_by_threshold(df, limits):
    """ Filters the DataFrame of models based on specified metric thresholds."""
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'pr_auc']
    models = list(results1_bin.keys())

    dfs = {}
    for metric in metrics:
        dfs[metric] = rank_df_based_on_metric(metric=metric)
    full_df = dfs[metrics[0]]
    for i in range(1, len(metrics)):
        full_df = full_df.merge(dfs[metrics[i]], on='model')
        
    for metric, threshold in limits.items():
        df = df[df[f'{metric}_at_13_RQs'] >= threshold]

    return df

limits = {
    'accuracy': 0,#0.5,
    'precision': 0,#.5,
    'recall': 0,#.5,
    'f1': 0,#.5,
    'pr_auc': .65
}

ranked_df = rank_df_based_on_metric()
pr_auc_top = ranked_df['model'].to_list()[:5]

f = plot_errors_gs(
    results1_bin,
    results2_bin,
    ylim=None,
    figsize=(17, 17),
    fontsize=15,
    width_ratios=[5, 1, 1],
    limit_models_to=pr_auc_top,
)

f.savefig('out/model_comparison/binary_top_pr_auc_13_rqs.png', dpi=300)

metrics = ['accuracy', 'precision', 'recall', 'f1', 'pr_auc']
metrics = ['accuracy', 'precision', 'recall', 'f1', 'pr_auc']
models = list(results1_bin.keys())

dfs = {}
for metric in metrics:
    dfs[metric] = rank_df_based_on_metric(metric=metric)
full_df = dfs[metrics[0]]
for i in range(1, len(metrics)):
    full_df = full_df.merge(dfs[metrics[i]], on='model')
    
LIMITS = {
    'accuracy': .5,
    'precision': .5,
    'recall': .5,
    'f1': .5,
    'pr_auc': .6
}

def filter_models_by_threshold(df, limits):
    for metric, threshold in limits.items():
        df = df[df[f'{metric}_at_13_RQs'] >= threshold]
    return df

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
def save_top_models_to_csv(df, fname, trim_to_n_decimals=2, sep='\t'):
    if trim_to_n_decimals==2:
        df.to_csv(fname, float_format="%.2f", index=False, sep=sep) 
    elif trim_to_n_decimals==3:
        df.to_csv(fname, float_format="%.3f", index=False, sep=sep) 
    elif trim_to_n_decimals==4:
        df.to_csv(fname, float_format="%.4f", index=False, sep=sep)
    else:
        df.to_csv(fname, index=False, sep=sep)

cols = {
    'model':'Model', 
    'pr_auc_at_13_RQs':'PR-AUC',
    'f1_at_13_RQs':'F1', 
    'recall_at_13_RQs':'Recall', 
    'precision_at_13_RQs':'Precision', 
    'accuracy_at_13_RQs':'Acc', 
}

def rename_and_sort_cols(full_df, cols, 
        remove_nonzero=True,
        sort=True,
        replace_class=False,
        replace_da=False):
    """ Renames columns based on the provided mapping and optionally 
        filters and sorts the DataFrame.
    """
    
    if remove_nonzero:
        df = full_df[full_df['pr_auc_at_13_RQs'] >= .01]
    else:
        df = full_df.copy()
        
    if sort:
        df = df.sort_values(by='pr_auc_at_13_RQs', ascending=False)
        
    if replace_class:
        df['model'] = df['model'].str.replace('Classifier', '', regex=False)
        
    if replace_da:
        df['model'] = df['model'].str.replace('DiscriminantAnalysis', '\nDiscriminantAnalysis', regex=False)
        
    df = df[cols.keys()]
    df = df.rename(columns=cols)
    return df

def apply_rename_dict(df):
    
    model_dict = {
        "CategoricalNB": "CategoricalNB",
        "NearestCentroid": "NearestCentroid",
        "LinearDiscriminantAnalysis": "Linear DA",
        "AdaBoost": "AdaBoost",
        "BernoulliNB": "BernoulliNB",
        "LogisticRegression": "LogisticRegression",
        "GaussianProcess": "GaussianProcess",
        "LogisticRegressionCV": "Logistic RegressionCV",
        "CalibratedCV": "CalibratedCV",
        "GaussianNB": "GaussianNB",
        "RandomForest": "RandomForest",
        "QuadraticDiscriminantAnalysis": "Quadratic DA",
        "MLP": "MLP",
        "HistGradientBoosting": "HistGradientBoosting",
        "KNeighbors": "KNeighbors",
        "ExtraTrees": "ExtraTrees",
        "Bagging": "Bagging",
        "GradientBoosting": "GradientBoosting",
        "DecisionTree": "DecisionTree",
        "LabelSpreading": "LabelSpreading",
        "ExtraTree": "ExtraTree",
        "LabelPropagation": "LabelPropagation",
        "ComplementNB": "ComplementNB"
    }
    
    df['Model'] = df['Model'].map(model_dict).fillna(df['Model'])
    return df
df = rename_and_sort_cols(full_df, cols)
fname = 'out/model_comparison/model_ranking_full_at_13_RQs.csv'
save_top_models_to_csv(df, fname, trim_to_n_decimals=2)

zero_limits = {l: 0.1 for l in LIMITS.keys()}
df_by_prauc_no_zeros = filter_models_by_threshold(full_df, zero_limits)
df_by_prauc_sorted = rename_and_sort_cols(df_by_prauc_no_zeros, cols, remove_nonzero=True)
models_by_prauc = df_by_prauc_sorted['Model'].tolist()

limit_df = filter_models_by_threshold(full_df, LIMITS)
limit_df_renamed = rename_and_sort_cols(limit_df, cols, remove_nonzero=True)
models_by_limits = limit_df_renamed['Model'].tolist()

result_model_order = models_by_limits + [m for m in models_by_prauc if m not in models_by_limits]

full_df_sorted = full_df.set_index('model').loc[result_model_order].reset_index()
full_df_sorted = rename_and_sort_cols(full_df_sorted, cols, 
                                      remove_nonzero=True, 
                                      sort=False, 
                                      replace_class=True,
                                      replace_da=False)
full_df_sorted = apply_rename_dict(full_df_sorted)

fname = 'out/model_comparison/model_ranking_full_at_13_RQ_limits_then_prauc_sorted.csv'
save_top_models_to_csv(full_df_sorted, fname, trim_to_n_decimals=2)

full_df_sorted
vc = df1_sel['atrisk'].value_counts()
print(vc)
print(f"TN: {vc.iloc[0] / vc.sum()}\nTP: {vc.iloc[1] / vc.sum()}")

vc = df2_sel['atrisk'].value_counts()
print(vc)
print(f"TN: {vc.iloc[0] / vc.sum()}\nTP: {vc.iloc[1] / vc.sum()}")

fname = 'out/model_comparison/binary_top_multi_metric_13_rqs.csv'
cols = [
    'model', 
    'pr_auc_at_13_RQs',
    'precision_at_13_RQs', 
    'f1_at_13_RQs', 
    'recall_at_13_RQs', 
    'accuracy_at_13_RQs', 
]
filtered_df = filtered_df.sort_values(by='pr_auc_at_13_RQs', ascending=False)
filtered_df = filtered_df[cols]
save_top_models_to_csv(filtered_df, fname)
models_to_plot = []
for d in results1_reg, results2_reg:
    
    for error in errors_reg:
        
        ranked = rank_models(d, error, top_k=2, higher_is_better=True)
        models_to_plot.append(ranked)
        
print(models_to_plot)
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
metric = 'accuracy'
f = f'out/model_comparison/{metric}.csv'
res = pd.read_csv(f)

res

result_files = [
    'out/model_comparison/accuracy.csv',
    'out/model_comparison/precision.csv',
    'out/model_comparison/recall.csv',
    'out/model_comparison/f1.csv',
    'out/model_comparison/pr_auc.csv',
]

result_dict = {}
for f in result_files:
    metric = f.split('/')[-1].split('.')[0]
    df = pd.read_csv(f)

    for model in df.columns:
        if model == 'x':
            continue
        
        if model not in result_dict:
            result_dict[model] = {}
        result_dict[model][metric] = df[model].values
        
result_dict
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def check_normality(data, fname=""):
    data = np.asarray(data)

    print("=== Normality Checks ===")

    stat, p = stats.shapiro(data)
    print(f"Shapiro-Wilk p={p:.4g}")

    stat, p = stats.normaltest(data)
    print(f"D'Agostino K² p={p:.4g}")

    result = stats.anderson(data)
    print(f"Anderson-Darling stat={result.statistic:.4g}")
    for sl, cv in zip(result.significance_level, result.critical_values):
        print(f"  {sl}%: {cv}")

    plt.clf()

    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"QQ Plot {fname}")
    plt.savefig(f"out/stats/normality/norm_{fname}.png")
    plt.show()
    
for x in ['Preparation', 'Application', 'Examination', 'Total']:
    print(f"\n--- Checking normality for {x} ---")
    check_normality(df1_sel[x], fname=f"q1_{x}")
    check_normality(df2_sel[x], fname=f"q2_{x}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

n_students = 400

winter_prep = np.random.normal(90, 8, n_students)
winter_app = np.random.normal(93, 6, n_students)
winter_exam = np.random.normal(88, 10, n_students)

fall_prep = winter_prep + np.random.normal(3, 3, n_students)
fall_app = winter_app + np.random.normal(3, 3, n_students)
fall_exam = winter_exam + np.random.normal(-3, 3, n_students)

winter_prep = np.clip(winter_prep, 0, 100)
winter_app = np.clip(winter_app, 0, 100)
winter_exam = np.clip(winter_exam, 0, 100)

fall_prep = np.clip(fall_prep, 0, 100)
fall_app = np.clip(fall_app, 0, 100)
fall_exam = np.clip(fall_exam, 0, 100)

winter_overall = np.minimum.reduce([winter_prep, winter_app, winter_exam])
fall_overall = np.minimum.reduce([fall_prep, fall_app, fall_exam])

df = pd.DataFrame({
    "Quarter": ["Winter 2025"] * n_students + ["Fall 2025"] * n_students,
    "Preparation": np.concatenate([winter_prep, fall_prep]),
    "Application": np.concatenate([winter_app, fall_app]),
    "Examination": np.concatenate([winter_exam, fall_exam]),
    "Overall": np.concatenate([winter_overall, fall_overall])
})

summary = df.groupby("Quarter")[[
    "Preparation", "Application", "Examination", "Overall"
]].mean().T

summary["Fall minus Winter"] = (
    summary["Fall 2025"] - summary["Winter 2025"]
)

print(summary.round(2))

categories = ["Preparation", "Application", "Examination", "Overall"]
x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    summary.loc[categories, "Winter 2025"],
    width,
    label="Winter 2025"
)

plt.bar(
    x + width / 2,
    summary.loc[categories, "Fall 2025"],
    width,
    label="Fall 2025"
)

plt.xticks(x, categories, rotation=20, ha="right")
plt.ylabel("Mean grade")
plt.ylim(75, 100)
plt.title("Minimum grading scheme can mask category-level shifts")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("out/simulation/min_grading_masking.png", dpi=300)
plt.show()
import pandas as pd
from scipy.stats import chi2_contingency

df = df1_sel
cont_table = pd.crosstab(
    df['atrisk'],
    df['min_category']
)

print(cont_table)

chi2, p, dof, expected = chi2_contingency(cont_table)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p:.4e}")
print(f"Degrees of freedom: {dof}")

expected_df = pd.DataFrame(
    expected,
    index=cont_table.index,
    columns=cont_table.columns
)

print("\nExpected frequencies:")
print(expected_df)
course_order = [True, False]
labels = {
    True: "Failing students",
    False: "Passing students"
}
df = df1_sel
cat_order = ["Application", "Examination", "Preparation"]

counts = (
    df
    .groupby(["atrisk", "min_category"])
    .size()
    .unstack(fill_value=0)
    .reindex(course_order)
    .reindex(columns=cat_order)
)

props = counts.div(counts.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(6.2, 4.2))

bottom = np.zeros(len(props))

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

ax.set_title("Minimum Grade Category by Passing Status", fontsize=16, pad=12)
ax.set_ylabel("Proportion of students", fontsize=12)
ax.set_xlabel("")
ax.set_ylim(0, 1.08)

ax.set_xticks(x)
ax.set_xticklabels([labels[c] for c in props.index], fontsize=12)

ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

ax.text(
    0.5, -0.152,
    r"$\chi^2$ test: $p =  {p:.4e}$".format(p=p),
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=11
)

ax.legend(
    title="Minimum category",
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=11,
    title_fontsize=12,
)
ax.grid(False)

plt.tight_layout()
plt.savefig("out/min_category_by_atrisk.png", bbox_inches="tight", dpi=300)
plt.savefig("out/min_category_by_atrisk.svg", bbox_inches="tight")
plt.show()

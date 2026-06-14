
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from scipy.stats import gaussian_kde, ks_2samp, mannwhitneyu

DIR = "./grades/"
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

def get_score_col_name(df, exam=True):
    """ Return the name of the final grade column that is a percentage. 
        If exam=False, return the final score column instead.
    """
    cols = [x for x in df.columns if 'Final' in x]
    name = df[cols].sum().idxmin()
    if not exam:
        return 'Total'
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



def generate_plots(exam=False):
    
    for i in range(len(files)):
        print('Processing', files[i])
        ax = axes[i]
        file_name = files[i]
        title = find_title(file_name)
        df = get_df(file_name)

        inperson = get_inperson_section_name(df)
        remote = get_remote_section_name(df)
        score = get_score_col_name(df, exam=exam)

        # left and right are arrays of values for remote vs. in person
        left = df.loc[df["Section"] == inperson, score].dropna().to_numpy()
        right = df.loc[df["Section"] == remote, score].dropna().to_numpy()

        # If one of the sides is empty, skip
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

        # Get the kernel density estimates to draw the violin sides
        def kde_width_at(kde, y_grid, widths, y0):
            return float(np.interp(y0, y_grid, widths))

        # draw center line
        ax.axvline(0, color="black", linewidth=1)

        # ADD LEFT 
        if not skip_left:
            # Get the kernel density estimates to draw the violin sides
            kde_left = gaussian_kde(left)
            w_left = kde_left(y)
            ax.fill_betweenx(y, -w_left, 0, alpha=0.6, label='in person', color='green')
            
            # draw mode lines
            grid_l = np.linspace(left.min(), left.max(), 2000)  # dense grid helps
            dens_l = kde_left(grid_l)
            mode_est_l = grid_l[np.argmax(dens_l)]  # y-value where KDE is maximal
            wl = kde_width_at(kde_left, grid_l, dens_l, mode_est_l)
            ax.hlines(mode_est_l, xmin=-wl, xmax=0, linewidth=1.8, color="black")

        # ADD RIGHT
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
            # KS test annotation
            ks = ks_2samp(left, right, alternative="two-sided", mode="auto")
            ax.text(
                0.5, 0.98,
                f"KS test: D={ks.statistic:.3g}, p={ks.pvalue:.3g}",
                transform=ax.transAxes,
                ha="center", va="top"
            )
            
            # Mann-Whitney U test annotation
            u, p = mannwhitneyu(left, right, alternative="two-sided") 
            ax.text(
                0.5, 0.0,
                f"Mann-Whitney U: U={u:.3g}, p={p:.3g}",
                transform=ax.transAxes,
                ha="center", va="top"
            )

        ax.set_xticks([])
        ax.set_ylabel("value")
        if not exam:
            ax.set_ylim(40, 100)
        ax.set_title(title)
        ax.set_xlim(-max(w_left.max() if not skip_left else 0, w_right.max() if not skip_right else 0)*1.1,
                         max(w_left.max() if not skip_left else 0, w_right.max() if not skip_right else 0)*1.1)
        print('\tMODE inperson=\t', mode_est_l, '\tremote=\t\t', mode_est_r)
        print('\tMEAN inperson=\t', np.mean(left), '\tremote=\t\t', np.mean(right))
        print('\tMED inperson=\t', np.median(left), '\tremote=\t\t', np.median(right))
    plt.tight_layout()
    axes[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.02))
    return fig

fig, axes = plt.subplots(1, 8, figsize=(28, 4))
fig = generate_plots(exam=True)
fig.suptitle("Final Exam Distributions by Section", fontsize=16, y=1.03)
fig.savefig("out/final_exam_distributions.png", bbox_inches="tight", dpi=300)
plt.show()


plt.clf()
fig, axes = plt.subplots(1, 8, figsize=(28, 4))
fig = generate_plots(exam=False)
fig.suptitle("Final Grade Distributions by Section", fontsize=16, y=1.03)
fig.savefig("out/final_grade_distributions.png", bbox_inches="tight", dpi=300)
plt.show()

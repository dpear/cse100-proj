import re

import pandas as pd
from cse100proj.utils import load_config

config = load_config()

DIR = config['data']['input_dir']

def get_df(f):
    """ Read in the grades excel file, skip first row """
    df = pd.read_excel(DIR + f)  # Skip the first two rows of metadata
    df = df.iloc[1:] # Remove the first row ("points out of")
    return df


def find_title(f):
    """ If file is in the format XX-Grades_<course>_X_<quarter>.xlsx, 
        return <course>_<quarter> """
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
    

def get_reg_cols(df, s):
    """ Return the names of the columns that contain the string s. 
        For example, if s='Reading', 
        return the names of the columns that contain 'Reading'.
    """
    return [col for col in df.columns if s in col]

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


def select_and_rename(
    df, cols_to_keep=['quarter', 'year', 'course', 'exam_type',
                      'Preparation', 'Application', 'Examination', 
                      'Total', 'remote']):
    """ Selects columns from df that have '.1' in their name, 
        renames them by removing the ' (1)' suffix, 
        and returns the resulting dataframe.
        Also limits the columns to only those relevant for analysis.
    """
    renamed = {}
    df_cols = []
    new_df_cols = []
    for x in df.columns:
        if '.1' in x:
            renamed[x] = x.split(' (')[0]
            df_cols.append(x)
            new_df_cols.append(renamed[x])
        if x in cols_to_keep:
            df_cols.append(x)
            new_df_cols.append(x)

    df = df[df_cols]
    df = df.rename(columns=renamed)
    return df
    
def add_atrisk_column(df, threshold=70):
    """ Adds an 'atrisk' column to the DataFrame based on the
        'Total' column and a specified threshold.
    """
    if 'Total' not in df.columns:
        raise ValueError("The DataFrame must contain a 'Total' column.")
    df['atrisk'] = df['Total'] < threshold
    return df



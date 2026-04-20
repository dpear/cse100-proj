import os
from cse100proj.utils import load_config
from cse100proj.preprocessing import (
    get_df,
    add_atrisk_column,
    get_remote_section_name,
    select_and_rename,  
    find_title,
    
)

config = load_config()

from cse100proj.preprocessing import (
    get_df,
    add_atrisk_column,
    get_remote_section_name,
)
    

DIR = config['data']['input_dir']
files = os.listdir(DIR)
files = sorted(files)

assert 'Winter' in files[6]
assert 'Fall' in files[7]

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

# Fix duplicate column names
df1_sel = select_and_rename(df1) 
df2_sel = select_and_rename(df2)

# Which was the min grade category? = Overall grade
df1_sel['min_category'] = df1_sel[['Preparation', 'Application', 'Examination']].idxmin(axis=1)
df2_sel['min_category'] = df2_sel[['Preparation', 'Application', 'Examination']].idxmin(axis=1)

df1_sel = add_atrisk_column(df1_sel)
df2_sel = add_atrisk_column(df2_sel)

df1_sel.to_csv(config['data']['processed_dir'] + 'winter2025.csv', index=False)
df2_sel.to_csv(config['data']['processed_dir'] + 'fall2025.csv',   index=False)
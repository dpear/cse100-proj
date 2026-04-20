## CSE100 vs. CSE100R Analysis

This folder contains analyses for comparing CSE100 vs. CSE100R.
- `data`: Folder contains the raw and processed data
- `info`: Contains relevant info docs
- `out`: Outputs from analyses, figures, tables, and logs
- `scripts`: Scripts for running classification tasks etc.
- `src`: Reusable logic
<!-- - `notebooks`: Notebooks for running interactive plots -->

To run any of the scripts in this folder, first set up the conda environment:
```
conda env create -f config/environment.yml
```

Then activate and install:
```
conda activate cse100proj
pip install -e .
```

### TODO
- Fix logging
- Add grade category analysis
- Add interactive hover notebook for model comparison
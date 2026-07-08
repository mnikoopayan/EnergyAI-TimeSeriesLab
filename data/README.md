# Data Directory

`data/raw/` contains the seven CU-BEMS floor CSV files used in the study:

- `Floor1.csv`
- `Floor2.csv`
- `Floor3.csv`
- `Floor4.csv`
- `Floor5.csv`
- `Floor6.csv`
- `Floor7.csv`

The files are tracked with Git LFS because they are large. After cloning the
repository, run `git lfs pull` if the CSVs appear as small pointer files.

Primary modeling uses Floor 6 for the source benchmark and Floor 4 for the
few-shot transfer experiment. The remaining floors are retained for
missing-data diagnostics and dataset context.

Please cite the original CU-BEMS dataset paper when using these data:

Pipattanasomporn, M. et al. CU-BEMS, smart building electricity consumption and
indoor environmental sensor datasets. *Scientific Data* 7, 241 (2020).
https://doi.org/10.1038/s41597-020-00582-3

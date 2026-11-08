# UCI Wine source

Aeberhard, S. & Forina, M. (1992). Wine [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J

Dataset: https://archive.ics.uci.edu/dataset/109/wine
Original unchanged files:
- https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
- https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names

UCI licenses this dataset under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/legalcode
The authors and UCI do not endorse this implementation. Code is MIT; this license does not replace the source data license.

178 chemical analyses, 13 numeric features. The source first column identifies a cultivar. The loader strips it, returns only features, and does not use it for splitting, fitting, restart selection, or scoring. There is no classification or wine-quality claim. Training standardization and browser projection are derived transformations; original source bytes are unchanged. See dataset.py for SHA256 verification and artifacts/protocol.json for the label-free 106/36/36 split. This is a newly defined deterministic split, not an official UCI test set.

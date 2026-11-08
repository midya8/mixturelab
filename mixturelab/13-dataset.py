"""Public source, fixed label-free split, and training-only scale transform."""
from hashlib import sha256
from importlib.resources import files
import json
import numpy as np
from .core import matrix

FEATURES = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
DATA_HASH = '6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659'
NAMES_HASH = 'f1b84f2ef845e0bdebf13e14fa7a213e56de4f1baa40c5974dbd1ee51c5ae710'


def load():
    raw = files('mixturelab').joinpath('data/wine.data').read_bytes()
    names = files('mixturelab').joinpath('data/wine.names').read_bytes()
    if sha256(raw).hexdigest() != DATA_HASH or sha256(names).hexdigest() != NAMES_HASH:
        raise ValueError('public source hash mismatch')
    rows = np.array([[float(v) for v in line.split(',')[1:]] for line in raw.decode().splitlines() if line.strip()])
    if rows.shape != (178, 13):
        raise ValueError('unexpected source shape')
    # Class identifiers are deliberately not returned or used by any fit/split.
    return matrix(rows)


def protocol():
    return json.loads(files('mixturelab').joinpath('artifacts/protocol.json').read_text())


def split(x=None):
    x = load() if x is None else matrix(x)
    p = protocol()
    return {part: x[np.array(ids)] for part, ids in p['split_ids'].items()}


def scaler(train):
    train = matrix(train)
    mean, scale = train.mean(axis=0), train.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError('training feature has zero variance')
    return mean, scale

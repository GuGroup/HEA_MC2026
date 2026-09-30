# FeCoNiPdPt H adsorption: fcc111, fcc100, fcc110

## Layout

```
FeCoNiPdPt_H_adsorption/
  fcc111/
    fcc_hollow/{training,testing}/
    top/{training,testing}/
  fcc100/
    hollow/{training,testing}/
    bridge/{training,testing}/
  fcc110/
    bridge/{training,testing}/
```

fcc110/bridge is the original long-bridge site. Each training directory contains
training_data.json, train_model.py, and model.json. Each testing directory contains
test_data.json, evaluate_model.py, and generated results.

Every dataset contains only the indices listed in its original energy CSV, in CSV
order, with the H-adsorbed CONTCAR encoded as ASE Atoms and adsorption energy in eV.
All CSV rows are retained. Numeric indices are local to each facet/site/split and
must not be treated as globally unique. No energy shifts are applied.
Original feature ordering, geometry rules and no-intercept regression are preserved.

## Run

```bash
python -m pip install -r requirements.txt
python fcc100/hollow/training/train_model.py
python fcc100/bridge/training/train_model.py
python fcc110/bridge/training/train_model.py
python fcc111/fcc_hollow/training/train_model.py
python fcc111/top/training/train_model.py
python evaluate_all.py
```

The pinned dependencies were verified with Python 3.12.7. A site can be used on its
own: run its training/train_model.py and testing/evaluate_model.py. Scripts resolve
paths relative to themselves and work from another current directory.
Evaluation loads fixed models and does not fit on test data.

## Historical plots versus all-test evaluation

The original fcc100 hollow parity figure retains only test points with absolute
prediction error <= 0.1 eV. The original fcc110 bridge figure excludes the three
largest absolute errors (index ascending breaks ties). These error-selected subsets
are reproduced explicitly; they are not presented as complete-test performance.
All original CSV test structures remain in test_data.json. metrics.json separates
training, test_all, and test_original_plot; excluded indices and all predictions
are provided. parity_all_test.png includes every test point for the two filtered
sites, while parity_original.png matches the historical selection and styling.
Facet-level figures preserve the original (a)-(e) labels and panel arrangement.

## Read ASE structures

```python
from pathlib import Path
from ase.io.jsonio import decode
r = decode(Path('fcc100/hollow/training/training_data.json').read_text())['records'][0]
index, atoms, energy_eV = r['index'], r['atoms'], r['adsorption_energy_eV']
```

ASE Atoms preserve atomic order, coordinates, cell, PBC, and constraints. Provenance
and source hashes are embedded in the JSON. validation.json reports original model
and prediction comparisons, test selections, image comparisons and archive tests.

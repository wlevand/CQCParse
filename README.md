[![docs absent](https://img.shields.io/badge/docs-absent-red)](https://www.gnu.org/licenses/lgpl-3.0.html)
[![License LGPL 3.0](https://img.shields.io/badge/license-LGPL_v3.0-blue)](https://www.gnu.org/licenses/lgpl-3.0.html)
[![DOI:](https://img.shields.io/badge/DOI-set_it_up-red)](https://www.gnu.org/licenses/lgpl-3.0.html)

# CQCParse

CQCParse is a Python package for Parsing of Computational Quantum Chemistry (**CQC**) prorgram outputs.

## Key features

- Reads the results of anharmonic (VPT2) frequency jobs into one Python dict, with one function,
  `parse_from_source`:
  - Gaussian 16: the `.out` file of an `Opt` + `Freq=(Anharmonic,ReadAnharm)` job
  - CFOUR: the folder of an anharmonic run (`out`, `cubic`, `quartic`, `dipolex`/`y`/`z`, `polar.pkl`,
    `MOLDEN`). `polar.pkl` is not a CFOUR file: `CQCParse.tools.build_polar_pkl` builds it from separate
    polarizability runs at displaced geometries.
- Not general: made for the outputs of these job types (examples in `CQCParse/files_examples`).
- What it reads:
  - harmonic and anharmonic vibrational states (cm-1) and harmonic frequencies
  - rotational constants and Coriolis couplings
  - first and second derivatives of the dipole moment and of the polarizability (atomic units)
  - cubic and quartic force constants, in atomic units and in cm-1
  - equilibrium geometry, atoms and normal modes
- The same keys and shapes from both programs for the properties above (not yet for the geometry and the normal
  modes). Modes are numbered 0 .. n-1, and each tensor has every index order filled.
- Not yet the same mode order: Gaussian results come in Gaussian's anharmonic-section order (in the examples, the
  Mulliken numbering: by symmetry species, then descending frequency), CFOUR results and Gaussian's normal modes in
  ascending frequency. So mode k from the two programs can be a different vibration; see `parse_from_source`'s
  docstring.

## Installation

### From source
1. Get source code
2. Prepare the environment
3. Install

```
git clone git@github.com:wlevand/CQCParse.git
cd CQCParse
<make env>
pip install .
```

## Quick start

```python
from CQCParse.parsing import parse_from_source

data = parse_from_source(
    requested_data=dict.fromkeys(['nc_sqrt_eigval', 'anharmonic_states', 'cff', 'diphess']),
    source_type='gaussian',     # or 'cfour', with the folder of the run as base_file_loc
    base_file_loc='CQCParse/files_examples/dftGaussian/FORM/HFcc_pVTZ/g16_inputFull_3q.out',
    lvl_theory='HF', basis_set='cc-pVTZ')

data['nc_sqrt_eigval']      # harmonic frequencies, cm-1: {0: 3084.24, 1: 1999.176, 2: 1652.075, ...}
data['anharmonic_states']   # {('0',): 2956.163, ('1',): 1974.299, ...}, cm-1
data['cff'].shape           # (6, 6, 6): cubic force constants in atomic units; data['cff_rc'] in cm-1
data['diphess'].shape       # (6, 6, 3): mode, mode, then x, y, z
```


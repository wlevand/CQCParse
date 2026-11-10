[![docs absent](https://img.shields.io/badge/docs-absent-red)](https://www.gnu.org/licenses/lgpl-3.0.html)
[![License LGPL 3.0](https://img.shields.io/badge/license-LGPL_v3.0-blue)](https://www.gnu.org/licenses/lgpl-3.0.html)
[![DOI:](https://img.shields.io/badge/DOI-set_it_up-red)](https://www.gnu.org/licenses/lgpl-3.0.html)

# CQCParse

Python package for reading the results of anharmonic frequency jobs (VPT2, second-order vibrational perturbation
theory) from two Computational Quantum Chemistry (**CQC**) programs: Gaussian 16 and CFOUR.

## Key features

- One function, `parse_from_source`, with one Python dict as the result.
- The same keys, shapes and units from both programs, except `modes_mapping`.
- Modes numbered by ascending harmonic frequency in every key.
- An error instead of wrong or empty data, e.g. for a missing section or an imaginary frequency.

## Installation

Python 3.12 or newer.

```
git clone git@github.com:wlevand/CQCParse.git
cd CQCParse
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

## Quick start

From the repository root:

```python
from CQCParse.parsing import parse_from_source

data = parse_from_source(
    requested_data=['nc_sqrt_eigval', 'anharmonic_states', 'cff', 'diphess'],
    source_type='gaussian',
    base_file_loc='CQCParse/files_examples/dftGaussian/FORM/HFcc_pVTZ/g16_inputFull_3q.out',
    lvl_theory='HF', basis_set='cc-pVTZ')     # required arguments, no effect on the result

data['nc_sqrt_eigval'][0]          # 1337.151: harmonic frequency of mode 0 (the lowest), cm-1
data['anharmonic_states'][('0',)]  # 1320.251: anharmonic fundamental of mode 0, cm-1
data['cff'].shape                  # (6, 6, 6): cubic force constants in atomic units
data['diphess'].shape              # (6, 6, 3): mode, mode, then x, y, z
```

## Inputs

| Program | `source_type` | `base_file_loc` |
|---|---|---|
| Gaussian 16 | `'gaussian'` | `.out` file of an `Opt` + `Freq=(Anharmonic,ReadAnharm)` job |
| CFOUR | `'cfour'` | folder of an anharmonic run, with `out`, `cubic`, `quartic`, `dipolex`, `dipoley`, `dipolez`, `polar.pkl`, `MOLDEN` |

- No other job types. Examples: `CQCParse/files_examples`.
- `polar.pkl`: not a CFOUR file, but the output of `CQCParse.tools.build_polar_pkl`, from polarizability runs at
  displaced geometries.

## Returned data

Names for `requested_data`, with n the number of vibrational modes:

| Name | Content | Form | Unit |
|---|---|---|---|
| `nc_sqrt_eigval` | harmonic frequencies | `{mode: value}` | cm-1 |
| `harmonic_states`, `anharmonic_states` | state energies relative to the ground state | `{state label: value}` | cm-1 |
| `B` | rotational constants along x, y, z | `(3,)` | cm-1 |
| `coriolis` | Coriolis zeta, by axis, mode, mode | `(3, n, n)` | none |
| `dipgrad`, `diphess` | first and second derivatives of the dipole moment | `(n, 3)`, `(n, n, 3)` | atomic units |
| `polgrad`, `polhess` | first and second derivatives of the polarizability | `(n, 3, 3)`, `(n, n, 3, 3)` | atomic units |
| `cff`, `qff` | cubic and quartic force constants | `(n, n, n)`, `(n, n, n, n)` | atomic units |
| `cff_rc`, `qff_rc` | the same, by dimensionless normal coordinates | `(n, n, n)`, `(n, n, n, n)` | cm-1 |
| `equilibrium_geometry` | Cartesian coordinates, origin at the center of mass | `(atoms, 3)` | bohr |
| `atoms` | element symbols | list | none |
| `normal_modes` | displacement per unit normal coordinate, dx/dQ | `{mode: (atoms, 3) array}` | m_e^(-1/2) |
| `modes_mapping` | Gaussian only: Gaussian's mode numbers to the ascending ones, both from 1 | `{int: int}` | none |

- Mode numbers: 0 .. n-1.
- Derivatives in atomic units, `cff` and `qff`: by mass-weighted normal coordinates.
- State label: mode numbers as strings, one per quantum, in numeric order, e.g. `('0',)`, `('0', '0')`, `('2', '10')`.
- Details: docstring of `parse_from_source`.

## Cautions

No error in these cases:

- Mode k in two results, e.g. from two methods: not necessarily the same vibration. Two modes close in frequency
  possibly in swapped order.
- Gaussian force constants below Gaussian's print cut-off: absent from the output, so 0 in the result.
- `coriolis`: same value at (i, j) and (j, i), so usable as zeta^2 only.

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
    polarizability runs at displaced geometries. Without `out`, the reader takes `outfile0.out`; without
    `polar.pkl`, `pol/polar.pkl`.
- Not general: made for the outputs of these job types (examples in `CQCParse/files_examples`).
- What it reads:
  - harmonic and anharmonic vibrational states (cm-1) and harmonic frequencies
  - rotational constants and Coriolis couplings
  - first and second derivatives of the dipole moment and of the polarizability (atomic units)
  - cubic and quartic force constants, in atomic units and in cm-1
  - equilibrium geometry, atoms and normal modes
- The same keys and shapes from both programs for the properties above (not yet for the geometry and the normal
  modes). Modes are numbered 0 .. n-1, and each tensor has every index order filled.
- One mode order: both programs' results number the modes by ascending harmonic frequency, in every key. Gaussian
  numbers most of its output in another order; the reader renumbers it (`reindex_modes=False` keeps Gaussian's
  order). Two close modes can still swap between methods, so mode k from two results can be a different vibration;
  see `parse_from_source`'s docstring.
- The equilibrium geometry is in the frame of the normal modes, so the two can be drawn together.

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

data['nc_sqrt_eigval'][0]          # 1337.151: harmonic frequency of mode 0 (the lowest), cm-1
data['anharmonic_states'][('0',)]  # 1320.251: anharmonic fundamental of mode 0, cm-1
data['cff'].shape                  # (6, 6, 6): cubic force constants in atomic units; data['cff_rc'] in cm-1
data['diphess'].shape              # (6, 6, 3): mode, mode, then x, y, z
```

## Changes

What changed for code written against the earlier `main` (up to commit 4e16cf4).

Mode order:
- Both readers number the modes by ascending harmonic frequency, in every key. So Gaussian values come with
  other mode labels than before (Gaussian's `normal_modes` excepted). Example, formaldehyde HF/cc-pVTZ: mode 0
  was 3084.24 cm-1, now it is 1337.151 cm-1.
- `parse_from_source(..., reindex_modes=True)` is the new default. `reindex_modes=False` keeps Gaussian's own
  order in every key, `normal_modes` included. The result's `'reindex_modes'` tells which order you got. CFOUR
  needs no renumbering, so its `'reindex_modes'` is always `False`.
- `parse_gaussian16_output` takes `reindex_modes` too, and so do the validation readers (below).

Numbers that changed:
- CFOUR `qff` was too large: the reader divided CFOUR's values in cm-1 by amu^2 only. The factor per term was
  Eh^3 / (amu^2 sqrt(nu_i nu_j nu_k nu_l)), with Eh = 219474.63 cm-1, amu = 1822.888 electron masses and nu the
  harmonic wavenumbers (cm-1) of the term's modes: about 350 when all four modes are near 3000 cm-1. Fixed.
- Gaussian `cff` and `qff` are converted from the force constants in cm-1 (column FI of the output), no longer
  read from column K, which has only 5 decimals. Each term moved by about the rounding of column K.
- CFOUR `diphess`, `qff` and `qff_rc`: the files list some values twice, once per index order. The reader returns
  the mean of the two. Before, `diphess` was not symmetric, and for `qff` the later line won. Each value moved by
  half the gap between the two, which comes from the precision of CFOUR's numerical derivatives.
- Gaussian `normal_modes` have 5 decimals when the file has the table that `Freq=HPModes` prints (before:
  always 2 decimals). The values moved by up to 0.005.

Keys:
- A request for `cff` or `cff_rc` returns both. The same for `qff` and `qff_rc`.
- CFOUR results have the key `'reindex_modes'`, like Gaussian results.
- CFOUR returns `harmonic_states` only when the request has `harmonic_states` or `nc_sqrt_eigval`.

Errors instead of wrong or empty results (before: zeros, empty arrays, `None` or a missing key):
- An unknown name in `requested_data`, or an unknown `source_type`: `ValueError`.
- A section missing from a file: `ValueError` with the section's name.
- A section cut short, where the reader pairs two lists.
- Gaussian unit lines other than Debye or a.u.
- A Gaussian geometry not in the frame of the normal modes.
- A harmonic frequency that is not positive (an imaginary one; the geometry is not a minimum). Before, the data
  came with no error.
- CFOUR: `equilibrium_geometry` and `normal_modes` from a `MOLDEN` whose geometry is not the out file's reference
  structure (another geometry, or other axes).
- `CQCParse.tools.build_polar_pkl`: a reference polarizability that does not fit the displaced runs (from another
  calculation). Its error would go into every diagonal second derivative, divided by the step squared.
- A reader's error keeps its type, with a note that names the file.

Requests that failed before and work now:
- Gaussian: `B`, `coriolis` or a force constant without `nc_sqrt_eigval` in the same request.
- Gaussian: the states of an output without the anharmonic IR tables.
- Gaussian: the atoms F, S, Cl and Br.
- CFOUR: a run folder without `MOLDEN`, for every name but the geometry and the normal modes.
- CFOUR: a run folder with `outfile0.out` and `pol/polar.pkl` instead of `out` and `polar.pkl`.
- CFOUR: an out file whose levels table has values run together (a value that fills its whole fixed-width
  column has no blank before it).
- Gaussian: the geometry of the ACAC and ACDM examples, and the normal modes of the ACAC, ACDM and CO examples.
  The geometry is always the one of the frequencies, in the frame of the normal modes. For ACAC, the reader takes
  it from the anharmonic analysis: the last orientation block in that file is the start of an optimization.

Function calls:
- `parse_from_source`: `linear` is ignored and gives a `DeprecationWarning`. Both readers find out from the file
  whether the molecule is linear.
- `parse_cfour_output`: no `linear_molecule` argument.
- `CQCParse.logger.setup_logger`: no `use_rich` and `no_color` arguments.
- `get_cubic_post`, `get_quartic_post` (Gaussian) and `getCubicPost`, `getQuarticPost` (CFOUR): no `recipcm` and
  `reduced` arguments.
- `get_normal_modes` takes the lines of the file, not its name.
- The readers print nothing.
- Validation readers (for comparing with the programs' own VPT2 results): `get_detected_resonances_g16` returns
  (I, J, K) triples (before: `None`); `get_detected_resonances_c4` returns the resonance lines (before: `[]`);
  `parse_anharmonic_x_matrix` raises when its section is missing (before: empty lists).

Imports:
- `CQCParse.parsing` exports only `parse_from_source`. Import the other functions from their module, e.g.
  `from CQCParse.parsing.parseGaussian_forWilson import get_normal_modes`.
- `CQCParse` exports only `parsing`, `relay` and `utils`.

Removed:
- The class-based parsers, with the modules `parser_template`, `gaussian_parser` and `cfour_parser`:
  `GaussianParser`, `GaussianOutput`, `CFOURParser`, `CFOUROutput`, `GaussianDataParser`, `CFOURdataParser`,
  `ParsedData`, `DataStorage`, `Parser`, `OutputFiles`.
- The modules `CQCParse.debug`, `parsing.simple_parsed_data`, `parsing.mockparser`, `parsing.filesparser`,
  `parsing.parseGaussian_extra`, `parsing.parseCFOUR_extra`, `relay.relay_data` (with `DataVault`),
  `relay.retrieve_data` and `relay.wilson_data_obtainer`.
- Unused functions:
  - `parsing.parseGaussian_forWilson`: `get_allStates_fromParsedResults`, `GHz2Nu`, `nm_floats`,
    `nm_matrix_check`, `nm_parse_lines`, `normal_modes_prec`, `parse_frequencies`, `parse_frequencies_v2`,
    `parse_rot_const`, `remove_extra_g16`, `reordered_modes`
  - `parsing.parseCFOUR_forWilson`: `convNu2Ene`, `GHz2Nu`, `getDipoleDers`, `pklCubic`,
    `pklDimless_normal_modes`, `pklDipole`, `pklOutFile`
  - `utils`: `change_idx_modes`, `change_idx_states`, `convNu2Ene`, `make_modes_idx`
  - `relay.modes_reindexing`: `clean_noise`, `reindex_result_data`

Renamed or moved:
- `getDipoleDers_anharm_au_simple` -> `getDipoleDers_anharm_au`, `getPolarDers_pkl_au_simple` ->
  `getPolarDers_pkl_au` (CFOUR reader).
- The `polar.pkl` builder moved from the CFOUR reader to `CQCParse.tools.build_polar_pkl`: `pklPoldata`,
  `pklPolder`, `getPolarDers`, `getDisplacementsPolarData`, `getRotationMatrix`, `pTensor`.

New:
- `CQCParse.utils`: `reduced_to_au` and `dimensionless_to_mass_weighted` (the unit conversions, with worked
  examples in their docstrings), `HARTREE_CM`, `AMU_IN_ME`.
- `parsing.parse_wilson_obtainer`: `SOURCE_TYPES`, `KNOWN_NAMES`, `REINDEXED_TENSORS`, `REINDEXED_DICTS`,
  `is_linear`.
- `parsing.parseGaussian_forWilson`: `ECKART_TOLERANCE`.

Install:
- Needs only numpy, scipy and pandas. `pyproject.toml` is the only install file: `setup.py` and
  `requirements.txt` are gone.

Speed:
- Large Gaussian files load several times faster.

Example files (`CQCParse/files_examples`):
- New: formaldehyde HF/cc-pVTZ from both programs (`dftGaussian/FORM/HFcc_pVTZ`, `refinedc4/HFcc_pVTZ`), and
  CFOUR HF/cc-pVDZ (`refinedc4/HFcc_pVDZ`).
- Removed: `calculations.csv`, `polarData.pkl` in `refinedc4/CCSDTcc_pVQZ` and `refinedc4/HFcc_pVQZ`, and
  `dimensionless.pkl` in `refinedc4/CCSDTcc_pVQZ`. The readers do not need them.


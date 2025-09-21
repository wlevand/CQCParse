from . import get_allStates_fromParsedResults, get_equil_geo, reordered_modes, get_normal_modes, \
    parse_frequencies, getDipDers_au, getPolarDers_au, parse_cubic_constants, parse_quartic_constants, get_cubic_post, \
    get_quartic_post, parse_coriolis
from .parser_template import Parser, OutputFiles
from .parser_template import VPT2Data, DerivativesData, StatesData, NormalModesData, StructureData
from dataclasses import dataclass
from CQCParse.debug import debugfunc
import numpy as np

@dataclass
class GaussianOutput(OutputFiles):
    log_file: str = None

    def get_main_file(self):
        return self.log_file


class GaussianParser(Parser):

    def __init__(self, relevant_files: GaussianOutput = None):
        super().__init__(relevant_files)
        self.relevant_files: GaussianOutput = relevant_files


    def load(self):
        if self.relevant_files.log_file is not None:
            with open(self.relevant_files.log_file, 'r') as file:
                self._log_lines = [i.strip() for i in file.readlines()]


    def getStructure(self) -> StructureData:
        # number_atoms = parse_Na(self._log_lines)
        atoms, coords = get_equil_geo(self._log_lines)

        return StructureData(atoms, coords)


    def getNormModes(self) -> NormalModesData:
        modes = get_normal_modes(self.relevant_files.log_file, self.natoms)
        rmodes = reordered_modes(self.relevant_files.log_file)
        normal_modes = {}
        for i in rmodes:
            normal_modes[i-1] = modes[rmodes[i]-1]
       
        self.normal_modes = normal_modes

        return NormalModesData(normal_modes)


    def getVibStates(self) -> StatesData:

        results_log = parse_frequencies(self._log_lines)
        fundamentals_anharmonic_int = {int(k) - 1: float(v) for k, v in
                                            zip(results_log['Fundamental Bands']['mode_a'],
                                                results_log['Fundamental Bands'][2])}
        fundamentals_harmonic_int = {int(k) - 1: float(v) for k, v in
                                          zip(results_log['Fundamental Bands']['mode_a'],
                                              results_log['Fundamental Bands'][1])}

        fundamentals_harmonic_str = {str(k):v for k,v in fundamentals_harmonic_int.items()}
        fundamentals_anharmonic_str = {str(k):v for k,v in fundamentals_anharmonic_int.items()}
        ah_sts = get_allStates_fromParsedResults(results_log, anharmonic=True)
        h_sts = get_allStates_fromParsedResults(results_log, anharmonic=False)

        anharmonic_states = {tuple(str(i) for i in key): value for key, value in ah_sts.items()}
        harmonic_states = {tuple(str(i) for i in key): value for key, value in h_sts.items()}
        lenfund = len({k:v for k,v in anharmonic_states.items() if len(k)==1})
        lenfund_h = len({k:v for k,v in harmonic_states.items() if len(k)==1})
        debugfunc(f'Anharmonic states: {len(anharmonic_states)}. '
                  f'Fund.: {lenfund}. '
                  f'2quanta.: {len({k:v for k,v in anharmonic_states.items() if len(k)==2})}, should be {lenfund*(lenfund-1)//2+lenfund}. '
                  f'3quanta.: {len({k:v for k,v in anharmonic_states.items() if len(k)==3})}, should be {lenfund*(lenfund-1)*(lenfund-2)//6+lenfund+lenfund*(lenfund-1)}. ',
                  tag='parser.getVibStates')
        debugfunc(f'Harmonic states: {len(harmonic_states)}. '
                  f'Fund.: {lenfund_h}. '
                  f'2quanta.: {len({k: v for k, v in harmonic_states.items() if len(k) == 2})}, should be {lenfund*(lenfund-1)//2+lenfund}. '
                  f'3quanta.: {len({k: v for k, v in harmonic_states.items() if len(k) == 3})}, should be {lenfund*(lenfund-1)*(lenfund-2)//6+lenfund+lenfund*(lenfund-1)}. ',
                  tag='parser.getVibStates')

        self.nc_sqrt_eigval = fundamentals_harmonic_int
        self.anharmonic_states = anharmonic_states
        self.harmonic_states = harmonic_states

        return StatesData(fundamentals_harmonic_str, fundamentals_anharmonic_str,
                          fundamentals_harmonic_int, fundamentals_anharmonic_int, # not needed for Gaussian
                          harmonic_states, anharmonic_states)


    def getDerivatives(self) -> DerivativesData:

        mu = getDipDers_au(self._log_lines)
        alpha = getPolarDers_au(self._log_lines)

        cubic_df = parse_cubic_constants(self._log_lines)[0]
        cubic_rcm = cubic_df[['I', 'J', 'K', 'FI(I,J,K)']].to_numpy()
        cubic = cubic_df[['I', 'J', 'K', 'K(I,J,K)']].to_numpy()

        quartic_df = parse_quartic_constants(self._log_lines)[0]
        quartic_rcm = quartic_df[['I', 'J', 'K', 'L', 'FI(I,J,K,L)']].to_numpy()
        quartic = quartic_df[['I', 'J', 'K', 'L', 'K(I,J,K,L)']].to_numpy()

        cff_reduced = get_cubic_post(self.nmodes, cubic)
        quartic_force_constants_reduced = get_quartic_post(self.nmodes, quartic)

        cff = get_cubic_post(self.nmodes, cubic_rcm, reduced=False)
        quartic_cm_1 = get_quartic_post(self.nmodes, quartic_rcm, reduced=False)

        debugfunc(f'Dipole moment derivs: {len(mu), type(mu)} -> {type(mu[0])}: first - {mu[0].shape} ; second {mu[1].shape}', tag='parser.getDerivatives()')
        debugfunc(f'Polarizability derivs: {len(alpha), type(alpha)} -> {type(alpha[0])}: first - {alpha[0].shape}, second - {alpha[1].shape}', tag='parser.getDerivatives()')
        debugfunc(f'Cubic cm-1 {cff.shape}, has only zeros - {not np.any(cff)}, #non-zero elements {np.count_nonzero(cff)}', tag='parser.getDerivatives()')
        debugfunc(f'Quartic cm-1 {quartic_cm_1.shape}, has only zeros - {not np.any(quartic_cm_1)}, #non-zero elements {np.count_nonzero(quartic_cm_1)}', tag='parser.getDerivatives()')
        debugfunc(f'Cubic Ha {cff_reduced.shape}, has only zeros - {not np.any(cff_reduced)}, #non-zero elements {np.count_nonzero(cff_reduced)}', tag='parser.getDerivatives()')
        debugfunc(f'Quartic Ha {quartic_force_constants_reduced.shape}, has only zeros - {not np.any(quartic_force_constants_reduced)}, #non-zero elements {np.count_nonzero(quartic_force_constants_reduced)}', tag='parser.getDerivatives()')

        self.dipgrad = mu[0]
        self.diphess = mu[1]
        self.polgrad = alpha[0]
        self.polhess = alpha[1]
        self.cff = cff
        self.qff = quartic_cm_1

        return DerivativesData(mu[0], mu[1],
                               alpha[0], alpha[1],
                               cff_reduced, quartic_force_constants_reduced, None,
                               cff, quartic_cm_1)


    def getPreVPT2Data(self) -> VPT2Data:

        rotational_constant, coriolis_constant = parse_coriolis(self._log_lines, self.nmodes)

        DD11 = ('No 1-1 Darling-Dennison resonance found' not in self._log_lines
                    and 'Search for 1-1 Darling-Dennison resonances deactivated' not in self._log_lines)
        DD13 = ('No 1-3 Darling-Dennison resonance found' not in self._log_lines
                    and 'Search for 1-3 Darling-Dennison resonances deactivated' not in self._log_lines)
        DD22 = ('No 2-2 Darling-Dennison resonance found' not in self._log_lines
                    and 'Search for 2-2 Darling-Dennison resonances deactivated' not in self._log_lines)

        debugfunc(f'Rotational constant: {rotational_constant}; Coriolis constant: {coriolis_constant.shape}, has only zeros - {not np.any(coriolis_constant)}', tag='parser.getPreVPT2Data()')

        self.B = rotational_constant
        self.coriolis = coriolis_constant

        # rotational_constants, coriolis_constants, fermi_resonance, DD11, DD13, DD22
        return VPT2Data(rotational_constant, coriolis_constant, [], DD11, DD13, DD22)

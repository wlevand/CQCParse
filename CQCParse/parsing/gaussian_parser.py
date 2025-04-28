from . import get_allStates_fromParsedResults, get_equil_geo, reordered_modes, get_normal_modes, \
    parse_frequencies, getDipDers_au, getPolarDers_au, parse_cubic_constants, parse_quartic_constants, get_cubic_post, \
    get_quartic_post, parse_coriolis
from .parser_template import Parser, ParsedData, OutputFiles
from .parser_template import VPT2Data, DerivativesData, StatesData, NormalModesData, StructureData
from dataclasses import dataclass


@dataclass
class GaussianOutput(OutputFiles):
    log_file: str = None

    def get_main_file(self):
        return self.log_file


class GaussianParser(Parser):

    def __init__(self, relevant_files: GaussianOutput):
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

        cubic_force_constants = get_cubic_post(self.nmodes, cubic)
        quartic_force_constants = get_quartic_post(self.nmodes, quartic)

        cubic_cm_1 = get_cubic_post(self.nmodes, cubic_rcm, reduced=False)
        quartic_cm_1 = get_quartic_post(self.nmodes, quartic_rcm, reduced=False)

        return DerivativesData(mu[0], mu[1],
                               alpha[0], alpha[1],
                               cubic_force_constants, quartic_force_constants,
                               cubic_cm_1, quartic_cm_1)


    def getPreVPT2Data(self) -> VPT2Data:

        rotational_constant, coriolis_constant = parse_coriolis(self._log_lines, self.nmodes)

        DD11 = ('No 1-1 Darling-Dennison resonance found' not in self._log_lines
                    and 'Search for 1-1 Darling-Dennison resonances deactivated' not in self._log_lines)
        DD13 = ('No 1-3 Darling-Dennison resonance found' not in self._log_lines
                    and 'Search for 1-3 Darling-Dennison resonances deactivated' not in self._log_lines)
        DD22 = ('No 2-2 Darling-Dennison resonance found' not in self._log_lines
                    and 'Search for 2-2 Darling-Dennison resonances deactivated' not in self._log_lines)

        return VPT2Data(rotational_constant, coriolis_constant, [], DD11, DD13, DD22)

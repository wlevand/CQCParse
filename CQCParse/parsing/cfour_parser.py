from .parser_template import Parser, OutputFiles
from .parser_template import VPT2Data, DerivativesData, StatesData, NormalModesData, StructureData
from  dataclasses import dataclass
from .parseCFOUR_forWilson import (pMOLDEN, parse_output_file, parse_coriolis,
                                  getCubicPost, getQuarticPost, getDipoleDers_anharm_au,
                                  getPolarDers_pkl_au,
                                  pCubicORQuartic)

@dataclass
class CFOUROutput(OutputFiles):
    # out_file_main: str = None
    out_file: str = None

    molden_file: str = None
    cubic_file: str = None
    quartic_file: str = None
    dipole_file: str = None

    polar_pkl: str = None

    def get_main_file(self):
        return self.out_file


class CFOURParser(Parser):

    def __init__(self, relevant_files: CFOUROutput):
        super().__init__(relevant_files)
        self.relevant_files: CFOUROutput = relevant_files
        self._saved_data = {}


    def load(self):
        if self.relevant_files.out_file is not None:
            with open(self.relevant_files.out_file, 'r') as file:
                self._out_lines = [i.strip() for i in file.readlines()]


    def getStructure(self) -> StructureData:
        if 'coords' in self._saved_data:
            atoms = self._saved_data['atoms']
            coords = self._saved_data['coords']
        else:
            coords, atoms, normal_modes_dict = pMOLDEN(self.relevant_files.molden_file)
            normal_modes = {k - self.nModesStart: v.flatten() for k, v in normal_modes_dict.items() if k >= self.nModesStart}
            self._saved_data['coords'] = coords
            self._saved_data['atoms'] = atoms
            self._saved_data['normal_modes_dict'] = normal_modes
        return StructureData(atoms, coords)

    def getNormModes(self) -> NormalModesData:
        if 'normal_modes_dict' in self._saved_data:
            normal_modes = self._saved_data['normal_modes_dict']
        else:
            coords, atoms, normal_modes_dict = pMOLDEN(self.relevant_files.molden_file)
            # print(normal_modes_dict)
            normal_modes = {k - self.nModesStart: v.flatten() for k, v in normal_modes_dict.items() if k >= self.nModesStart}
            self._saved_data['coords'] = coords
            self._saved_data['atoms'] = atoms
            self._saved_data['normal_modes_dict'] = normal_modes
        # print(normal_modes)
        return NormalModesData(normal_modes)


    def getVibStates(self) -> StatesData:

        parsed_data = parse_output_file(self.relevant_files.out_file)
        vib_energy_levels_list, labelsTable, anharmonic_freqs, anharmonic_ints, harmonic_freqs = parsed_data

        anharm_states_dict = dict(zip(vib_energy_levels_list, anharmonic_freqs))
        anharm_tuple_dict = {tuple([o - self.nModesStart for o in k]): v for k, v in anharm_states_dict.items()}
        harm_states_dict = dict(zip(vib_energy_levels_list, harmonic_freqs))
        harm_tuple_dict = {tuple([o - self.nModesStart for o in k]): v for k, v in harm_states_dict.items()}

        fundamentals_harmonic_str = {str(k[0]):v for k,v in harm_tuple_dict.items() if len(k)==1}
        fundamentals_anharmonic_str = {str(k[0]):v for k,v in anharm_tuple_dict.items() if len(k)==1}
        self._saved_data['fundamentals_harmonic_str'] = fundamentals_harmonic_str

        fundamentals_harmonic_int = {int(k):v for k,v in fundamentals_harmonic_str.items()}
        fundamentals_anharmonic_int = {int(k):v for k,v in fundamentals_anharmonic_str.items()}
        self._saved_data['fundamentals_harmonic_int'] = fundamentals_harmonic_int

        anharmonic_states = {tuple(str(i-self.nModesStart) for i in k): v for k, v in anharm_states_dict.items()}
        harmonic_states = {tuple(str(i-self.nModesStart) for i in k): v for k, v in harm_states_dict.items()}

        return StatesData(fundamentals_harmonic_str, fundamentals_anharmonic_str,
                          fundamentals_harmonic_int, fundamentals_anharmonic_int,
                          harmonic_states, anharmonic_states)


    def getDerivatives(self) -> DerivativesData:

        cubic = pCubicORQuartic(self.relevant_files.cubic_file)
        quartic = pCubicORQuartic(self.relevant_files.quartic_file)

        cff = getCubicPost(self._saved_data['fundamentals_harmonic_int'], cubic,
                                                  startmode=self.nModesStart, recipcm=False)
        quartic_force_constants = getQuarticPost(self._saved_data['fundamentals_harmonic_int'], quartic,
                                                  startmode=self.nModesStart, recipcm=False)

        cubic_cm_1 = getCubicPost(self._saved_data['fundamentals_harmonic_int'], cubic,
                                                  startmode=self.nModesStart, recipcm=True)
        quartic_cm_1 = getQuarticPost(self._saved_data['fundamentals_harmonic_int'], quartic,
                                                  startmode=self.nModesStart, recipcm=True)

        labelsModes_original = [i + self.nModesStart for i in list(self._saved_data['fundamentals_harmonic_int'])]
        mu = getDipoleDers_anharm_au(self.relevant_files.dipole_file, labelsModes_original, self.nModesStart,
                                     self._saved_data['fundamentals_harmonic_str'])
        # dipgrad = mu[0]
        # diphess = mu[1]

        if self.relevant_files.polar_pkl is not None:
            alpha = getPolarDers_pkl_au(self.relevant_files.polar_pkl, self._saved_data['fundamentals_harmonic_str'])
            # polgrad = alpha[0]
            # polhess = alpha[1]
        else:
            print('Why no polarizability pickle?')

        return DerivativesData(mu[0], mu[1],
                               alpha[0], alpha[1],
                               cff, quartic_force_constants,
                               cubic_cm_1, quartic_cm_1)


    def getPreVPT2Data(self) -> VPT2Data:
        rotational_constant, coriolis_constant = parse_coriolis(self.relevant_files.out_file,
                                                                          self.nmodes,
                                                                          startmode=self.nModesStart)

        return VPT2Data(rotational_constant, coriolis_constant)



from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class OutputFiles(ABC):
    """
        Dataclass template: Collection of files with data for 2D IR spectrum

    GaussianOutput('FORM', 'B3LYP', 'cc_pVTZ', 'gaussian', log_file='')
    CFOUROutput('FORM', 'CCSDT', 'cc_pVQZ', 'gaussian', log_file='')
    """
    molecule: str
    method: str
    basis: str
    program: str

    @abstractmethod
    def get_main_file(self):
        """Would return the main file for processing."""
        pass


@dataclass
class StructureData:
    atoms: np.ndarray | list = field(default_factory=lambda: np.array([]))
    equilibrium_geometry: np.ndarray = field(default_factory=lambda: np.array([]))
    # nModesStart: int = 3*2-5


@dataclass
class DerivativesData:
    dipole_first_derivatives: np.ndarray = field(default_factory=lambda: np.array([]))
    dipole_second_derivatives: np.ndarray = field(default_factory=lambda: np.array([]))
    polarizability_first_derivatives: np.ndarray = field(default_factory=lambda: np.array([]))
    polarizability_second_derivatives: np.ndarray = field(default_factory=lambda: np.array([]))
    cubic_force_constants: np.ndarray = field(default_factory=lambda: np.array([]))
    quartic_constants: np.ndarray = field(default_factory=lambda: np.array([]))
    cubic_cm_1: np.ndarray = field(default_factory=lambda: np.array([]))
    quartic_cm_1: np.ndarray = field(default_factory=lambda: np.array([]))

    def upd_indices(self, new_idx_dict):
        from CQCParse.utils import change_idx_modes

        deriv_data  = change_idx_modes(self, new_idx_dict,
                                       data2transform='derivatives')
        self.dipole_first_derivatives = deriv_data['mu_Q']
        self.dipole_second_derivatives = deriv_data['mu_QQ']
        self.polarizability_first_derivatives = deriv_data['alpha_Q']
        self.polarizability_second_derivatives = deriv_data['alpha_QQ']
        self.cubic_force_constants = deriv_data['F_abc']

        self.cubic_cm_1, self.quartic_cm_1 = change_idx_modes(self, new_idx_dict,
                                                              data2transform='force_consts')


@dataclass
class StatesData:
    fundamentals_harmonic_str: Dict = field(default_factory=dict)
    fundamentals_anharmonic_str: Dict = field(default_factory=dict)
    fundamentals_harmonic_int: Dict = field(default_factory=dict)
    fundamentals_anharmonic_int: Dict = field(default_factory=dict)
    harmonic_states: Dict = field(default_factory=dict)
    anharmonic_states: Dict = field(default_factory=dict)
    vpt2_states_all: Dict = field(default_factory=dict)
    vpt2_states_fund: Dict = field(default_factory=dict)

    def upd_indices(self, new_idx_dict):
        from CQCParse.utils import change_idx_modes

        all_states, all_states_harmonic = change_idx_modes(self, new_idx_dict,
                                                           data2transform='states')
        self.anharmonic_states = all_states
        self.harmonic_states = all_states_harmonic
        # self.fundamentals_anharmonic_str = fundamentals
        # self.fundamentals_harmonic_str = fundamentals_harmonic
        self.fundamentals_anharmonic_str = {k[0]:v for k,v in all_states.items() if len(k)==1}
        self.fundamentals_harmonic_str = {k[0]:v for k,v in all_states_harmonic.items() if len(k)==1}


@dataclass
class VPT2Data:
    """Pre-VPT2 procedure - necessary data"""
    # cubic_cm_1: np.ndarray = np.array([])
    # quartic_cm_1: np.ndarray = np.array([])
    rotational_constants: np.ndarray = field(default_factory=lambda: np.array([]))
    coriolis_constants: np.ndarray = field(default_factory=lambda: np.array([]))
    fermi_resonance: list = field(default_factory=lambda: [])
    DD11: str = "'1-1 Darling-Dennison resonance weren't parsed"
    DD13: str = "'1-3 Darling-Dennison resonance weren't parsed"
    DD22: str = "'2-2 Darling-Dennison resonance weren't parsed"

    def upd_indices(self, new_idx_dict):
        from CQCParse.utils import change_idx_modes

        self.coriolis_constants = change_idx_modes(self, new_idx_dict,
                                                   data2transform='coriolis')
        self.fermi_resonance = change_idx_modes(self, new_idx_dict,
                                                   data2transform='fermi_resonance')

@dataclass
class NormalModesData:
    # nmodes: int
    normal_modes: Dict = field(default_factory=dict)
    # Q_normal_coordinates: np.ndarray
    # q_normal_coordinates_dimensionless: np.ndarray


@dataclass
class ParsedData:
    """
    Storage of parsed data - for a single calculation instance
    """
    # identifiers
    molecule: str
    program: str
    basis: str
    method: str

    structure: StructureData = field(default_factory=lambda: StructureData())
    nmodes: int = 0
    vib_states: StatesData = field(default_factory=lambda: StatesData())
    derivatives: DerivativesData = field(default_factory=lambda: DerivativesData())
    normal_modes: NormalModesData = field(default_factory=lambda: NormalModesData())
    anharm_correction_data: VPT2Data = field(default_factory=lambda: VPT2Data())
    anharm_treatment: str = 'original'
    list2exclude: list = field(default_factory=lambda: list)

    def get_vpt2(self, vpt2settings, list2exclude=None, print_level=0):
        if list2exclude is None:
            list2exclude = []

        # if vpt2settings is not None:
        from wilson.spectrum.vpt2 import get_vpt2_corrected_levels
        all_states, fermi_resonance = get_vpt2_corrected_levels(self, vpt2settings,
                                               list2exclude,
                                               print_level=print_level)
        self.vib_states.fundamentals_anharmonic_str = {k[0]: v for k, v in all_states.items() if len(k)==1}
        self.vib_states.anharmonic_states = all_states
        self.vib_states.vpt2_states_all = all_states
        self.vib_states.vpt2_states_fund = {k[0]: v for k, v in all_states.items() if len(k)==1}
        self.anharm_treatment = 'homebrew'
        self.list2exclude = list2exclude
        self.anharm_correction_data.fermi_resonance = fermi_resonance

    def __repr__(self):
        return f"<ParsedData: {self.__dict__.keys()}"

import pickle
class DataStorage:
    """
    Saves and tracks multiple ParsedData instances.
    """
    _instances = {}

    @classmethod
    def save(cls, molecule, basis, method, program, enelvls, instance):
        if (molecule, program, basis, method) in cls._instances:
            raise ValueError(f"Instance '{molecule, program, basis, method}' already exists!")
        cls._instances[(molecule, basis, method, program, enelvls)] = instance
        return instance

    @classmethod
    def get(cls, name_tuple):
        return cls._instances.get(name_tuple)

    @classmethod
    def save_to_file(cls, filename):
        """Save the _instances dictionary to a file."""
        with open(filename, 'wb') as file:
            pickle.dump(cls._instances, file)
        print(f"DataStorage saved to {filename}")

    @classmethod
    def load_from_file(cls, filename):
        """Load the _instances dictionary from a file."""
        try:
            with open(filename, 'rb') as file:
                cls._instances = pickle.load(file)
            print(f"DataStorage loaded from {filename}")
        except FileNotFoundError:
            print(f"No existing file found at {filename}. Starting with an empty storage.")
            cls._instances = {}

    @classmethod
    def append_to_file(cls, filename, molecule, basis, method, program, enelvls, instance):
        """Append new data to the storage saved in a file."""
        cls.load_from_file(filename)

        cls.save(molecule, basis, method, program, enelvls, instance)

        cls.save_to_file(filename)

    @classmethod
    def initialize_empty_storage(cls, filename):
        """Initialize an empty storage and save it to a file."""
        cls._instances = {}

    @classmethod
    def list_instances(cls):
        return list(cls._instances.keys())

    @classmethod
    def get_allstates(cls):

        allstates = {}
        for i, d in enumerate(cls._instances):
            dd = cls._instances[d]
            allstates[f'{' '.join(d)} {dd.anharm_treatment}'] = dd.vib_states.anharmonic_states
        return allstates

    @classmethod
    def get_dmu(cls):
        axes = {0:'x', 1:'y', 2:'z'}
        dipolfirstders = {}

        for i, d in enumerate(cls._instances):
            dd = cls._instances[d]
            dmu = dd.derivatives.dipole_first_derivatives
            currdict = {}
            for i in range(dmu.shape[0]):
                for j in range(dmu.shape[1]):
                    currdict[f'Q{i} {axes[j]}'] = dmu[i,j]
            dipolfirstders[f'{' '.join(d)}'] = currdict

        return dipolfirstders

    @classmethod
    def get_d2mu(cls):
        axes = {0:'x', 1:'y', 2:'z'}
        dipolsecders = {}

        for i, d in enumerate(cls._instances):
            dd = cls._instances[d]
            dmu = dd.derivatives.dipole_second_derivatives
            currdict = {}
            for i in range(dmu.shape[0]):
                for j in range(dmu.shape[1]):
                    for k in range(dmu.shape[2]):
                        currdict[f'Q{i}Q{j} {axes[k]}'] = dmu[i,j,k]
            dipolsecders[f'{' '.join(d)}'] = currdict

        return dipolsecders

    @classmethod
    def get_dalpha(cls):
        axes = {0:'x', 1:'y', 2:'z'}
        polarfirstders = {}

        for i, d in enumerate(cls._instances):
            dd = cls._instances[d]
            dalpha = dd.derivatives.polarizability_first_derivatives
            currdict = {}
            for i in range(dalpha.shape[0]):
                for j in range(dalpha.shape[1]):
                    for k in range(dalpha.shape[2]):
                        currdict[f'Q{i} {axes[j]}{axes[k]}'] = dalpha[i,j,k]
            polarfirstders[f'{' '.join(d)}'] = currdict

        return polarfirstders

    @classmethod
    def get_d2alpha(cls):
        axes = {0:'x', 1:'y', 2:'z'}
        polarsecders = {}

        for i, d in enumerate(cls._instances):
            dd = cls._instances[d]
            dalpha = dd.derivatives.polarizability_second_derivatives
            currdict = {}
            for i in range(dalpha.shape[0]):
                for j in range(dalpha.shape[1]):
                    for k in range(dalpha.shape[2]):
                        for l in range(dalpha.shape[3]):
                            currdict[f'Q{i}Q{j} {axes[k]}{axes[l]}'] = dalpha[i,j,k,l]
            polarsecders[f'{' '.join(d)}'] = currdict

        return polarsecders

    @classmethod
    def get_CFF(cls):
        cff = {}

        for i, d in enumerate(cls._instances):
            dd = cls._instances[d]
            dalpha = dd.derivatives.cubic_force_constants
            currdict = {}
            for i in range(dalpha.shape[0]):
                for j in range(dalpha.shape[1]):
                    for k in range(dalpha.shape[2]):
                        currdict[f'Q{i} Q{j} Q{k}'] = dalpha[i,j,k]
            cff[f'{' '.join(d)}'] = currdict

        return cff


class Parser(ABC):
    """
        Functional class template.

    Load - parse - store (in a dataclass)
    """
    def __init__(self, relevant_files: OutputFiles):

        self.relevant_files = relevant_files
        self.natoms = None
        self.nmodes = None


    @abstractmethod
    def load(self):
        pass


    @abstractmethod
    def getStructure(self) -> StructureData:
        pass

    @abstractmethod
    def getNormModes(self) -> NormalModesData:
        pass

    @abstractmethod
    def getVibStates(self) -> StatesData:
        pass

    @abstractmethod
    def getDerivatives(self) -> DerivativesData:
        pass

    @abstractmethod
    def getPreVPT2Data(self) -> VPT2Data:
        pass


    def parse(self, linear_molecule: bool = False) -> ParsedData:
        """
        Parse and keep
        """
        # for cfour data
        self.linear_molecule = linear_molecule
        self.nModesStart = 6 if self.linear_molecule else 7

        structure_data = self.getStructure()

        x = 5 if linear_molecule else 6
        self.nmodes = 3 * len(structure_data.atoms) - x
        self.natoms = len(structure_data.atoms)

        norm_modes = self.getNormModes()
        vib_states = self.getVibStates()
        derivs = self.getDerivatives()
        anharm_correction_data = self.getPreVPT2Data()

        return ParsedData(self.relevant_files.molecule,
                          self.relevant_files.program,
                          self.relevant_files.basis,
                          self.relevant_files.method,

                          structure_data, self.nmodes, vib_states,
                          derivs, norm_modes, anharm_correction_data)

    # def get_vpt2(self, parsedData, vpt2settings, list2exclude=None, print_level=0):
    #     if list2exclude is None:
    #         list2exclude = []
    #
    #     # if vpt2settings is not None:
    #     from wilson.spectrum.vpt2 import get_vpt2_corrected_levels
    #     all_states = get_vpt2_corrected_levels(parsedData, vpt2settings,
    #                                            list2exclude,
    #                                            print_level=print_level)
    #     parsedData.vib_states.fundamentals_anharmonic_str = {k[0]: v for k, v in all_states.items() if len(k)==1}
    #     parsedData.vib_states.anharmonic_states = all_states
    #     parsedData.vib_states.vpt2_states_all = all_states
    #     parsedData.vib_states.vpt2_states_fund = {k[0]: v for k, v in all_states.items() if len(k)==1}

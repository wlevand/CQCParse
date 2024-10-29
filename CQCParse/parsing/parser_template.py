
class MockParser:

    def __init__(self, relevant_files: dict):
        self.relevant_files = relevant_files

        # if relevant for parsing
        self.nModesStart = None

        self.dipole_first_derivatives = None
        self.dipole_second_derivatives = None
        self.polarizability_first_derivatives = None
        self.polarizability_second_derivatives = None

        self.fundamentals_harmonic_str = None
        self.fundamentals_anharmonic_str = None
        self.harmonic_states = None
        self.anharmonic_states = None
        self.cubic_force_constants = None
        self.quartic_constants = None

        self.equilibrium_geometry = None
        self.Q_normal_coordinates = None
        self.q_normal_coordinates_dimensionless = None

        self.atoms = None
        self.basis = None
        self.lot = None

    def getData(self, linear_molecule: bool = False):
        """Set attributes here"""
        # if relevant for parsing
        self.nModesStart = 6 if linear_molecule else 7

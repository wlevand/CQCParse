"""
#################################################################################################
##                                                                                             ##
##                             Parsing Gaussian output files                                   ##
##                                                                                             ##
#################################################################################################
#   Some other parsing methods that are not used now in Wilson calculations.
# Files:
#     - .log  --- the main full output file that contains all the relevant data
#     - .fchk --- formcheck (generated from checkpoint file)
"""

import numpy as np
# np.set_printoptions(linewidth=250, suppress=True, precision=3)
import sys
import pandas as pd
pd.set_option('display.max_rows', sys.maxsize)

# not used now
def getForceConstants_fchk(fchkfile: str):

    with open(fchkfile, 'r') as file:
        file_content = file.read()

    start_index = file_content.find("Cartesian Force Constants")

    end_index = file_content.find("Cartesian 3rd/4th derivatives", start_index)

    hessian_section = file_content[start_index:end_index]
    brinx = hessian_section.find("\n")
    hessian_section = hessian_section[brinx:]

    # supposedly projected out
    hessvec = np.array([float(i) for i in hessian_section.strip().split()])

    return hessvec

# not used now
def getForceConstants_fchk2(fchkfile: str):

    with open(fchkfile, 'r') as file:
        file_content = file.read()

    start_index = file_content.find("Cartesian Force Constants")

    end_index = file_content.find("Nonadiabatic coupling", start_index)
    hessian_section = file_content[start_index:end_index]
    brinx = hessian_section.find("\n")
    hessian_section = hessian_section[brinx:]
    # supposedly projected out
    hessvec = np.array([float(i) for i in hessian_section.strip().split()])

    return hessvec

# not used now
def getForceConstants_out(outfile: str):

    with open(outfile, 'r') as fileR:
        file_contentR = fileR.read()

    start_indexR = file_contentR.rfind(" Force constants in Cartesian coordinates:")
    end_indexR = file_contentR.find("Leave Link  716 at", start_indexR)
    hessian_sectionR = file_contentR[start_indexR:end_indexR]

    brinxR = hessian_sectionR.find("\n")
    hessian_sectionR = hessian_sectionR[brinxR:]
    brinxR = hessian_sectionR.find("\n")
    hessian_sectionR = hessian_sectionR[brinxR:]
    hessvecR = np.array([float(i.replace('D', 'e')) for i in hessian_sectionR.strip().split() if '.' in i])

    return hessvecR

# not used now
def getMass_fchk(fchkfile: str):
    with open(fchkfile, 'r') as file:
        file_content = file.read()

    start_index = file_content.find("Nuclear charges")
    end_index = file_content.find("Current cartesian coordinates", start_index)

    coords_section = file_content[start_index:end_index]
    brinx = coords_section.find("\n")
    coords_section = coords_section[brinx:]
    # supposedly projected out
    massvec = np.array([float(i) for i in coords_section.strip().split()])

    return massvec

# not used now
def getCoords_fchk(fchkfile: str):
    with open(fchkfile, 'r') as file:
        file_content = file.read()

    start_index = file_content.find("Current cartesian coordinates")
    end_index = file_content.find("Number of symbols in", start_index)
    coords_section = file_content[start_index:end_index]
    brinx = coords_section.find("\n")
    coords_section = coords_section[brinx:]

    # supposedly projected out
    coordsvec = np.array([float(i) for i in coords_section.strip().split()])

    return coordsvec

# used for fchk methods
class FormchkInterface:

    def __init__(self, file_path):
        self.file_path = file_path
        self.natm = NotImplemented
        self.nao = NotImplemented
        self.nmo = NotImplemented
        self.initialization()

    def initialization(self):
        self.natm = int(self.key_to_value("Number of atoms"))
        self.nao = int(self.key_to_value("Number of basis functions"))
        self.nmo = int(self.key_to_value("Number of independent functions"))

    def key_to_value(self, key, file_path=None):
        if file_path is None:
            file_path = self.file_path
        flag_read = False
        expect_size = -1
        vec = []
        with open(file_path, "r") as file:
            for l in file:
                if l[:len(key)] == key:
                    try:
                        expect_size = int(l[len(key):].split()[2])
                        flag_read = True
                        continue
                    except IndexError:
                        try:
                            return float(l[len(key):].split()[1])
                        except IndexError:
                            continue
                if flag_read:
                    try:
                        vec += [float(i) for i in l.split()]
                    except ValueError:
                        break
        if len(vec) != expect_size:
            raise ValueError("Number of expected size is not consistent with read-in size!")
        return np.array(vec)

    def total_energy(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Total Energy", file_path)

    def grad(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Cartesian Gradient", file_path).reshape((self.natm, 3))

    def dipole(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Dipole Moment", file_path)

    @staticmethod
    def tril_to_symm(tril: np.ndarray):
        dim = int(np.floor(np.sqrt(tril.size * 2)))
        if dim * (dim + 1) / 2 != tril.size:
            raise ValueError("Size " + str(tril.size) + " is probably not a valid lower-triangle matrix.")
        indices_tuple = np.tril_indices(dim)
        iterator = zip(*indices_tuple)
        symm = np.empty((dim, dim))
        for it, (row, col) in enumerate(iterator):
            symm[row, col] = tril[it]
            symm[col, row] = tril[it]
        return symm

    def hessian(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.tril_to_symm(self.key_to_value("Cartesian Force Constants", file_path))

    def polarizability(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        # two space after `Polarizability' is to avoid `Polarizability Derivative'
        return self.tril_to_symm(self.key_to_value("Polarizability  ", file_path))

    def polarderiv(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        # two space after `Polarizability' is to avoid `Polarizability Derivative'
        pre = self.key_to_value("Polarizability Derivatives", file_path).reshape(-1, 6)
        f = []
        for i in pre:
            f.append(self.tril_to_symm(i))
        return np.array(f)

    def dipolederiv(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Dipole Derivatives", file_path).reshape(-1, 3)

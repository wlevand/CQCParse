import numpy as np
import pickle

# not used now
def pNORMCO(filepath: str):
    """
    Getting mass-weighted normal coordinates

    :param filepath:
    :return:  massweightgeo - np.ndarray
    """

    with open(filepath, 'r') as file1:
        linesnormco = file1.readlines()
    #massweighted_normal_coordinates
    massweighted_geometry = []

    # Flags to identify sections
    in_geometry_section = False

    # Loop through each line in the file
    for lin in linesnormco:
        # Check for the start of geometry section
        if '% mass weighted coordinates' in lin:
            in_geometry_section = True
            continue

        # Check for the start of vibration section - can be used to collect normal coordinates
        elif '% frequency' in lin:
            in_geometry_section = False
            break

        # Capture atoms and geometry data
        if in_geometry_section:
            data1 = lin.strip().split()
            massweighted_geometry.append(data1)  # Exclude the atom label

    # Convert lists to numpy arrays
    massweighted_geometry = np.array(massweighted_geometry, dtype=float)
    return massweighted_geometry

# not used now
def pDIPDER(filepath: str):
    """
    Parsing the DIPDER file

    :param filepath:
    :return: atomsorder, dipolefull - (3*Natoms, 3)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
        dipmoment_cartder = []
        for l in lines:
            lr = l.strip().split()
            if len(lr) == 4:
                lr = np.array([float(i) for i in lr])
                dipmoment_cartder.append(lr)
    dipmoment_cartder = np.array(dipmoment_cartder)

    natoms = int(len(dipmoment_cartder) / 3)

    # order of atoms in cartesian coordinates
    atomsorder = dipmoment_cartder[:natoms, 0]

    x = dipmoment_cartder[:natoms, 1:].flatten()
    y = dipmoment_cartder[natoms:2 * natoms, 1:].flatten()
    z = dipmoment_cartder[2 * natoms:3 * natoms, 1:].flatten()

    dipolefull = np.array([x, y, z]).T
    return atomsorder, dipolefull

# used
def get_anharmonic_fundamentals(outfile: str, filetype: str = 'out') -> dict:
    """
    Extracts fundamental frequencies with anharmonic corrections from a given file.

    :param outfile: The path to the output file or the preloaded object
    :param filetype: The type of the file to process: 'out' for output files or 'pkl' for pickle files

    :return dict: A dictionary where keys are the mode indices (adjusted by subtracting 7)
            and values are the corresponding anharmonically corrected frequencies
    """
    if filetype == 'out':
        from parsing import parseCFOUR_forWilson
        things = parseCFOUR_forWilson.parse_output_file(outfile)
    elif filetype == 'pkl':
        with open(outfile, 'rb') as file:
            things = pickle.load(file)
    else:
        raise ValueError('Wrong file type. Choose "out" or "pkl".')

    labels = sorted({t[0] for t in things[0]})
    all_states_dict = dict(zip(things[0], things[2]))
    freqs = {b[0] - 7: all_states_dict[b] for b in (tuple([e]) for e in labels)}

    return freqs

def describe_structure(obj, level=0):
    """
    Recursively describe the structure of a Python object with complex data types
    """
    indent = '  ' * level
    obj_type = type(obj).__name__

    # Base case for non-iterable types
    if not hasattr(obj, '__iter__') or isinstance(obj, str):
        return obj_type

    # Handle dictionaries separately
    if isinstance(obj, dict):
        key_descriptions = {describe_structure(k, level + 1): describe_structure(v, level + 1) for k, v in obj.items()}
        dict_description = ',\n'.join([f"{indent}  {k}: {v}" for k, v in key_descriptions.items()])
        return f"dict(\n{dict_description}\n{indent})"

    # Handle other iterables (lists, tuples, sets, etc.)
    if isinstance(obj, (list, tuple, set)):
        # Describe each item in the iterable
        item_descriptions = [describe_structure(item, level + 1) for item in obj]
        # Get the type name ('list', 'tuple', or 'set') for the description
        type_name = obj_type
        iterable_description = ',\n'.join([f"{indent}  {item}" for item in item_descriptions])
        return f"{type_name}(\n{iterable_description}\n{indent})"

    # Fallback for any other types
    return obj_type

def getQuarticPost(freq: dict, quartic: np.ndarray, recipcm: bool = False):
    """ Derives cubic and quartic anharmonic constants.
        It takes reduced values [cm-1] from gaussian output
        and transforms it to :
          * cubic   force constants : [Hartree*amu(-3/2)*Bohr(-3)]
          * quartic force constants : [Hartree*amu(-2  )*Bohr(-4)]
    """
    if recipcm:

        n = len(freq)
        K4 = np.zeros((n, n, n, n), dtype=np.float64)

        for fijkl in quartic:
            i = int(fijkl[0]) - 7
            j = int(fijkl[1]) - 7
            k = int(fijkl[2]) - 7
            l = int(fijkl[3]) - 7
            d = np.float64(fijkl[4])

            indices = [(i, j, k, l), (i, j, l, k), (i, k, j, l), (i, k, l, j),
                       (i, l, j, k), (i, l, k, j), (j, i, k, l), (j, i, l, k),
                       (j, k, i, l), (j, k, l, i), (j, l, i, k), (j, l, k, i),
                       (k, i, j, l), (k, i, l, j), (k, j, i, l), (k, j, l, i),
                       (k, l, i, j), (k, l, j, i), (l, i, j, k), (l, i, k, j),
                       (l, j, i, k), (l, j, k, i), (l, k, i, j), (l, k, j, i)]

            for idx in indices:
                K4[idx] = d

        return K4

    else:
        n = len(freq)
        K4 = np.zeros((n, n, n, n), dtype=np.float64)

        for fijkl in quartic:

            i = int(fijkl[0]) - 7
            j = int(fijkl[1]) - 7
            k = int(fijkl[2]) - 7
            l = int(fijkl[3]) - 7
            d = np.float64(fijkl[4])
            d *= np.sqrt(freq[i] * freq[j] * freq[k] * freq[l])

            from scipy import constants
            a = np.sqrt(constants.h / constants.c / constants.physical_constants['unified atomic mass unit'][0] / 100)
            b = 10 ** 10 / 2 / np.pi / constants.physical_constants['Bohr radius'][0] / 10 ** 10
            Fact3R = (constants.physical_constants['hartree-joule relationship'][
                          0] / constants.h / constants.c / 100) * (a * b) ** 4

            d /= Fact3R

            indices = [(i, j, k, l), (i, j, l, k), (i, k, j, l), (i, k, l, j),
                       (i, l, j, k), (i, l, k, j), (j, i, k, l), (j, i, l, k),
                       (j, k, i, l), (j, k, l, i), (j, l, i, k), (j, l, k, i),
                       (k, i, j, l), (k, i, l, j), (k, j, i, l), (k, j, l, i),
                       (k, l, i, j), (k, l, j, i), (l, i, j, k), (l, i, k, j),
                       (l, j, i, k), (l, j, k, i), (l, k, i, j), (l, k, j, i)]

            for idx in indices:
                K4[idx] = d


        from scipy import constants
        # to go from amu to au mass unit (m_e)
        amc_au = constants.physical_constants['atomic mass constant'][0] / \
                 constants.physical_constants['atomic unit of mass'][0]
        # to Wilson units
        K4 /= amc_au ** 2
        return K4

# For pickled things
def unpickle(file: str):
    """
    Example here:

    import pickle
    import numpy as np

    vibdata = '../scriptsHPC/cfourscripts/vibdata.pkl'
    hf_cubicarray = '../scriptsHPC/cfourscripts/hf_cubicarray.pkl'
    hf_polarders = '../scriptsHPC/cfourscripts/hf_polarders.pkl'
    hf_vibdata = '../scriptsHPC/cfourscripts/hf_vibdata.pkl'
    # this data is from MOLDEN file
    hf_normalmodes = '../scriptsHPC/cfourscripts/hf_normalmodes.pkl'
    hf_rawdata_polar = '../scriptsHPC/cfourscripts/hf_rawdata_polar.pkl'

    lst = [hf_vibdata, hf_normalmodes, hf_cubicarray, hf_polarders, hf_rawdata_polar]
    for l in lst:
        unpickle(l)

    :param file:
    :return:
    """

    import pickle

    print(f'      Opening a pickle file {file}')

    with open(file, 'rb') as f:
        stuff = pickle.load(f)
    if type(stuff) == tuple and (type(stuff[0]) != str or type(stuff[0]) != float or type(stuff[0]) != int):
        print('  There are several things here')
        baselength = 7
        for i, thing in enumerate(stuff):
            print('\n' + ' ' * 6 + f'{i}:' + ' ' * (baselength - len(str(i))), type(thing))
            if type(thing) == dict:
                dictinfo(thing)
            elif type(thing) == np.ndarray:
                print(' ' * 6 + f'---- Numpy array with shape {thing.shape}')

            print(' ' * 6 + '==' * 20)

    else:
        print(f'  There is just one thing here: {type(stuff)}')
        if type(stuff) == np.ndarray:
            print('\n' + ' ' * 6 + f'---- Numpy array with shape {stuff.shape}')
            print(stuff)
        elif type(stuff) == list:
            print('\n' + ' ' * 6 + f'---- List of length {len(stuff)}')
        elif type(stuff) == dict:
            dictinfo(stuff)

def dictinfo(dct: dict, level: int = 0):
    """
    Getting info from dictionaries

    :param dct:
    :param level:
    :return:
    """
    levels = {0: 6, 1: 12, 2: 18}
    print('\n' + ' ' * levels[level] + f'---- Dictionary with {len(dct)} pairs')
    keystype = type(list(dct.keys())[0])
    valuestype = type(dct[list(dct.keys())[0]])

    print('\n' + ' ' * levels[level] + f'Keys are {keystype} and values are {valuestype}')
    print('\n' + ' ' * levels[level] + f'List of keys:')
    print(' ' * (levels[level] + 6) + str(list(dct.keys())))

    if valuestype == dict:
        print(' ' * levels[level] + '>>' * 30)
        level += 1

        for k1 in dct:
            if k1 == 'metadata':
                width = 25
                print('\n' + ' ' * levels[level] + 'Description' + ' ' * (width - 11), dct['metadata']['description'])
                print('\n' + ' ' * levels[level] + 'Contents' + ' ' * (width - 11))
                for descr in dct['metadata']['contents']:
                    # print(dct['input_data_info'][descr])
                    if type(dct['input_data_info'][descr]) == np.ndarray:
                        extra = 'with shape ' + str(dct['input_data_info'][descr].shape)
                    elif type(dct['input_data_info'][descr]) == list:
                        extra = 'with length ' + str(len(dct['input_data_info'][descr]))
                    else:
                        extra = ''
                    print(' ' * levels[level + 1] + f'{descr}:' + ' ' * (width - len(descr)),
                          dct['metadata']['contents'][descr], extra)

                print('\n')

            else:
                dictinfo(dct[k1], level)

            print(' ' * levels[level] + '>>' * 30)

# this is about getting raw hessian matrix and doing vib analysis with it
def hessianfromout(outfilename: str):
    """
    Getting hessian data from outfile0.out

    :param outfilename:
    :return:
    """
    # Read the contents of the file
    with open(outfilename, 'r') as file:
        file_content = file.read()

    hessian_start_index = file_content.find("Molecular hessian")
    dipole_line_index = file_content.find("Total dipole moment", hessian_start_index)
    hessian_section = file_content[hessian_start_index:dipole_line_index]

    hessian_lines = hessian_section.split('\n\n\n\n')[1]
    h = [k for k in hessian_lines.split('\n') if k!='']

    cleanh = []
    for i in h[1:]:
        if '.' in i:
            cleanh.extend([float(k) for k in i.split() if '.' in k])
    hvec = np.array(cleanh)

    return hvec

def solve_quadratic(a: float, b: float, c: float):
    import cmath  # Import the complex math module

    discriminant = cmath.sqrt(b**2 - 4*a*c)

    # Calculate the two solutions
    root1 = (-b + discriminant) / (2*a)
    root2 = (-b - discriminant) / (2*a)

    return root1, root2

def gethessinmat(hessinvec: np.ndarray):
    """
    Formation of Hessian matrix, for any source of it.
    Example:
        hessmat = gethessinmat(hessvec)
        eigenvalues2, mass_weighted_eigenvectors2 = jn.hessian_diagonalizer(hessmat)
    Or
        newnewhess = jn.project_hessian(mol, hessmat, is_linear= False, internal_coordinates= False)
        eigenvalues, mass_weighted_eigenvectors = jn.hessian_diagonalizer(newnewhess[0])

    Or
        hessmatR = gethessinmat(hessvecR)
        # frequency_analysis(coords, Hessian, elem=None, mass=None, energy=0.0,
        #             temperature=300.0, pressure=1.0, verbose=0, outfnm=None,
        #             note=None, wigner=None, ignore=0, normalized=True)
        gg = g.normal_modes.frequency_analysis(coordsvec, hessmat, mass=massvec)

    :param hessinvec:
    :return:
    """
    # Solve the quadratic equation
    roots = solve_quadratic(a=1, b=1, c=-2*len(hessinvec))

    # Print the solutions
#    print("Root 1:", roots[0])

    ncoords = int(roots[0].real)
    hessmat_mat = np.zeros((ncoords, ncoords))

    ind = 0
    for i in range(ncoords):
        for j in range(i+1):
            hessmat_mat[i, j] = hessinvec[ind]
            hessmat_mat[j, i] = hessinvec[ind]
            ind += 1

    return hessmat_mat

def getrotproj(filename: str):
    """
    Getting Rotationally projected vibrational frequencies from.out file

    E.g, filename = '../data/rawouts/anharm_hf_outfile0.out'
    :return:
    """
    # Read the contents of the file fchk
    with open(filename, 'r') as file:
        file_content = file.read()

    start_index = file_content.find("Rotationally projected vibrational frequencies")
    end_index = file_content.find("Zero-point energy:", start_index)

    # Extract the content starting from "Molecular hessian" up to "Total dipole moment"
    coords_section = file_content[start_index:end_index]
    brinx = coords_section.find("\n")
    coords_section = [float(k[1]) for k in [j.strip().split() for j in coords_section[brinx:][1:-1].split('\n')] if k and 'i' not in k[1]]

    print(coords_section[4:])
    print(len(coords_section[4:]))

def extract_anharmvib(filename: str):
    """
    From 'HARMONIC AND FUNDAMENTAL FREQUENCIES.' section of out file

    :param filename:
    :return:
    """
    with open(filename, 'r') as file:
        file_content = file.read()
    import re
    import pandas as pd

    # Define start and end patterns for the block
    start_pattern = r'HARMONIC AND FUNDAMENTAL FREQUENCIES.*\n-+\n'
    end_pattern = r'-+\n'

    start_match = re.search(start_pattern, file_content)
    end_match = re.search(end_pattern, file_content[start_match.end():])

    data_block = file_content[start_match.end():start_match.end() + end_match.start()]

    lines = data_block.strip().split('\n')
    head = [word + ' ' for word in lines[0].split()]
    headers = lines[1].split()

    column_names = [str1 + str2 for str1, str2 in zip(head, headers[1:])]
    column_names.insert(0, lines[1].split()[0])

    values = [list(map(float, line.split())) for line in lines[2:]]

    numpy_array = np.array(values)
    df = pd.DataFrame(numpy_array, columns=column_names)

    df['Mode'] = df['Mode'].astype(int)

    return df

# not used now
def computeRedMass4nm(filename: str):
    """
    Source : https://github.com/psi4/psi4/blob/8a781dcad54eac2114f8af0d742ccd8beb036ddb/psi4/driver/qcdb/vib.py#L579
    https://github.com/psi4/psi4/blob/master/psi4/driver/qcdb/vib.py

    :return:
    """
    mass = np.array([15.994914630, 12.000000000, 1.007825035, 1.007825035])
    nat = len(mass)

    # sqrtmmm = np.repeat(np.sqrt(mass), 3)
    # sqrtmmminv = np.divide(1.0, sqrtmmm)

    # filename = '../scriptsHPC/data/rawouts/anharm_hf_MOLDEN'
    with open(filename, 'rb') as f:
        moldendata = pMOLDEN(filename)

    print('Atoms:', moldendata[1])

    wL = np.array([moldendata[2][i] for i in moldendata[2]]).reshape(-1, 3 * nat)

    # this works now
    reduced_mass_u = np.divide(1.0, np.linalg.norm(wL.T, axis=0) ** 2)

    return reduced_mass_u

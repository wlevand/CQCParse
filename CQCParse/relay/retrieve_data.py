"""
Is this a documentation?
"""
from parsing import parseGaussian_forWilson, parseCFOUR_forWilson, parseCFOUR_extra, parseGaussian_extra
from typing import Any

import numpy as np
np.set_printoptions(linewidth=250, suppress=False, precision=12)

class CFOURdata:

    def __init__(self, data: dict[str:[str, dict]]):
        self.sourcetype = data['type']
        self.files = data['files']

    def getFundamentals(self) -> dict[int:float]:
        """
        Fundamental frequency with anharmonic corrections
        Returns: dict[int:float]
        """
        if self.sourcetype == 'out':
            fundamentals = parseCFOUR_extra.get_anharmonic_fundamentals(self.files['out'], filetype='out')
            # fundamentals_harm = parseCFOUR_forWilson.get_anharmonic_fundamentals(self.files['out'], filetype='out')
            allstates_CFOUR, allstates_CFOUR_harm = self.getAllStates()
            funds_c4 = {k: v for k, v in allstates_CFOUR_harm.items() if len(k) == 1}
            sorted_data_c4 = {k[0]: funds_c4[k] for k in sorted(funds_c4)}

            return fundamentals, sorted_data_c4

        elif self.sourcetype == 'pkl':
            fundamentals = parseCFOUR_extra.get_anharmonic_fundamentals(self.files['vibdata'], filetype='pkl')
            return fundamentals

    def getAllStates(self) -> dict[tuple[int]: float, tuple[int, int]: float,
                                   tuple[int, int, int]: float]:
        """
        Dictionary of all the states and their frequencies
        Return: dict[tuple[int]: float, tuple[int, int]: float, tuple[int, int, int]: float]
        E.g.: {(0,): 1000., (0, 1): 2000., (0, 1, 2): 3000.}
        """
        if self.sourcetype == 'out':
            ls0, ls1, ls2, ls3, ls4 = parseCFOUR_forWilson.parse_output_file(self.files['out'])

        elif self.sourcetype == 'pkl':
            vibdatapkl = self.files['vibdata']
            import pickle
            with open(vibdatapkl, 'rb') as file:
                ls0, ls1, ls2, ls3, ls4 = pickle.load(file)

        combd = dict(zip(ls0, ls2))
        Delta = {tuple([o - 7 for o in k]): v for k, v in combd.items()}
        combd_harm = dict(zip(ls0, ls4))
        Delta_harm = {tuple([o - 7 for o in k]): v for k, v in combd_harm.items()}

        return Delta, Delta_harm

    def getDipDers(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Dipole derivatives: first order and second order
        Return: tuple[np.ndarray - shape(NM, 3), np.ndarray - shape(NM, NM, 3)]
        """
        if self.sourcetype == 'out':
            mu = parseCFOUR_forWilson.getDipoleDers(self.files['dipolexyz'], self.files['out'])
            return mu

        elif self.sourcetype == 'pkl':
            dipolepkl = self.files['dipole']
            import pickle
            with open(dipolepkl, 'rb') as file:
                d = pickle.load(file)
            return d

    def getPolarDers(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Polarizability derivatives: first order and second order
        Return: tuple[np.ndarray - shape(NM, 3, 3), np.ndarray - shape(NM, NM, 3, 3)]
        """

        dipolepkl = self.files['polar']
        import pickle
        with open(dipolepkl, 'rb') as file:
            dalpha, d2alpha = pickle.load(file)

        return dalpha, d2alpha

    def getCFF(self) -> np.ndarray:
        """
        CFF: cubic force constant tensor
        Return: np.ndarray - shape(NM, NM, NM)
        """
        if self.sourcetype == 'out':
            cubic = parseCFOUR_forWilson.pCubicORQuartic(self.files['cubic'])
            freq, freq_harm = self.getFundamentals()
            cff = parseCFOUR_forWilson.getCubicPost(freq_harm, cubic)
            return cff

        elif self.sourcetype == 'pkl':
            cubicpkl = self.files['cubic']
            import pickle
            with open(cubicpkl, 'rb') as file:
                # first 3 columns are the normal mode indices, the last column holds the derivatives
                cff = pickle.load(file)

            freq, freq_harm = self.getFundamentals()
            cubicFC = parseCFOUR_forWilson.getCubicPost(freq_harm, cff)

            return cubicFC

    def getQFF(self, file: str = None) -> np.ndarray:
        """
        QFF: quartic force constant tensor
        Return: np.ndarray - shape(NM, NM, NM, NM)
        """

        if file is not None:
            quartic = parseCFOUR_forWilson.pCubicORQuartic(file)
            freq, freq_harm = self.getFundamentals()
            qff = parseCFOUR_extra.getQuarticPost(freq_harm, quartic)
            return qff

        else:
            if self.sourcetype == 'out':
                quartic = parseCFOUR_forWilson.pCubicORQuartic(self.files['quartic'])
                freq, freq_harm = self.getFundamentals()
                qff = parseCFOUR_extra.getQuarticPost(freq_harm, quartic)
                return qff

            elif self.sourcetype == 'pkl':
                cubicpkl = self.files['cubic']
                import pickle
                with open(cubicpkl, 'rb') as file:
                    # first 3 columns are the normal mode indices, the last column holds the derivatives
                    qff = pickle.load(file)

                freq, freq_harm = self.getFundamentals()
                quarticFC = parseCFOUR_extra.getQuarticPost(freq_harm, qff)

                return quarticFC


def getDimensionlessNM(datafile: str = None) -> dict:
    """
    Reduced (dimensionless) normal coordinates from QUADRATURE file
    Return: a transformation matrix with dimensionless normal coordinates
    """
    if datafile[-3:] == 'pkl':
        import pickle
        with open(datafile, 'rb') as file:
            # first 3 columns are the normal mode indices, the last column holds the derivatives
            undisplaced_matrix, dimless, freqs = pickle.load(file)
        return dimless

    else:
        undisplaced_matrix, freqs, dimless  = parseCFOUR_forWilson.pQUADRATURE(datafile)
        return dimless

class GaussianData:
    """
    Also fundamental frequencies with anharmonic corrections are needed
    Overtones and combination bands too
    input_data_info is a list of np.arrays 'mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'
    """
    def __init__(self, data: dict[str:[str, dict]]):
        self.sourcetype = data['type']
        self.files = data['files']

    def getDipDersCart_fchk(self):
        fchk_parser = parseGaussian_extra.FormchkInterface(self.files['fchk'])
        dipderCart = fchk_parser.dipolederiv()
        return dipderCart

    def getPolarDersCart_fchk(self):
        fchk_parser = parseGaussian_extra.FormchkInterface(self.files['fchk'])
        polder = fchk_parser.polarderiv()
        return polder

    def get_hessian_tensor_fchk(self):
        fchk_parser = parseGaussian_extra.FormchkInterface(self.files['fchk'])
        hessian = fchk_parser.hessian()
        return hessian

    def getFundamentals(self) -> dict[int:float]:
        """
        Fundamental frequency with anharmonic corrections
        Returns: dict[int:float]
        """
        results = parseGaussian_forWilson.parse_frequencies(self.files['3quanta'])
        funddict = {int(k)-1: float(v) for k, v in zip(results['Fundamental Bands']['mode_a'], results['Fundamental Bands'][2])}
        funddict_harm = {int(k)-1: float(v) for k, v in zip(results['Fundamental Bands']['mode_a'], results['Fundamental Bands'][1])}
        return funddict, funddict_harm

    def getAllStates(self) -> dict[tuple[int]: float, tuple[int, int]: float,
                                    tuple[int, int, int]: float]:
          """
          Dictionary of all the states and their frequencies
          Return: dict[tuple[int]: float, tuple[int, int]: float, tuple[int, int, int]: float]
          """
          if self.sourcetype == 'fchk':
                pass
          elif self.sourcetype == 'log':
                results = parseGaussian_forWilson.parse_frequencies(self.files['3quanta'])
                results['Combination Bands']['mode_c'] = results['Combination Bands']['mode_c'].fillna(0)
                results['Combination Bands']['n_c'] = results['Combination Bands']['n_c'].fillna(0)

                funddict = {tuple([int(k)-1]): float(v) for k, v in
                            zip(results['Fundamental Bands']['mode_a'], results['Fundamental Bands'][2])}
                states = {
                    tuple(sorted([int(k) - 1 for k in t.split()] * int(n))): float(v)
                    for t, v, n in
                    zip(results['Overtones']['mode_a'], results['Overtones'][2], results['Overtones']['n_a'])
                }
                combinationbands = {
                    tuple(
                        sorted([int(k) - 1 for k in t1.split()] * int(n1) + [int(l) - 1 for l in t2.split()] * int(n2) + [(int(t3)-1)] * int(n3))
                    ): float(v)
                    for t1, t2, t3, v, n1, n2, n3 in
                    zip(results['Combination Bands']['mode_a'], results['Combination Bands']['mode_b'],
                        results['Combination Bands']['mode_c'],
                        results['Combination Bands'][4], results['Combination Bands']['n_a'],
                        results['Combination Bands']['n_b'], results['Combination Bands']['n_c'])
                }
                allstates_anharm = {**funddict, **states, **combinationbands}
                funddict1 = {tuple([int(k)-1]): float(v) for k, v in
                            zip(results['Fundamental Bands']['mode_a'], results['Fundamental Bands'][1])}
                states1 = {tuple(sorted([int(k)-1 for k in t.split()]*int(n))): float(v) for t, v, n in
                          zip(results['Overtones']['mode_a'], results['Overtones'][1], results['Overtones']['n_a'])}
                combinationbands1 = {
                    tuple(
                        sorted([int(k) - 1 for k in t1.split()] * int(n1) + [int(l) - 1 for l in t2.split()] * int(n2) + [(int(t3)-1)] * int(n3))
                    ): float(v)
                    for t1, t2, t3, v, n1, n2, n3 in
                    zip(results['Combination Bands']['mode_a'], results['Combination Bands']['mode_b'],
                        results['Combination Bands']['mode_c'],
                        results['Combination Bands'][3], results['Combination Bands']['n_a'],
                        results['Combination Bands']['n_b'], results['Combination Bands']['n_c'])
                }
                allstates_harm = {**funddict1, **states1, **combinationbands1}

                # allstates_anharm = {key:allstates_anharm[key] for key in sorted(allstates_anharm,key=allstates_anharm.get)}
                return allstates_anharm, allstates_harm

    def getDipDers(self) -> tuple[Any, ...]:
        """
        Dipole derivatives: first order and second order
        Return: tuple[np.ndarray - shape(NM, 3), np.ndarray - shape(NM, NM, 3)]
        """
        if self.sourcetype == 'fchk':
            dipderCart = self.getDipDersCart_fchk()
            pass

        elif self.sourcetype == 'log':
            dipl, units = parseGaussian_forWilson.parse_dipole_moment(self.files['log'])
            a2d = dipl.loc[dipl['P'] == 'P1', ['X', 'Y', 'Z']].to_numpy()
            a2d2 = dipl.loc[dipl['P'] == 'P2', ['X', 'Y', 'Z']].to_numpy()
            a2d2_3d = np.zeros((6, 6, 3))

            for i, j, xyz in zip(dipl.loc[dipl['P'] == 'P2', 'i'], dipl.loc[dipl['P'] == 'P2', 'j'], a2d2):
                a2d2_3d[int(i) - 1, int(j) - 1] = xyz
                a2d2_3d[int(j) - 1, int(i) - 1] = xyz
            return tuple([a2d, a2d2_3d])

    def getPolarDers(self) -> tuple[Any, ...]:
        """
        Polarizability derivatives: first order and second order
        Return: tuple[np.ndarray - shape(NM, 3, 3), np.ndarray - shape(NM, NM, 3, 3)]
        """
        if self.sourcetype == 'log':
            pol = parseGaussian_forWilson.parse_polarizability(self.files['log'])
            nm = int(pol.loc[pol[0] == 'P1', 1].max())
            p1_3d = np.zeros((nm, 3, 3))
            for i in pol.loc[pol[0] == 'P1', 1].unique():
                # in the rows with the current 'i' value and 'P' is 'P1', select columns 5, 6, and 7
                xyz = pol.loc[(pol[0] == 'P1') & (pol[1] == i), [5, 6, 7]].values
                # xyz - 2d array, will be part of 3d array
                p1_3d[int(i) - 1] = xyz

            nm_i = int(pol.loc[pol[0] == 'P2', 1].max())
            nm_j = int(pol.loc[pol[0] == 'P2', 2].max())

            p2_4d = np.zeros((nm_i, nm_j, 3, 3))

            for i in pol.loc[pol[0] == 'P2', 1].unique():
                for j in pol.loc[pol[0] == 'P2', 2].unique():
                    # in the rows with the current 'i' and 'j' values and 'P' is 'P2', select columns 5, 6, and 7
                    xyz = pol.loc[(pol[0] == 'P2') & (pol[1] == i) & (pol[2] == j), [5, 6, 7]].values
                    # check if xyz is not empty
                    if xyz.shape[0] != 0:
                        p2_4d[int(i) - 1, int(j) - 1] = xyz
                        p2_4d[int(j) - 1, int(i) - 1] = xyz

            return tuple([p1_3d, p2_4d])

    def getCFF(self) -> np.ndarray:
        """
        CFF: cubic force constant tensor
        Return: np.ndarray - shape(NM, NM, NM)
        """
        if self.sourcetype == 'log':
            cubic_df = parseGaussian_forWilson.parse_cubic_constants(self.files['3quanta'])[0]
            selected_df = cubic_df[['I', 'J', 'K', 'K(I,J,K)']]
            cubic = selected_df.to_numpy()
            freq, freq_harm = self.getFundamentals()
            cff = parseGaussian_forWilson.get_cubic_post(len(freq_harm), cubic)
            return cff

    def getQFF(self, file: str = None) -> np.ndarray:
        """
        CFF: cubic force constant tensor
        Return: np.ndarray - shape(NM, NM, NM)
        """
        if file is not None:
            file_here = file
        else:
            if self.sourcetype == 'log':
                file_here = self.files['3quanta']
            else:
                file_here = ''
                print('No file provided')

        quartic_df = parseGaussian_forWilson.parse_quartic_constants(file_here)[0]
        selected_df = quartic_df[['I', 'J', 'K', 'L', 'K(I,J,K,L)']]
        quartic = selected_df.to_numpy()
        freq, freq_harm = self.getFundamentals()
        qff = parseGaussian_forWilson.get_quartic_post(len(freq_harm), quartic)
        return qff
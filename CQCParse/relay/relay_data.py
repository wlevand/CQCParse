"""
Collecting the paths to CFOUR and Gaussian output files from a CSV file.
Column names in the CSV are:
     code, method, basis_set                        - identifications of the data
     c4_ZMAT, c4_outfile_orig_hess, c4_QUADRATURE   - supplementary CFOUR files
     pkl_dimensionless, pkl_dipole, pkl_polar,
     pkl_polar_raw, pkl_polar_data, pkl_vibdata     - "pickled" CFOUR data
     c4_dipolexyz, c4_cubic, c4_out                 - CFOUR files that contain derivatives and vib. energy levels
     g16_3quanta_full                               - Gaussian output file
"""

import pandas as pd

class DataVault:

    def __init__(self, csv_location: str = None):

        if csv_location is None:
            self.csv_location = './test_database/mini_files_database.csv'
            #'/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/files_database.csv'
        else:
            self.csv_location = csv_location


    def read_csv_DB(self) -> pd.DataFrame:
        """
        """
        database = pd.read_csv(self.csv_location)
        # columnsDB = list(database.columns)
        # print('\ncolumnsDB\n', columnsDB)

        return database

    def getting_files_DB(self, sourceProgram: str, printing: bool = False) -> pd.DataFrame:
        """
        """

        DB = self.read_csv_DB()

        if sourceProgram == 'gaussian':
            filtered_df = DB.query('g16_3quanta_full.notna() and g16_3quanta_full != ""')
            selected_columns_df = filtered_df[['code', 'method', 'basis_set', 'g16_3quanta_full']]
            return selected_columns_df

        else:
            # c4_ZMAT, c4_outfile_orig_hess,
            # c4_QUADRATURE, pkl_dimensionless, pkl_dipole,
            # c4_dipolexyz, pkl_polar, pkl_polar_raw, pkl_polar_data, c4_cubic,
            # c4_out, pkl_vibdata
            columns_to_check = ['c4_dipolexyz', 'pkl_polar', 'c4_cubic', 'c4_out']
            conditions = " and ".join([f"{col}.notna() and {col} != ''" for col in columns_to_check])
            filtered_df = DB.query(conditions)

            # filtered_df = DB.query('c4_dipolexyz.notna() and c4_dipolexyz != "" and pkl_polar.notna() and pkl_polar != "" ')
            if printing:
                selected_columns_df = filtered_df[['code', 'method', 'basis_set', 'c4_out']]
            else:
                selected_columns_df = filtered_df[['code', 'method', 'basis_set', 'c4_out', 'c4_cubic',
                                                   'c4_dipolexyz', 'pkl_polar']]

            return selected_columns_df

    def make_DatainputDict(self, sourceProgram: str, mol_tuple: tuple, pref_dir: str = '') -> dict:
        """
        Returns a dictionary of filetypes and their locations taken from the CSV file
        for selected software source and (molecule, method, basis) identification tuple

        pref_dir - prefix directory to the file locations given in the CSV files
        """
        dataframe = self.getting_files_DB(sourceProgram)
        mol_code, method, basis = mol_tuple
        files_dict = {'mol_code': mol_code, 'method': method, 'basis': basis}

        narrow_df = dataframe.loc[(dataframe['code'] == mol_code)
                                  & (dataframe['method'] == method)
                                  & (dataframe['basis_set'] == basis)]

        if len(narrow_df) > 1:
            print('Something is wrong, more than one file found. First one is taken here.')

        elif len(narrow_df) == 0:
            print('Not found requested (molecule, method, basis) identificator. Try again.')

        else:
            if sourceProgram == 'gaussian':
                result = {'source': 'gaussian', 'type': 'log'}

                files_dict.update({'3quanta': pref_dir+narrow_df.iloc[0]['g16_3quanta_full'],
                                   'log': pref_dir+narrow_df.iloc[0]['g16_3quanta_full']})
                result['files'] = files_dict
                return result

            elif sourceProgram == 'cfour':
                result = {'source': 'cfour', 'type': 'out'}
                files_dict.update({'out': pref_dir+narrow_df.iloc[0]['c4_out'],
                                   'cubic': pref_dir+narrow_df.iloc[0]['c4_cubic'],
                                   'dipolexyz': pref_dir+narrow_df.iloc[0]['c4_dipolexyz'][:-1],
                                   'polar': pref_dir+narrow_df.iloc[0]['pkl_polar'],
                                   'out_anharm_final': pref_dir+narrow_df.iloc[0]['c4_out'],
                                   'polar_pkl': pref_dir+narrow_df.iloc[0]['pkl_polar']})
                result['files'] = files_dict
                return result

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

import logging
logger = logging.getLogger("CQCParse")

# class DataVault:
#     def __init__(self, csv_location: str = None):
#         self.csv_location = csv_location or './test_database/mini_files_database.csv'
    
#     def read_csv_DB(self) -> pd.DataFrame:
#         """
#         Reads the CSV database and returns it as a DataFrame.
#         """
#         try:
#             return pd.read_csv(self.csv_location)
#         except FileNotFoundError:
#             logger.error(f"CSV file not found at {self.csv_location}")
#             raise FileNotFoundError(f"CSV file not found at {self.csv_location}")
        
#     def filter_database(self, source_program: str, printing: bool = False) -> pd.DataFrame:
#         """
#         Filters the database based on the source program and returns the relevant DataFrame.
#         """
#         db = self.read_csv_DB()
#         if source_program == 'gaussian':
#             return self._filter_gaussian(db)
#         elif source_program == 'cfour':
#             return self._filter_cfour(db, printing)
#         else:
#             raise ValueError(f"Unsupported source program: {source_program}")
        
#     def _filter_gaussian(self, db: pd.DataFrame) -> pd.DataFrame:
#         """
#         Filters the database for Gaussian source program.
#         """
#         filtered_df = db[db["g16_3quanta_full"].notna() & (db["g16_3quanta_full"] != "")]
#         return filtered_df[['code', 'method', 'basis_set', 'g16_3quanta_full']]
    
#     def _filter_cfour(self, db: pd.DataFrame, printing: bool) -> pd.DataFrame:
#         """
#         Filters the database for CFOUR source program.
#         """
#         columns_to_check = ['c4_dipolexyz', 'pkl_polar', 'c4_cubic', 'c4_out']
#         filtered_df = db.query(" and ".join([f"{col}.notna() and {col} != ''" for col in columns_to_check]))
#         if printing:
#             return filtered_df[['code', 'method', 'basis_set', 'c4_out']]
#         else:
#             return filtered_df[['code', 'method', 'basis_set', 'c4_out', 'c4_cubic', 'c4_quartic',
#                                 'c4_dipolexyz', 'pkl_polar', 'molden']]
        
#     def make_data_input_dict(self, source_program: str, mol_tuple: tuple, csvfile_dir: str = '') -> dict:
#         """
#         Creates a dictionary of file types and their locations for the given source program and molecule tuple.
#         """
#         dataframe = self.filter_database(source_program)
#         mol_code, method, basis = mol_tuple
#         narrow_df = dataframe.loc[
#             (dataframe['code'] == mol_code) &
#             (dataframe['method'] == method) &
#             (dataframe['basis_set'] == basis)
#         ]
#         if len(narrow_df) > 1:
#             raise AssertionError('More than one file found. Please check the database.')
#         elif len(narrow_df) == 0:
#             raise AssertionError('No matching entry found for the given molecule, method, and basis.')
#         return self._build_file_dict(source_program, narrow_df.iloc[0], csvfile_dir)
    
#     def _build_file_dict(self, source_program: str, row: pd.Series, csvfile_dir: str) -> dict:
#         """
#         Builds the file dictionary based on the source program and row data.
#         """
#         files_dict = {'mol_code': row['code'], 'method': row['method'], 'basis': row['basis_set']}
#         if source_program == 'gaussian':
#             return {
#                 'source': 'gaussian',
#                 'type': 'log',
#                 'files': {**files_dict, 'log': csvfile_dir + row['g16_3quanta_full']}
#             }
#         elif source_program == 'cfour':
#             return {
#                 'source': 'cfour',
#                 'type': 'out',
#                 'files': {
#                     **files_dict,
#                     'out': csvfile_dir + row['c4_out'],
#                     'cubic': csvfile_dir + row['c4_cubic'],
#                     'quartic': csvfile_dir + row['c4_quartic'],
#                     'dipolexyz': csvfile_dir + row['c4_dipolexyz'][:-1],
#                     'polar': csvfile_dir + row['pkl_polar'],
#                     'out_anharm_final': csvfile_dir + row['c4_out'],
#                     'polar_pkl': csvfile_dir + row['pkl_polar'],
#                     'molden': csvfile_dir + row['molden']
#                 }
#             }
#         else:
#             raise ValueError(f"Unsupported source program: {source_program}")

class DataVault:
    def __init__(self, csv_location: str = None):
        self.csv_location = csv_location #or './test_database/mini_files_database.csv'

    def read_csv_DB(self) -> pd.DataFrame:
        """
        Reads the CSV database and returns it as a DataFrame.
        """
        try:
            return pd.read_csv(self.csv_location)
        except FileNotFoundError:
            logger.error(f"CSV file not found at {self.csv_location}")
            raise FileNotFoundError(f"CSV file not found at {self.csv_location}")
        
    def filter_database(self, source_program: str, printing: bool = False) -> pd.DataFrame:
        """
        Filters the database based on the source program and returns the relevant DataFrame.
        """
        db = self.read_csv_DB()
        # print(db)
        if source_program == 'gaussian':
            return self._filter_gaussian(db)
        elif source_program == 'cfour':
            return self._filter_cfour(db, printing)
        else:
            raise ValueError(f"Unsupported source program: {source_program}")
        
    def _filter_gaussian(self, db: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the database for Gaussian source program.
        """
        filtered_df = db[db["file_location"].notna() & (db["file_location"] != "")]
        return filtered_df[['Calc_Type', 'Name', 'Conformer_ID', 'Method', 'Basis', 'file_location']]
    
    def _filter_cfour(self, db: pd.DataFrame, printing: bool = True) -> pd.DataFrame:
        """
        Filters the database for CFOUR source program.
        """
        columns_to_check = ['out', 'polar_pkl', 'cff', 'qff', 'dipolex', 'molden']
        filtered_df = db.query(" and ".join([f"{col}.notna() and {col} != ''" for col in columns_to_check]))
        if printing:
            return filtered_df[['Calc_Type', 'Name', 'Conformer_ID', 'Method', 'Basis', 'out']]
        else:
            return filtered_df[['Calc_Type', 'Name', 'Conformer_ID', 'Method', 'Basis', 'out', 'cff', 'qff', 'dipolex', 'polar_pkl', 'molden']]
    
    def make_data_input_dict(self, source_program: str, mol_tuple: tuple) -> dict:
        """
        Creates a dictionary of file types and their locations for the given source program and molecule tuple.
        >> mol_name, conformer, method, basis = mol_tuple
        """

        dataframe = self.filter_database(source_program)
        mol_name, conformer, method, basis = mol_tuple
        narrow_df = dataframe.loc[
            (dataframe['Name'] == mol_name) &
            (dataframe['Conformer_ID'] == conformer) &
            (dataframe['Method'] == method) &
            (dataframe['Basis'] == basis)
        ]

        try:
            if narrow_df.empty:
                raise AssertionError('No matching entry found for the given molecule, method, and basis.')
            
            calc_type = narrow_df.iloc[0]['Calc_Type']
            
            if calc_type=='full':
                if len(narrow_df) > 1:
                    raise AssertionError('More than one file found. Please check the database.')
                elif len(narrow_df) == 0:
                    raise AssertionError('No matching entry found for the given molecule, method, and basis.')
                return self._build_file_dict(source_program, narrow_df.iloc[0])
            else:
                logger.warning(f"Skipping {source_program} data collection for {mol_tuple} as Calc_Type is not 'full'.")
                return {}        
        
        except AssertionError as e:
            logger.error(f"AssertionError: {e}")
            raise e

    
    def _build_file_dict(self, source_program: str, row: pd.Series) -> dict:
        """
        Builds the file dictionary based on the source program and row data.
        """
        files_dict = {'mol_name': row['Name'], 'method': row['Method'], 'basis': row['Basis']}
        if source_program == 'gaussian':
            return {
                'source': 'gaussian',
                'type': 'log',
                'files': {**files_dict, 'log': row['file_location']}
            }
        elif source_program == 'cfour':
            return {
                'source': 'cfour',
                'type': 'out',
                'files': {
                    **files_dict,
                    'out': row['out'],
                    'out_anharm_final': row['out'],
                    'cubic': row['cff'],
                    'quartic': row['qff'],
                    'dipolexyz': row['dipolex'][:-1],
                    'polar_pkl': row['polar_pkl'],
                    'molden': row['molden']
                }
            }
        else:
            raise ValueError(f"Unsupported source program: {source_program}")
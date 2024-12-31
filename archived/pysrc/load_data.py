import pandas as pd
import numpy as np
import os
import warnings

# def load_data(datafile, Spec, sample=None, load_excel=False):
#     """
#     Load vintage of data from file and format as structure

#     Parameters:
#         datafile (str): Filename of Microsoft Excel workbook file
#         Spec (dict): Model specification containing SeriesID and other info
#         sample (float, optional): Sample period start date in numeric form
#         load_excel (bool, optional): Flag to force loading from Excel

#     Returns:
#         X (np.ndarray): T x N numeric array, transformed dataset
#         Time (np.ndarray): T x 1 numeric array, date number with observation dates
#         Z (np.ndarray): T x N numeric array, raw (untransformed) dataset
#     """
#     print('Loading data...')

#     ext = os.path.splitext(datafile)[1]  # file extension
#     idx = datafile.rfind(os.path.sep)
#     datafile_mat = os.path.join(datafile[:idx], 'mat', os.path.splitext(datafile[idx + 1:])[0] + '.npz')

#     if os.path.exists(datafile_mat) and not load_excel:
#         # Load raw data from a NumPy formatted binary (.npz) file
#         with np.load(datafile_mat) as data:
#             Z = data['Z']
#             Time = data['Time']
#             Mnem = data['Mnem']
#     elif ext in ['.xlsx', '.xls']:
#         # Read raw data from Excel file
#         Z, Time, Mnem = read_data(datafile)
#         np.savez(datafile_mat, Z=Z, Time=Time, Mnem=Mnem)
#     else:
#         raise ValueError('Only Microsoft Excel workbook files supported.')

#     # Sort data based on model specification
#     Z = sort_data(Z, Mnem, Spec)
    
#     # Transform data based on model specification
#     X, Time, Z = transform_data(Z, Time, Spec)

#     # Drop data not in estimation sample
#     if sample is not None:
#         X, Time, Z = drop_data(X, Time, Z, sample)

#     return X, Time, Z

def load_data(datafile, Spec, sample=None, load_excel=False):
    """
    Load vintage of data from file and format as structure

    Parameters:
        datafile (str): Filename of Microsoft Excel workbook file
        Spec (dict): Model specification containing SeriesID and other info
        sample (float, optional): Sample period start date in numeric form
        load_excel (bool, optional): Flag to force loading from Excel

    Returns:
        X (np.ndarray): T x N numeric array, transformed dataset
        Time (np.ndarray): T x 1 numeric array, date number with observation dates
        Z (np.ndarray): T x N numeric array, raw (untransformed) dataset
    """
    print('Loading data...')

    ext = os.path.splitext(datafile)[1]  # file extension
    idx = datafile.rfind(os.path.sep)
    datafile_mat = os.path.join(datafile[:idx], 'mat', os.path.splitext(datafile[idx + 1:])[0] + '.npz')

    if os.path.exists(datafile_mat) and not load_excel:
        # Load raw data from a NumPy formatted binary (.npz) file
        with np.load(datafile_mat, allow_pickle=True) as data:
            Z = data['Z']
            Time = data['Time']
            Mnem = data['Mnem']
    elif ext in ['.xlsx', '.xls']:
        # Read raw data from Excel file
        Z, Time, Mnem = read_data(datafile)
        # np.savez(datafile_mat, Z=Z, Time=Time, Mnem=Mnem)
    else:
        raise ValueError('Only Microsoft Excel workbook files supported.')

    # Sort data based on model specification
    Z = sort_data(Z, Mnem, Spec)
    
    # Transform data based on model specification
    X, Time, Z, header = transform_data(Z, Time, Spec)

    # Drop data not in estimation sample
    if sample is not None:
        X, Time, Z = drop_data(X, Time, Z, sample)

    # Z = np.vstack([header, Z])
    # X = np.vstack([header, X])

    return X, Time, Z


def read_data(datafile):
    """
    Read data from Microsoft Excel workbook file

    Parameters:
        datafile (str): Filename of the Excel file

    Returns:
        Z (np.ndarray): Raw (untransformed) observed data
        Time (np.ndarray): Observation periods for the time series data
        Mnem (list): Series ID for each variable
    """
    df = pd.read_excel(datafile, sheet_name='data', header=None, engine="openpyxl")
    Mnem = df.iloc[0, 1:].tolist()

    # if os.name == 'nt':  # Check if the operating system is Windows
    #     Time = pd.to_datetime(df.iloc[1:, 0], format='%m/%d/%Y').astype(np.int64) // 10**9
    #     Z = df.iloc[1:, 1:].to_numpy()
    # else:
    #     Time = (df.iloc[1:, 0] + pd.Timestamp('1899-12-31').to_julian_date()).to_numpy()
    #     Z = df.iloc[:, 1:].to_numpy()
    Time = df.iloc[1:, 0].to_numpy()
    Z = df.iloc[:, 1:].to_numpy()
    return Z, Time, Mnem


def sort_data(Z, Mnem, Spec):
    """
    Sort series by order of model specification

    Parameters:
        Z (np.ndarray): Raw data
        Mnem (list): Series ID for each variable
        Spec (dict): Model specification

    Returns:
        Z (np.ndarray): Sorted data according to Spec.SeriesID
    """
    in_spec = np.isin(Mnem, Spec['seriesid'])
    Mnem = [mnem for mnem, keep in zip(Mnem, in_spec) if keep]
    Z = Z[:, in_spec]

    # Sort series by ordering of Spec
    N = len(Spec['seriesid'])
    print(Mnem)
    permutation = [Mnem.index(spec_id) for spec_id in Spec['seriesid']]

    Mnem = [Mnem[i] for i in permutation]
    Z = Z[:, permutation]

    return Z


# def transform_data(Z, Time, Spec):
#     """
#     Transforms each data series based on Spec.Transformation

#     Parameters:
#         Z (np.ndarray): Raw (untransformed) observed data
#         Time (np.ndarray): Observation periods for the time series data
#         Spec (dict): Model specification

#     Returns:
#         X (np.ndarray): Transformed data (stationary to enter DFM)
#         Time (np.ndarray): Adjusted time data
#         Z (np.ndarray): Adjusted raw data
#     """
#     T, N = Z.shape

#     X = np.full((T, N), np.nan)
#     for i in range(N):
#         formula = Spec["transformation"][i]
#         freq = Spec["frequency"][i]
#         step = 1 if freq == "m" else 3
#         t1 = step
#         n = step / 12
#         series = Spec["seriesname"][i]

#         if formula == "lin":  # Levels (No Transformation)
#             X[:, i] = Z[:, i]
#         elif formula == "chg":  # Change (Difference)
#             X[t1:, i] = np.concatenate(([np.nan], Z[t1:, i] - Z[:-t1, i]))
#         elif formula == "ch1":  # Year over Year Change (Difference)
#             if T > 12:
#                 X[12 + t1 :, i] = Z[12 + t1 :, i] - Z[:-12, i]
#         elif formula == "pch":  # Percent Change
#             X[t1:, i] = 100 * np.concatenate(([np.nan], Z[t1:, i] / Z[:-t1, i] - 1))
#         elif formula == "pc1":  # Year over Year Percent Change
#             if T > 12:
#                 X[12 + t1 :, i] = 100 * (Z[12 + t1 :, i] / Z[:-12, i] - 1)
#         elif formula == "pca":  # Percent Change (Annual Rate)
#             X[t1:, i] = 100 * np.concatenate(
#                 ([np.nan], (Z[t1:, i] / Z[:-step, i]) ** (1 / n) - 1)
#             )
#         elif formula == "log":  # Natural Log
#             X[:, i] = np.log(Z[:, i])
#         else:
#             warnings.warn(
#                 f"Transformation '{formula}' not found for {series}. Using untransformed data."
#             )
#             X[:, i] = Z[:, i]

#     # Drop first quarter of observations since transformations cause missing values
#     Time = Time[3:]
#     Z = Z[3:, :]
#     X = X[3:, :]

#     return X, Time, Z

def transform_data(Z, Time, Spec):
    """
    Transforms each data series based on Spec.Transformation

    Parameters:
        Z (np.ndarray): Raw (untransformed) observed data
        Time (np.ndarray): Observation periods for the time series data
        Spec (dict): Model specification

    Returns:
        X (np.ndarray): Transformed data (stationary to enter DFM)
        Time (np.ndarray): Adjusted time data
        Z (np.ndarray): Adjusted raw data
    """
    header = Z[0, :]
    Z = Z[1:, :]

    T, N = Z.shape

    X = np.full((T, N), np.nan)

    for i in range(N):
        formula = Spec["transformation"][i]
        freq = Spec["frequency"][i]
        step = 1 if freq == "m" else 3
        t1 = step
        n = step / 12

        assert header[i]== Spec["seriesid"][i]
        series = Spec["seriesname"][i]
        
        if formula == "lin":  # Levels (No Transformation)
            X[:, i] = Z[:, i]
        elif formula == "chg":  # Change (Difference)
            X[(t1-1):, i] = np.concatenate(([np.nan], Z[t1:, i] - Z[:-t1, i]))
        elif formula == "ch1":  # Year over Year Change (Difference)
            if T > 12:
                X[12 + t1 :, i] = Z[12 + t1 :, i] - Z[:-12, i]
        elif formula == "pch":  # Percent Change
            X[(t1-1):, i] = 100 * np.concatenate(([np.nan], Z[t1:, i] / Z[:-t1, i] - 1))
        elif formula == "pc1":  # Year over Year Percent Change
            if T > 12:
                X[12 + t1 :, i] = 100 * (Z[12 + t1 :, i] / Z[:-12, i] - 1)
        elif formula == "pca":  # Percent Change (Annual Rate)
            X[(t1-1):, i] = 100 * np.concatenate(
                ([np.nan], (Z[t1:, i] / Z[:-step, i]) ** (1 / n) - 1)
            )
        elif formula == "log":  # Natural Log
            X[:, i] = np.log(Z[:, i])
        else:
            warnings.warn(
                f"Transformation '{formula}' not found for {series}. Using untransformed data."
            )
            X[:, i] = Z[:, i]

    # Drop first quarter of observations since transformations cause missing values
    Time = Time[3:]
    Z = Z[3:, :]
    X = X[3:, :]

    return X, Time, Z, header


def drop_data(X, Time, Z, sample):
    """
    Remove data not in estimation sample

    Parameters:
        X (np.ndarray): Transformed data
        Time (np.ndarray): Time data
        Z (np.ndarray): Raw data
        sample (float): Sample period start date in numeric form

    Returns:
        X (np.ndarray): Filtered transformed data
        Time (np.ndarray): Filtered time data
        Z (np.ndarray): Filtered raw data
    """
    idx_drop = Time < sample

    Time = Time[~idx_drop]
    X = X[~idx_drop, :]
    Z = Z[~idx_drop, :]

    return X, Time, Z

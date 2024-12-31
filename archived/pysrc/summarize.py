import numpy as np
import pandas as pd

def summarize(X, Time, Spec, vintage):
    """
    Summarize and display the detail table for data entering the DFM.

    Description:
        Display the detail table for the nowcast, decomposing nowcast changes
        into news and impacts for released data series.
    """
    print('\n\n\n')
    print('Table 2: Data Summary \n')

    T, N = X.shape
    print(f'N = {N:4d} data series')
    print(f'T = {T:4d} observations from {Time[0]:%Y-%m-%d} to {Time[-1]:%Y-%m-%d}')

    print(f'{"Data Series":30s} | {"Observations":17s} {"Units":12s} {"Frequency":10s} {"Mean":8s} {"Std. Dev.":8s} {"Min":10s} {"Max":10s}')
    print('-' * 130)

    for i in range(N):
        # time indexes for which there are observed values for series i
        t_obs = ~np.isnan(X[:, i])

        data_series = Spec['seriesname'][i]
        if len(data_series) > 30:
            data_series = f'{data_series[:27]}...'

        series_id = Spec['seriesid'][i]
        if len(series_id) > 28:
            series_id = f'{series_id[:25]}...'
        series_id = f'[{series_id}]'

        num_obs = np.sum(t_obs)
        freq = Spec['frequency'][i]

        if freq == 'm':
            format_date = '%b %Y'
            frequency = 'Monthly'
        elif freq == 'q':
            format_date = 'Q%q %Y'
            frequency = 'Quarterly'

        units = Spec['units'][i]
        transform = Spec['transformation'][i]

        # display transformed units
        if 'Index' in units:
            units_transformed = 'Index'
        elif transform == 'chg':
            if '%' in units:
                units_transformed = 'Ppt. change'
            else:
                units_transformed = 'Level change'
        elif transform == 'pch' and freq == 'm':
            units_transformed = 'MoM%'
        elif transform == 'pca' and freq == 'q':
            units_transformed = 'QoQ% AR'
        else:
            units_transformed = f'{units} ({transform})' if len(f'{units} ({transform})') <= 12 else f'{units[:6]} ({transform})'

        t_obs_start = np.where(t_obs)[0][0]
        t_obs_end = np.where(t_obs)[0][-1]
        obs_date_start = pd.to_datetime(Time[t_obs_start]).strftime(format_date)
        obs_date_end = pd.to_datetime(Time[t_obs_end]).strftime(format_date)
        date_range = f'{obs_date_start}-{obs_date_end}'

        y = X[t_obs, i]
        d = Time[t_obs]
        mean_series = np.nanmean(y)
        stdv_series = np.nanstd(y)
        min_series, t_min = np.min(y), np.argmin(y)
        max_series, t_max = np.max(y), np.argmax(y)
        min_date = pd.to_datetime(d[t_min]).strftime(format_date)
        max_date = pd.to_datetime(d[t_max]).strftime(format_date)

        print(f'{data_series:30s} | {num_obs:17d} {units_transformed:12s} {frequency:10s} {mean_series:8.1f} {stdv_series:8.1f} {min_series:10.1f} {max_series:10.1f}')
        print(f'{series_id:30s} | {date_range:17s} {"":12s} {"":10s} {"":8s} {"":8s} {min_date:8s} {max_date:8s}')

    print('\n\n\n')

# Example usage:
# X = np.array(...)  # Your data array
# Time = np.array(...)  # Your time array
# Spec = {
#     'SeriesName': [...],
#     'SeriesID': [...],
#     'Frequency': [...],
#     'Units': [...],
#     'Transformation': [...]
# }
# vintage = '2024-08-08'
# summarize(X, Time, Spec, vintage)

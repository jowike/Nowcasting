def shift_to_fill_trailing_nans(dataframe):
    """
    Adjust columns in a DataFrame by shifting values to fill trailing NaN values with 
    a lagged version of the same series, if possible.
    
    Parameters:
    dataframe (pd.DataFrame): A DataFrame containing time series data in columns.
    
    Returns:
    pd.DataFrame: The modified DataFrame with adjusted columns.
    """
    # Iterate over each column in the DataFrame
    for column in dataframe.columns:
        series = dataframe[column]
        
        # Check if the column has missing values at the end
        if series.iloc[-1:].isna().all():
            # Find the index of the last non-NaN value
            last_valid_index = series.last_valid_index()
            
            # If a valid non-NaN index exists, create a lagged version of the series
            if last_valid_index is not None:
                lag_amount = len(series) - series.index.get_loc(last_valid_index) - 1
                dataframe[column] = series.shift(lag_amount)
    
    return dataframe

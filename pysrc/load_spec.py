import pandas as pd
import numpy as np

def load_spec(specfile):
    """
    Load model specification for a dynamic factor model (DFM).

    Parameters:
    - specfile: str, path to the Excel file containing the model specification.

    Returns:
    - spec: dict, containing the model specification.
    """

    # Read the Excel file
    raw_data = pd.read_excel(specfile, sheet_name=None, header=None)
    raw_data = list(raw_data.values())[0]  # Get the first sheet data if multiple sheets exist
    raw_data.columns = raw_data.iloc[0].str.replace(' ', '')
    raw_data = raw_data.drop(0)

    # Convert all headers to lowercase for consistency
    raw_data.columns = raw_data.columns.str.lower()

    # Find and drop series from Spec that are not in Model
    model_idx = raw_data['model'].astype(int) != 0
    raw_data = raw_data[model_idx]

    # Initialize spec dictionary
    spec = {}

    # Fields to extract from the Excel file
    field_names = ['seriesid', 'seriesname', 'frequency', 'units', 'transformation', 'category']
    for field in field_names:
        if field in raw_data.columns:
            spec[field] = raw_data[field].tolist()
        else:
            raise ValueError(f"{field} column missing from model specification.")

    # Parse blocks
    block_cols = [col for col in raw_data.columns if col.startswith('block')]
    blocks = raw_data[block_cols].fillna(0).astype(int).values
    if not np.all(blocks[:, 0] == 1):
        raise ValueError('All variables must load on global block.')
    spec['blocks'] = blocks

    # Sort all fields of 'Spec' in order of decreasing frequency
    frequency_order = ['d', 'w', 'm', 'q', 'sa', 'a']
    permutation = []
    for freq in frequency_order:
        permutation.extend(np.where(np.array(spec['frequency']) == freq)[0])

    for field in spec.keys():
        spec[field] = [spec[field][i] for i in permutation]

    # Extract block names from header
    spec['blocknames'] = [col.replace('block', '').replace('-', '') for col in block_cols]

    # Transformations
    transformation_map = {
        'lin': 'Levels (No Transformation)',
        'chg': 'Change (Difference)',
        'ch1': 'Year over Year Change (Difference)',
        'pch': 'Percent Change',
        'pc1': 'Year over Year Percent Change',
        'pca': 'Percent Change (Annual Rate)',
        'cch': 'Continuously Compounded Rate of Change',
        'cca': 'Continuously Compounded Annual Rate of Change',
        'log': 'Natural Log'
    }
    spec['unitstransformed'] = [
        transformation_map.get(trans, trans) for trans in spec['transformation']
    ]

    # Summarize model specification
    print('Table 1: Model specification')
    try:
        tabular = pd.DataFrame({
            'SeriesID': spec['seriesid'],
            'SeriesName': spec['seriesname'],
            'Units': spec['units'],
            'Transformation': spec['unitstransformed']
        })
        print(tabular)
    except Exception as e:
        print(f"Failed to display table: {e}")

    return spec

# Example usage:
# spec = load_spec('your_spec_file.xlsx')

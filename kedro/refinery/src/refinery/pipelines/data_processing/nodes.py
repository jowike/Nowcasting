# import pandas as pd


# def _is_true(x: pd.Series) -> pd.Series:
#     return x == "t"


# def _parse_percentage(x: pd.Series) -> pd.Series:
#     x = x.str.replace("%", "")
#     x = x.astype(float) / 100
#     return x


# def _parse_money(x: pd.Series) -> pd.Series:
#     x = x.str.replace("$", "").str.replace(",", "")
#     x = x.astype(float)
#     return x


# def preprocess_companies(companies: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
#     """Preprocesses the data for companies.

#     Args:
#         companies: Raw data.
#     Returns:
#         Preprocessed data, with `company_rating` converted to a float and
#         `iata_approved` converted to boolean.
#     """
#     companies["iata_approved"] = _is_true(companies["iata_approved"])
#     companies["company_rating"] = _parse_percentage(companies["company_rating"])
#     return companies, {"columns": companies.columns.tolist(), "data_type": "companies"}


# def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
#     """Preprocesses the data for shuttles.

#     Args:
#         shuttles: Raw data.
#     Returns:
#         Preprocessed data, with `price` converted to a float and `d_check_complete`,
#         `moon_clearance_complete` converted to boolean.
#     """
#     shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
#     shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
#     shuttles["price"] = _parse_money(shuttles["price"])
#     return shuttles


# def create_model_input_table(
#     shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
# ) -> pd.DataFrame:
#     """Combines all data to create a model input table.

#     Args:
#         shuttles: Preprocessed data for shuttles.
#         companies: Preprocessed data for companies.
#         reviews: Raw data for reviews.
#     Returns:
#         Model input table.

#     """
#     rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
#     rated_shuttles = rated_shuttles.drop("id", axis=1)
#     model_input_table = rated_shuttles.merge(
#         companies, left_on="company_id", right_on="id"
#     )
#     model_input_table = model_input_table.dropna()
#     return model_input_table

import sys

sys.path.append(
    "/Users/ejowik001/Desktop/Github/Nowcasting/kedro/refinery/src/refinery/pipelines/scripts"
)

import pandas as pd
from typing import List, Literal

from utils import _convert_to_datetime
from data_revisions import prepare_real_time_vintage_data
from load_spec import load_spec


def prepare_vintage_data(
    ds: pd.DataFrame, parameters: dict, spec_options: dict = None
) -> pd.DataFrame:
    """
    This function prepares retrospective dataset (vintage data) based on revision history
    """
    ds = _convert_to_datetime(
        ds, [parameters["ref_date_col"], parameters["pub_date_col"]]
    )

    if spec_options:
        Spec = load_spec(spec_options["filepath"])
        SeriesID, SeriesName, Units, UnitsTransformed, Frequency = (
            Spec["seriesid"],
            Spec["seriesname"],
            Spec["units"],
            Spec["unitstransformed"],
            Spec["frequency"],
        )
        df = ds.loc[ds[parameters["series_code_col"]].isin(SeriesID)]
    else:
        df = ds.copy()

    # df = df.loc[df[parameters["freq_desc_col"]].isin(parameters["freq_desc"])][parameters["cols"]]
    cols = [
        parameters["series_code_col"],
        parameters["ref_date_col"],
        parameters["pub_date_col"],
        parameters["series_val_col"],
        parameters["freq_desc_col"],
    ]
    df = df[cols].drop_duplicates()

    if parameters["vintage_name"] == "real-time":
        df_long = prepare_real_time_vintage_data(
            ds=df,
            y_code=parameters["y_code"],
            ref_date=parameters["ref_date"],
            series_code_col=parameters["series_code_col"],
            ref_date_col=parameters["ref_date_col"],
            pub_date_col=parameters["pub_date_col"],
            series_val_col=parameters["series_val_col"],
            freq_desc_col=parameters["freq_desc_col"],
        )
    return df_long

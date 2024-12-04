from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    prepare_vintage_data,
    suggest_spec,
    harmonize_ragged_edges,
    transform_time_series,
    test_variance,
    test_stationarity,
    apply_series_selection,
    ensemble_forecasts
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_vintage_data,
                inputs=[
                    "revision_history",
                    "params:options",
                    # "params:spec_options",
                ],
                outputs="vintage_data",
                name="prepare_vintage_data_node",
            ),
            node(
                func=suggest_spec,
                inputs=[
                    "revision_history",
                    "params:options",
                    ],
                outputs="ds_spec",
                name="build_vars_spec",
            ),
            node(
                func=harmonize_ragged_edges,
                inputs=[
                    "vintage_data",
                    "ds_spec",
                    "params:options",
                    ],
                outputs="harmonized_data",
                name="harmonize_ragged_edges_node",
            ),
            node(
                func=transform_time_series,
                inputs=[
                    "harmonized_data",
                    "ds_spec",
                    "params:options",
                    # "params:spec_options"
                    ],
                outputs=[
                    "transformed_aligned_data",
                    "aligned_non_transformed_data"
                    ],
                name="transform_time_series_node",
            ),
            node(
                func=test_variance,
                inputs=[
                    "transformed_aligned_data",
                    "params:options",
                    # "params:spec_options"
                    ],
                outputs="transformed_data_var",
                name="test_variance_node",
            ),
            node(
                func=test_stationarity,
                inputs=[
                    "transformed_data_var",
                    "params:options",
                    # "params:spec_options"
                    ],
                outputs="transformed_data_stat",
                name="test_stationarity_node",
            ),
            node(
                func=apply_series_selection,
                inputs=[
                    "transformed_data_var",
                    "params:options",
                    # "params:spec_options"
                    ],
                outputs="selected_series",
                name="select_series_node",
            ),
            node(
                func=ensemble_forecasts,
                inputs=[
                    "selected_series",
                    "params:options"
                    ],
                outputs=None,
                name="ensemble_forecasts_node",
            ),
        ]
    )

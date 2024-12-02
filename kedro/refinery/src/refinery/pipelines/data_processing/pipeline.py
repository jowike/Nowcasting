from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    prepare_vintage_data,
    build_spec_from_source,
    harmonize_ragged_edges,
    transform_time_series,
    reduce_features_by_variance_and_stationarity,
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
                func=build_spec_from_source,
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
                func=reduce_features_by_variance_and_stationarity,
                inputs=[
                    "transformed_aligned_data",
                    "params:options",
                    # "params:spec_options"
                    ],
                outputs="reduced_transformed_data",
                name="reduce_features_by_variance_and_stationarity_node",
            ),
            node(
                func=apply_series_selection,
                inputs=[
                    "reduced_transformed_data",
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

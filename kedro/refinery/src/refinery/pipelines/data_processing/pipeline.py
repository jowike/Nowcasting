from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    prepare_vintage_data,
    prepare_freq_details,
    harmonize_ragged_edges,
    transform_time_series,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_vintage_data,
                inputs=[
                    "revision_history",
                    "params:vintage_options",
                    "params:dataprep_options",
                    # "params:spec_options",
                ],
                outputs="vintage_data",
                name="prepare_vintage_data_node",
            ),
            node(
                func=prepare_freq_details,
                inputs=[
                    "revision_history",
                    "params:dataprep_options",
                    ],
                outputs="freq_details",
                name="prepare_freq_details_node",
            ),
            node(
                func=harmonize_ragged_edges,
                inputs=[
                    "vintage_data",
                    "freq_details",
                    "params:dataprep_options",
                    ],
                outputs="harmonized_data",
                name="harmonize_ragged_edges_node",
            ),
            node(
                func=transform_time_series,
                inputs=[
                    "harmonized_data",
                    "freq_details",
                    "params:vintage_options",
                    "params:dataprep_options",
                    # "params:spec_options"
                    ],
                outputs=[
                    "aligned_transformed_data",
                    "aligned_non_transformed_data"
                    ],
                name="transform_time_series_node",
            ),
        ]
    )

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    prepare_vintage_data,
    suggest_spec,
    harmonize_ragged_edges,
    transform_time_series,
    test_variance,
    test_stationarity,
    apply_series_selection,
    estimate_ml_node,
    estimate_arima_node,
    estimate_var_node,
    collect_results
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_vintage_data,
                inputs=[
                    "revision_history",
                    "params:options",
                    "params:spec_options",
                ],
                outputs=["vintage_data", "spec"],
                name="prepare_vintage_data_node",
            ),
            # node(
            #     func=suggest_spec,
            #     inputs=[
            #         "revision_history",
            #         "params:options",
            #         ],
            #     outputs="ds_spec",
            #     name="suggest_spec_node",
            # ),
            node(
                func=harmonize_ragged_edges,
                inputs=[
                    "vintage_data",
                    "spec",
                    "params:options",
                ],
                outputs="harmonized_data",
                name="harmonize_ragged_edges_node",
            ),
            node(
                func=transform_time_series,
                inputs=[
                    "harmonized_data",
                    "spec",
                    "params:options",
                ],
                outputs=["transformed_aligned_data", "aligned_non_transformed_data"],
                name="transform_time_series_node",
            ),
            node(
                func=test_variance,
                inputs=[
                    "transformed_aligned_data",
                    "spec",
                    "params:options",
                ],
                outputs="transformed_data_var",
                name="test_variance_node",
            ),
            node(
                func=test_stationarity,
                inputs=[
                    "transformed_data_var",
                    "spec",
                    "params:options",
                ],
                outputs="transformed_data_stat",
                name="test_stationarity_node",
            ),
            node(
                func=apply_series_selection,
                inputs=[
                    "transformed_data_stat",
                    "spec",
                    "params:options",
                ],
                outputs="selected_series",
                name="select_series_node",
            ),
            node(
                func=estimate_ml_node,
                inputs=[
                    "selected_series",
                    "aligned_non_transformed_data",
                    "spec",
                    "params:options",
                ],
                outputs="ml_estimation_results",
                name="estimate_ml_models_node",
            ),
            node(
                func=estimate_arima_node,
                inputs=[
                    "selected_series",
                    "aligned_non_transformed_data",
                    "spec",
                    "params:options",
                ],
                outputs="ar_estimation_results",
                name="estimate_arima_node",
            ),
            node(
                func=estimate_var_node,
                inputs=[
                    "selected_series",
                    "aligned_non_transformed_data",
                    "spec",
                    "params:options",
                ],
                outputs="var_estimation_results",
                name="estimate_var_node",
            ),
            node(
                func=collect_results,
                inputs=[
                    "spec",
                    "revision_history",
                    "aligned_non_transformed_data",
                    "params:options",
                    "ar_estimation_results",
                    "ml_estimation_results",
                    "var_estimation_results"
                ],
                outputs="dash_data_model",
                name="collect_results",
            ),
        ]
    )

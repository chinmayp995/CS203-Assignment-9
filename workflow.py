import mlrun
from kfp import dsl

@dsl.pipeline(name="breast-cancer-ci-pipeline")
def pipeline(model_name="rf_model"):
    # Step 1: Load data
    ingest = mlrun.run_function(
        "data-prep",
        name="data-loader",
        params={"format": "csv"},
        outputs=["dataset"]
    )

    # Step 2: Train with hyperparams
    train = mlrun.run_function(
        "trainer",
        inputs={"dataset": ingest.outputs["dataset"]},
        hyperparams={
            "n_estimators": [10, 100, 200],
            "max_depth": [2, 5, 10]
        },
        selector="max.accuracy",
        outputs=["model"]
    )

    # Step 3: Deploy
    mlrun.deploy_function(
        "serving",
        models=[{
            "key": model_name,
            "model_path": train.outputs["model"],
            "class_name": "RFModel"
        }],
        mock=True
    )

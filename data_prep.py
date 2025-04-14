import mlrun
import pandas as pd
from sklearn.datasets import load_breast_cancer

#defining functions and nlrun handler
@mlrun.handler(outputs=["dataset", "label_column"])
def cancer_loader(context, format="csv"):
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame
    df['target'] = cancer.target
    context.logger.info(f"Saving dataset to {context.artifact_path}")
    context.log_dataset("cancer_dataset", df=df, format=format, index=False)
    return df, "target"

#we now run the code  and upload the artifacts through the parameters
with mlrun.get_or_create_ctx("cancer_generator", upload_artifacts=True) as context:
        cancer_loader(context, context.get_param("format", "csv"))

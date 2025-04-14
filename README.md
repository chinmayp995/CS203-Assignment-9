# CS203-Assignment-9
## Team 20
### Dakshata Bhamare (23210027)
### Chinmay Pendse (23110245)

## Introduction

**Github Link:** https://github.com/chinmayp995/CS203-Assignment-9

In this Assignment, we deployed a training model using MLRun and learnt CI/CD for Machine Learning. We firstly  started by creating a Kubernetes cluster using the Docker  Desktop and then after the successful connection we downloaded Helm using the command 
```bash
winget install Helm.Helm 
```
and then ran all the given commands on terminal. Thereafter, we got our Jupyter notebook on http://localhost:30040. Then we referred the tutorial and got the necessry files of trainer.py, serving.py, workflow.py; And then we ran the main functioning code in assignment9.py. All the codes and results are being given in the Github repository. 

## Part 1. Create a MLRun Project

This screenshot is suggestive of our successful implementation of MLRun

![WhatsApp Image 2025-04-14 at 19 46 46_af47289c](https://github.com/user-attachments/assets/8aea28c6-43ce-4ca3-9f7d-75b1c890b652)


## Part 2.  Create the following Python script:

### a. data_prep.py
```python
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
```
### b. trainer.py
```python
import mlrun  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlrun.frameworks.sklearn import apply_mlrun

#calling ml hander and defining training function
@mlrun.handler(outputs=["model"])
def train(dataset: mlrun.DataItem, label_column="target", n_estimators=100, max_depth=5):
    df = dataset.as_df()

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset.")

    X = df.drop(columns=[label_column])
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    apply_mlrun(model=model, model_name="rf_model", x_test=X_test, y_test=y_test)
    model.fit(X_train, y_train)
```

### c. serving.py
```python
from cloudpickle import load
import numpy as np
import mlrun

class RFModel(mlrun.serving.V2ModelServer):
    def load(self):
        model_file, _ = self.get_model('.pkl')
        self.model = load(open(model_file, 'rb'))

    def predict(self, body: dict):
        feats = np.asarray(body['inputs'])
        results = self.model.predict(feats)
        return results.tolist()
```

### d. workflow.py
```python
import mlrun
from kfp import dsl

@dsl.pipeline(name="breast-cancer-ci-pipeline")
def pipeline(model_name="rf_model"):
    #loading the data  
    ingest = mlrun.run_function(
        "data-prep",
        name="data-loader",
        params={"format": "csv"},
        outputs=["dataset"]
    )

    # we now use the lists of different hyperparamters whch will be deployed
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
    #deploeing the model
    mlrun.deploy_function(
        "serving",
        models=[{
            "key": model_name,
            "model_path": train.outputs["model"],
            "class_name": "RFModel"
        }],
        mock=True
    )
```

## Part 3. project.run Implementation and Results

Some of the screenshots may not be visible in the jupyter notebook So, pls refer colab file at
https://colab.research.google.com/drive/15XR4ZdK-PJievYamWlkKUwN0db4pQL3G?usp=sharing

**There are many other features and paramters which are present in the colab file; where we can specifically choose them.** Here I have only kept the imporatnt things.
### Workflow diagram

![WhatsApp Image 2025-04-14 at 19 52 32_f2253c4a](https://github.com/user-attachments/assets/deb373cb-481e-4ee0-ba30-e45d2e990375)

### Data Prep Artifacts and dataset

![Screenshot 2025-04-14 214735](https://github.com/user-attachments/assets/b8cb7294-7eb9-4e37-919e-826576ad29fe)

![Screenshot 2025-04-14 214829](https://github.com/user-attachments/assets/4f72ab61-58bc-4c33-bb2d-0f63a889eb52)

![Screenshot 2025-04-14 214906](https://github.com/user-attachments/assets/5840b114-3304-44e9-89a7-0f03b0080927)


###  Confusion-matrix artifact of train.py

![Screenshot 2025-04-14 215034](https://github.com/user-attachments/assets/59474e9c-f53a-45da-8442-5e738173abdb)

![Screenshot 2025-04-14 215214](https://github.com/user-attachments/assets/421cfa95-7edf-41bb-a255-e9601ba88279)

![Screenshot 2025-04-14 215237](https://github.com/user-attachments/assets/7d40234d-2793-4b4d-9d5d-0b53f012ef8e)


### Feature Importance Artifact

![Screenshot 2025-04-14 215411](https://github.com/user-attachments/assets/e2a303ca-1382-453a-8cdd-e5d0d51c9b47)

![Screenshot 2025-04-14 215532](https://github.com/user-attachments/assets/1c60e16e-1ab7-44b3-b047-630b471fa821)

![Screenshot 2025-04-14 215548](https://github.com/user-attachments/assets/d4a7c320-8db0-47a9-bd7a-881ef328125b)


![Screenshot 2025-04-14 215606](https://github.com/user-attachments/assets/f2c8ca0f-3ce8-4949-a231-f2ba1fca8e93)


### Hyperparamters

![Screenshot 2025-04-14 215726](https://github.com/user-attachments/assets/1768282e-4a0e-4b98-8791-878675baa265)

![Screenshot 2025-04-14 215756](https://github.com/user-attachments/assets/74ba2010-329b-41a0-bb93-be5c2835e0fe)

![Screenshot 2025-04-14 215811](https://github.com/user-attachments/assets/6eb524e4-0e86-431b-bcfe-76cd2ca49bca)

#### Accuracies

![Screenshot 2025-04-14 215839](https://github.com/user-attachments/assets/dcd53ada-46d2-4c76-8683-279bb13b392b)

#### F1 score

![Screenshot 2025-04-14 215910](https://github.com/user-attachments/assets/b85e5a44-c458-4913-85b8-05f967158d27)


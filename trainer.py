import mlrun  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlrun.frameworks.sklearn import apply_mlrun

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

    

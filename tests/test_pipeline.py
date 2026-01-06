import pandas as pd
import joblib

model = joblib.load("best_random_forest_pipeline.joblib")

def test_preprocessing_imputes():
    row = {
        "age": 55,
        "sex": "Male",
        "cp": "atypical angina",
        "trestbps": None,
        "chol": None,
        "fbs": None,
        "restecg": None,
        "thalch": None,
        "exang": None,
        "oldpeak": None,
        "slope": None,
        "ca": None,
        "thal": None
    }
    df = pd.DataFrame([row])
    pred = model.predict(df)
    assert len(pred) == 1

def test_predict_output_format():
    row = {
        "age": 60,
        "sex": "Male",
        "cp": "typical angina",
        "trestbps": 140,
        "chol": 239,
        "fbs": 0,
        "restecg": "normal",
        "thalch": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": "upsloping",
        "ca": 0,
        "thal": "normal"
    }
    df = pd.DataFrame([row])
    proba = model.predict_proba(df)[:,1]
    assert len(proba) == 1
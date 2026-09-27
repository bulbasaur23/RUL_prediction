import joblib 
import pandas as pd
import os
import xgboost as xgb
# os.path.dirname walks up from app/predict.py ---> app/ -->project root
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# joins the BASE_DIR to models/ .pkl files 
MODEL_PATH=os.path.join(BASE_DIR,"models","xgb_model.json")
FEATURES_PATH=os.path.join(BASE_DIR,"models","features.pkl")

model=xgb.Booster()
model.load_model(MODEL_PATH)
features=joblib.load(FEATURES_PATH)

def predict_rul(sensor_data:dict) -> float:

    row=pd.DataFrame([sensor_data])[features]
    dmatrix=xgb.DMatrix(row)
    pred=model.predict(dmatrix)[0]
    return float(pred)

if __name__=="__main__":
    sample={f:0.0 for f in features}
    print(predict_rul(sample))

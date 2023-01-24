from fastapi import FastAPI, Request
from keras.models import load_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import uvicorn

app = FastAPI()

models = {}


@app.on_event("startup")
async def startup_event():
    models["lstm"] = load_model('model/mv_lstm_model_3_week.h5')
    models["scaler"] = pickle.load(open('model/scaler_3_week.pkl', 'rb'))

def fill_missing_values(df):
    imputer = IterativeImputer(random_state=42, max_iter=50)
    imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(imputed, columns=df.columns, index=df.index)
    return df_imputed

def split_series(series, n_past, n_future):
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)

@app.post("/predict")
async def predict(features: Request):
    logger.info("Data received")
    logger.debug("Processing data..")
    body_data = await features.json()
    df = pd.DataFrame(body_data)
    
    df.sort_values(by=['time'], ascending=True, inplace=True)
    df.set_index(['time'], inplace=True)
    
    df_imputed = fill_missing_values(df)
    
    df_imputed = df_imputed.drop(df_imputed.columns.difference(['rel. hum.',
                                                                'mean rel. hum.',
                                                                'min rel. hum.',
                                                                'max rel. hum.']), axis = 1)
    train = df_imputed

    scalers = {}

    for i in df_imputed.columns:
        scaler = MinMaxScaler(feature_range = (0,1))
        s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
        s_s = np.reshape(s_s, len(s_s))
        scalers['scaler_' + i] = scaler
        train[i] = s_s
    
    X_test, y_test = split_series(train.values, 3024, 144)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 4))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 4))
    
    logger.debug("Data processing finished")
    
    logger.info("Predicting relative humidity - Please be patient..")
    
    prediction = models["lstm"].predict(X_test)
    prediction = prediction.reshape(prediction.shape[0], prediction.shape[1] * prediction.shape[2])
    prediction = models["scaler"].inverse_transform(prediction)

    logger.info("Prediction successful, serving response..")
    return {"prediction": float("{}.0".format(int(round(float(prediction[0][0]), 1))))}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
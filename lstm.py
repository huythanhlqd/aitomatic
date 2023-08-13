import streamlit as st 
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import numpy as np
from datetime import datetime

n_classes = 2
def create_model():
    inputs = layers.Input(shape=(24,17), name='input')

    x1 = layers.Bidirectional(layers.LSTM(units=50,return_sequences=False))(inputs)
    x2 = layers.Bidirectional(layers.LSTM(units=50,return_sequences=False))(inputs)
    x2 = layers.Dropout(0.2)(x2)
    outputs = layers.Dense(n_classes, activation="softmax")(x2)
    return keras.models.Model(inputs=inputs, outputs=outputs)

@st.cache_resource
def load_model():
    print('load model')
    model = create_model()
    model.load_weights("cp.ckpt")
    return model

df_machineID = pd.read_csv("machineID_origin.csv",index_col=0)
df_age = pd.read_csv("age_origin.csv",index_col=0)
df_model = pd.read_csv("model_origin.csv",index_col=0)
numerical_cols = ['volt', 'rotate', 'pressure', 'vibration']
mean_cols = ['mean_volt', 'mean_rotate', 'mean_pressure', 'mean_vibration']
std_cols = ['std_volt', 'std_rotate', 'std_pressure', 'std_vibration']
# rank_cols = ['rank_volt', 'rank_rotate', 'rank_pressure', 'rank_vibration']
columns = list(np.arange(12)) + numerical_cols + ['machineID'] #+ rank_cols

def predict(model, df):
    machine_id = df.iloc[0]['machineID']
    age  = df.iloc[0]['age']
    mdel = df.iloc[0]['model']
    df[np.arange(4)] = (df[numerical_cols].values - df_machineID.loc[machine_id,mean_cols].values) / df_machineID.loc[machine_id,std_cols].values
    df[np.arange(4,8)] = (df[numerical_cols].values - df_age.loc[age,mean_cols].values) / df_age.loc[age,std_cols].values
    df[np.arange(8,12)] = (df[numerical_cols].values - df_model.loc[mdel,mean_cols].values) / df_model.loc[mdel,std_cols].values
    df[numerical_cols] = ((df[numerical_cols] - df[numerical_cols].mean(0)) / df[numerical_cols].std(0)).astype(int)
    df['machineID'] = df_machineID.loc[machine_id,'machineID_rate']
    # df[rank_cols] = df[np.arange(4)].rank()
    input = df[columns].values.astype(np.float32)
    print(input)
    return model.predict(input[np.newaxis,...])[0]
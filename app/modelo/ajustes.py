import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def ajuste_lineal(df_num):
    df_ajuste = df_num.astype(float).copy()
    for col in df_ajuste.columns:
        x_min = df_ajuste[col].min()
        x_max = df_ajuste[col].max()
        if pd.isna(x_min) or pd.isna(x_max):
            df_ajuste[col] = 0.0
        elif x_max == x_min:
            df_ajuste[col] = 0.0
        else:
            df_ajuste[col] = (df_ajuste[col] - x_min) / (x_max - x_min)
    return df_ajuste

def ajuste_z(df_num):
    df_ajuste = df_num.astype(float).copy()
    for col in df_ajuste.columns:
        media = df_ajuste[col].mean()
        sigma = df_ajuste[col].std(ddof=0)
        if sigma == 0 or pd.isna(sigma):
            df_ajuste[col] = df_ajuste[col] - media
        else:
            df_ajuste[col] = (df_ajuste[col] - media) / sigma
    return df_ajuste

def ajuste_log(df_num):
    df_ajuste = df_num.astype(float).copy()
    for col in df_ajuste.columns:
        minimo = df_ajuste[col].min()
        if pd.isna(minimo):
            df_ajuste[col] = 0.0
            continue
        if minimo <= 0:
            df_ajuste[col] = df_ajuste[col] - minimo + 1
        df_ajuste[col] = np.log(df_ajuste[col])
    return df_ajuste

def aplicar_ajuste(df_num, tipo_ajuste):
    if df_num.empty:
        return df_num

    if tipo_ajuste == "lineal":
        return ajuste_lineal(df_num)
    elif tipo_ajuste == "z":
        return ajuste_z(df_num)
    elif tipo_ajuste == "log":
         return ajuste_log(df_num) 
    else:
        return df_num.astype(float).copy()
    
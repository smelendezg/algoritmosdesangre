import pandas as pd
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kmodes.kmodes import KModes
from .ajustes import aplicar_ajuste

def k_por_defecto(n_filas: int) -> int:
    try:
        n = int(n_filas)
    except:
        n = 2
    return max(2, min(8, int(math.sqrt(max(n, 1)))))


def ejecutar_kmedias(df: pd.DataFrame, tipo_ajuste: str, k: int):
    num_df = df.select_dtypes(include="number").copy()
    if num_df.empty:
        raise ValueError("No hay columnas numéricas en el archivo para K-Medias.")

    nan_mask = num_df.isna()

    num_df_imputado = num_df.fillna(num_df.mean())

    num_df_ajustado = aplicar_ajuste(num_df_imputado, tipo_ajuste)
    X = num_df_ajustado.values

    modelo = KMeans(n_clusters=k, random_state=42, n_init=2)
    etiquetas = modelo.fit_predict(X)

    df_agrupado = df.copy()
    df_agrupado["grupo"] = etiquetas

    if nan_mask.any().any():
        num_df_con_grupo = num_df.copy()
        num_df_con_grupo["grupo"] = etiquetas

        medias_cluster = num_df_con_grupo.groupby("grupo").mean()

        for idx, row in df_agrupado[nan_mask.any(axis=1)].iterrows():
            g = row["grupo"]
            for col in num_df.columns:
                if pd.isna(row[col]):
                    df_agrupado.at[idx, col] = medias_cluster.loc[g, col]

    return {
        'df': df_agrupado,
        'k': k,
        'tipo_ajuste': tipo_ajuste
    }


def ejecutar_kmodas(df: pd.DataFrame, k: int):
    df_categ = df.copy()
    nan_mask = df_categ.isna()

    df_para_cluster = df_categ.fillna("__DESCONOCIDO__").astype(str)

    km = KModes(n_clusters=k, init="Huang", n_init=5, verbose=0)
    etiquetas = km.fit_predict(df_para_cluster)

    df_agrupado = df_categ.copy()
    df_agrupado["grupo"] = etiquetas

    if nan_mask.any().any():
        df_con_grupo = df_agrupado.copy()

        for col in df_categ.columns:
            if col == "grupo":
                continue

            modas = df_con_grupo.groupby("grupo")[col].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else None
            )

            filas_col_faltante = nan_mask[col]
            for idx, row in df_agrupado[filas_col_faltante].iterrows():
                g = row["grupo"]
                valor = modas.loc[g]
                df_agrupado.at[idx, col] = valor

    return {
        'df': df_agrupado,
        'k': k
    }

def kmedias_por_clase(df: pd.DataFrame, tipo_ajuste: str, k: int, col_grupo: str, col_valor: str):
    df_agrupado = df.copy()

    grupos = df_agrupado[col_grupo].astype(str)
    codigos, etiquetas = pd.factorize(grupos)
    df_agrupado["grupo"] = codigos
    k = len(etiquetas)

    cols_num = [col_valor]

    num_df = df_agrupado[cols_num].astype(float)
    filas_vacias = num_df.isna()

    medias = df_agrupado.groupby("grupo")[cols_num].mean()

    for idx, row in df_agrupado[filas_vacias.any(axis=1)].iterrows():
            g = row["grupo"]
            for col in cols_num:
                if pd.isna(row[col]):
                    df_agrupado.at[idx, col] = medias.loc[g, col]

    num_df_ajustado = aplicar_ajuste(df_agrupado[cols_num], tipo_ajuste)
    for col in cols_num:
        df_agrupado[col] = num_df_ajustado[col]

    return {
       'df': df_agrupado,
        'k': k,
        'tipo_ajuste': tipo_ajuste,
        'col_grupo_original': col_grupo,
        'labels_grupo': list(etiquetas)             
    }



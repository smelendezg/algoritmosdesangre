import pandas as pd
import numpy as np
from scipy.stats import chi2

def chi_merge(df, col_atributo, col_clase, umbral=None, alpha=0.05):
    datos = df[[col_atributo, col_clase]].dropna().copy()
    clases = sorted(datos[col_clase].unique())
    n_clases = len(clases)

    if n_clases == 0:
        raise ValueError("No hay clases para Chi-Merge.")

    if umbral is None:
        df_chi = n_clases - 1
        umbral = chi2.ppf(1 - alpha, df_chi)

    intervalos = _crear_intervalos_iniciales(datos, col_atributo, col_clase, clases)
    intervalos = _fusionar_intervalos(intervalos, umbral)
    df_resultado = _asignar_etiquetas(df, col_atributo, intervalos)

    return df_resultado, intervalos, umbral, col_atributo, col_clase

def _crear_intervalos_iniciales(datos, col_atributo, col_clase, clases):
    agrupado = datos.groupby(col_atributo)[col_clase].value_counts().unstack(fill_value=0)
    agrupado = agrupado.reindex(columns=clases, fill_value=0).sort_index()

    intervalos = []
    for valor, fila in agrupado.iterrows():
        intervalos.append({
            "min": valor,
            "max": valor,
            "conteo": fila.values.astype(float)
        })    
    return intervalos    


def _fusionar_intervalos(intervalos, umbral):
    fusion = True
    while fusion and len(intervalos) > 1:
        fusion = False
        
        chis = []
        for i in range(len(intervalos) - 1):
            chi_val = _calcular_chi2(intervalos[i], intervalos[i + 1])
            chis.append((chi_val, i))

        chis.sort(key=lambda x: x[0])
        chi_min, idx = chis[0]

        if chi_min < umbral:
            a = intervalos[idx]
            b = intervalos[idx + 1]
            nuevo = {
                "min": a["min"],
                "max": b["max"],
                "conteo": a["conteo"] + b["conteo"]
            }
            intervalos[idx:idx + 2] = [nuevo]
            fusion = True

    return intervalos


def _calcular_chi2(intervalo1, intervalo2):
    try:
        obs = np.vstack([intervalo1["conteo"], intervalo2["conteo"]])
        fila_tot = obs.sum(axis=1, keepdims=True)
        col_tot = obs.sum(axis=0, keepdims=True)
        total = obs.sum()
        
        if total == 0:
            return 0.0
        
        esperado = fila_tot @ col_tot / total
        esperado[esperado == 0] = 1e-9
        
        return ((obs - esperado) ** 2 / esperado).sum()
    except:
        return 0.0


def _asignar_etiquetas(df, col_atributo, intervalos):
    etiquetas = []
    etiquetas_intervalo = []
    
    for _, fila in df.iterrows():
        val = fila[col_atributo]
        asignado = False
        
        for i, intervalo in enumerate(intervalos):
            if intervalo["min"] <= val <= intervalo["max"]:
                etiquetas.append(i)
                etiquetas_intervalo.append(f"[{intervalo['min']}, {intervalo['max']}]")
                asignado = True
                break
        
        if not asignado:
            etiquetas.append(-1)
            etiquetas_intervalo.append("sin_intervalo")

    df_resultado = df.copy()
    df_resultado["chi_cluster"] = etiquetas
    df_resultado["intervalo"] = etiquetas_intervalo
    
    return df_resultado